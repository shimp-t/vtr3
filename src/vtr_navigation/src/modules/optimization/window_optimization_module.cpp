
#include <asrl/steam_extensions/evaluator/common/RangeConditioningEval.hpp>
#include <asrl/steam_extensions/evaluator/common/ScaleErrorEval.hpp>

#include <asrl/common/timing/SimpleTimer.hpp>

#include <asrl/messages/TrajectoryStatus.pb.h>
#include <asrl/messages/lgmath_conversions.hpp>

#include <asrl/vision/Types.hpp>
#include <asrl/vision/geometry/geometry_tools.hpp>
#include <asrl/vision/messages/bridge.hpp>

#include <asrl/navigation/modules/optimization/window_optimization_module.h>

namespace asrl {
namespace navigation {

void WindowOptimizationModule::setConfig(std::shared_ptr<Config> &config) {
  // Set the base module
  auto down_casted_config =
      std::dynamic_pointer_cast<SteamModule::Config>(config);
  SteamModule::setConfig(down_casted_config);
  config_ = config;
}

std::shared_ptr<steam::OptimizationProblem>
WindowOptimizationModule::generateOptimizationProblem(
    QueryCache &qdata, MapCache &mdata,
    const std::shared_ptr<const Graph> &graph) {
  // get references to the relevent data.
  LandmarkMap &lm_map = *mdata.landmark_map;
  auto &poses = *mdata.pose_map;
  auto &tsv_transforms = *mdata.T_sensor_vehicle_map;
  const auto &calibrations = *qdata.rig_calibrations;
  auto calibration_itr = calibrations.begin();

  // reset to remove any old data from the problem setup
  resetProblem();

  // monocular or stereo?
  bool monocular = calibration_itr->intrinsics.size() == 1 ? true : false;

  // get calibration for this rig.
  auto &calibration = *calibration_itr;

  // Setup camera intrinsics, TODO: This should eventually be different for each
  // rig.
  StereoCalibPtr sharedStereoIntrinsics;
  MonoCalibPtr sharedMonoIntrinsics;

  // setup the calibration
  if (monocular) {
    sharedMonoIntrinsics = toMonoSteamCalibration(calibration);
  } else {
    sharedStereoIntrinsics = toStereoSteamCalibration(calibration);
  }

  // Go through all of the landmarks
  for (auto &landmark : lm_map) {
    // 1. get the pose associated with this map
    auto vertex = graph->fromPersistent(
        messages::copyPersistentId(landmark.first.persistent));
    auto &lm_pose = poses[vertex];

    // Extract the point associated with this landmark
    Eigen::Vector3d lm_point(landmark.second.point->x(),
                             landmark.second.point->y(),
                             landmark.second.point->z());

    // Extract the validity of the landmark
    bool map_point_valid = *landmark.second.valid;

    // If the point and its depth is valid, then add it as a landmark.
    if (map_point_valid && isLandmarkValid(lm_point, mdata) == true &&
        landmark.second.observations.size() > 1) {
      landmark.second.steam_lm.reset(
          new steam::se3::LandmarkStateVar(lm_point));
      auto &steam_lm = landmark.second.steam_lm;

      // set the lock only if the map is initialized
      // if(*mdata.map_initialized == true) {
      steam_lm->setLock(lm_pose.isLocked());
      //}

      // add the depth prior
      if (config_->depth_prior_enable) {
        addDepthCost(steam_lm);
      }
    } else {
      landmark.second.steam_lm.reset();
      continue;
    }

    // get a reference to our new landmark.
    auto &steam_lm = landmark.second.steam_lm;
    // 3. go through the observations, and add cost terms for each.
    for (LandmarkObs &obs : landmark.second.observations) {
      try {  // steam throws?

        // get the keypoints for this observation.
        auto obs_vertex =
            graph->fromPersistent(obs.origin_ref->from().persistent());
        auto &obs_pose = poses[obs_vertex];

        // Set up the transform evaluator (i.e. The transform that takes the
        // landmark into the observation frame)
        steam::se3::TransformEvaluator::Ptr pose_obs_lm;

        // if we are observing the landmark in the parent frame, then use
        // identity.
        if (obs_vertex == vertex) {
          pose_obs_lm = tf_identity_;
        } else {
          // otherwise compose the transform
          steam::se3::TransformEvaluator::Ptr pose_lm_0 =
              steam::se3::TransformStateEvaluator::MakeShared(
                  lm_pose.tf_state_var);
          steam::se3::TransformEvaluator::Ptr pose_obs_0 =
              steam::se3::TransformStateEvaluator::MakeShared(
                  obs_pose.tf_state_var);
          pose_obs_lm = steam::se3::composeInverse(pose_obs_0, pose_lm_0);
        }

        // make a pointer to the landmark->observation transform
        steam::se3::TransformEvaluator::Ptr T_obs_lm;

        // Compose with non-fixed camera to vehicle transform
        auto T_s_v_obs_ptr = tsv_transforms.find(obs_vertex);
        auto T_s_v_ptr = tsv_transforms.find(vertex);

        // check if we have pre-loaded the transforms
        if (T_s_v_ptr == tsv_transforms.end() ||
            T_s_v_obs_ptr == tsv_transforms.end()) {
          // no, compose with fixed camera to vehicle transform
          LOG(WARNING) << "Couldn't find transform! for either " << obs_vertex
                       << " or " << vertex;
          T_obs_lm = steam::se3::composeInverse(
              steam::se3::compose(tf_sensor_vehicle_, pose_obs_lm),
              tf_sensor_vehicle_);
        } else {
          // yes, create the non-static (but fixed for this optimisation)
          // sensor->vehicle transform have we created this specific transform
          // for *vertex* before?
          auto composed_T_s_v_fixed_ptr = tf_sensor_vehicle_map_.find(vertex);
          if (composed_T_s_v_fixed_ptr == tf_sensor_vehicle_map_.end()) {
            // we haven't, make it
            tf_sensor_vehicle_map_[vertex] =
                steam::se3::FixedTransformEvaluator::MakeShared(
                    T_s_v_ptr->second);
            // this should now get the correct reference
            composed_T_s_v_fixed_ptr = tf_sensor_vehicle_map_.find(vertex);
          }
          // have we created this specific transform for *obs_vertex* before?
          auto composed_T_s_v_fixed_obs_ptr =
              tf_sensor_vehicle_map_.find(obs_vertex);
          if (composed_T_s_v_fixed_obs_ptr == tf_sensor_vehicle_map_.end()) {
            // we haven't, make it
            tf_sensor_vehicle_map_[obs_vertex] =
                steam::se3::FixedTransformEvaluator::MakeShared(
                    T_s_v_obs_ptr->second);
            // this should now get the correct reference
            composed_T_s_v_fixed_obs_ptr =
                tf_sensor_vehicle_map_.find(obs_vertex);
          }
          // ok...., now we can compose the transform evaluator. Compose with
          // non-fixed camera to vehicle transform
          T_obs_lm = steam::se3::composeInverse(
              steam::se3::compose(composed_T_s_v_fixed_obs_ptr->second,
                                  pose_obs_lm),
              composed_T_s_v_fixed_ptr->second);
        }

        // set up the mono and stereo noise for each potential type
        steam::BaseNoiseModelX::Ptr noise_mono;
        steam::BaseNoiseModel<4>::Ptr noise_stereo;

        // set up the measurement covariance vector
        unsigned m_sz = monocular ? 2 : 4;
        Eigen::MatrixXd meas_cov(m_sz, m_sz);
        meas_cov.setZero();

        // add the measurement covariances from the stored memory
        unsigned idx = 0;
        for (auto &cov : obs.covariances) {
          meas_cov(2 * idx, 2 * idx) = cov[0];
          meas_cov(2 * idx + 1, 2 * idx + 1) = cov[3];
          idx++;
        }

        // add to the noise models
        if (monocular) {
          noise_mono.reset(new steam::StaticNoiseModelX(meas_cov));
        } else {
          noise_stereo.reset(new steam::StaticNoiseModel<4>(meas_cov));
        }

        // Construct the measurement vector for the current camera
        Eigen::MatrixXd data(obs.keypoints.size() * 2, 1);
        for (uint32_t idx = 0; idx < obs.keypoints.size(); idx++) {
          data(idx * 2) = obs.keypoints.at(idx)->position().x();
          data(idx * 2 + 1) = obs.keypoints.at(idx)->position().y();
        }

        if (monocular) {
          // Construct error function for the current camera
          steam_extensions::MonoCameraErrorEval::Ptr errorfunc(
              new steam_extensions::MonoCameraErrorEval(
                  data, sharedMonoIntrinsics, T_obs_lm, steam_lm));
          // Construct cost term for the current camera
          steam::WeightedLeastSqCostTermX::Ptr cost(
              new steam::WeightedLeastSqCostTermX(errorfunc, noise_mono,
                                                  sharedLossFunc_));

          // finally, add the cost.
          cost_terms_->add(cost);
        } else {
          // Construct error function for the current camera
          steam::StereoCameraErrorEval::Ptr errorfunc(
              new steam::StereoCameraErrorEval(data, sharedStereoIntrinsics,
                                               T_obs_lm, steam_lm));
          // Construct cost term for the current camera
          steam::WeightedLeastSqCostTerm<4, 6>::Ptr cost(
              new steam::WeightedLeastSqCostTerm<4, 6>(errorfunc, noise_stereo,
                                                       sharedLossFunc_));

          // finally, add the cost.
          cost_terms_->add(cost);
        }

        // steam throws?
      } catch (std::exception &e) {
        LOG(ERROR) << "Error with noise model:\n" << e.what();
        continue;
      }
    }
  }

  // we need to add a scaling factor if we are using a monocular scheme
  double max_d = 0;
  steam::se3::TransformStateEvaluator::Ptr max_d_tf_state_eval;

  // find the most distant pose from the origin
  if (monocular) {
    for (auto &pose : poses) {
      // if there are poses from other runs in the window, we don't need to add
      // the scale cost
      if (pose.first.majorId() != qdata.live_id->majorId()) {
        max_d_tf_state_eval = nullptr;
        max_d = -1.0;
        break;
      }
      auto &steam_pose = pose.second;
      // get the norm of the translation to find the distance from the origin
      double d =
          steam_pose.tf_state_var->getValue().matrix().col(3).topRows(3).norm();
      // is this the most distant?
      if (d > max_d) {
        max_d = d;
        max_d_tf_state_eval = steam_pose.tf_state_eval;
      }
    }

    // if we have found a pose
    if (max_d_tf_state_eval != nullptr && max_d > 0) {
      // make a squared loss term (we don't want this to be marginalised
      steam::LossFunctionBase::Ptr scaleLossFunc;
      scaleLossFunc.reset(new steam::L2LossFunc());

      // make the uncertainty for the scale error really small
      steam::BaseNoiseModelX::Ptr scaleUncertainty;
      scaleUncertainty.reset(new steam::StaticNoiseModelX(
          Eigen::Matrix<double, 1, 1>::Identity()));

      // make the scale error evaluator from the original translational norm
      asrl::steam_extensions::ScaleErrorEval::Ptr scale_error_func(
          new asrl::steam_extensions::ScaleErrorEval(max_d,
                                                     max_d_tf_state_eval));

      // Create cost term and add to problem
      steam::WeightedLeastSqCostTermX::Ptr scale_cost(
          new steam::WeightedLeastSqCostTermX(scale_error_func,
                                              scaleUncertainty, scaleLossFunc));
      cost_terms_->add(scale_cost);
    }
  }

  // add pose variables
  int jj = 0;
  for (auto &pose : poses) {
    auto &steam_pose = pose.second;
    problem_->addStateVariable(steam_pose.tf_state_var);
    jj++;
  }

  // Add landmark variables
  for (auto &landmark : lm_map) {
    if (landmark.second.steam_lm != nullptr &&
        (*landmark.second.valid) == true) {
      problem_->addStateVariable(landmark.second.steam_lm);
    }
  }

  problem_->addCostTerm(cost_terms_);
  if (config_->depth_prior_enable) {
    problem_->addCostTerm(depth_cost_terms_);
  }

  // add trajectory stuff
  if (config_->trajectory_smoothing == true) {
    // reset the trajectory
    trajectory_.reset(new steam::se3::SteamTrajInterface(
        smoothing_factor_information_, true));
    bool prior_added = false;
    for (auto &pose : poses) {
      auto &steam_pose = pose.second;
      if (steam_pose.velocity == nullptr) {
        LOG(ERROR) << "Trajectory velocity was null!";
        continue;
      }
      trajectory_->add(steam_pose.time, steam_pose.tf_state_eval,
                       steam_pose.velocity);
      problem_->addStateVariable(steam_pose.velocity);
      if (config_->velocity_prior == true && steam_pose.isLocked() == false &&
          prior_added == false) {
        trajectory_->addVelocityPrior(steam_pose.time, velocity_prior_,
                                      velocity_prior_cov_);
      }
    }

    // Add cost terms
    trajectory_->appendPriorCostTerms(cost_terms_);
    if (config_->depth_prior_enable) {
      trajectory_->appendPriorCostTerms(depth_cost_terms_);
    }
  }

  return problem_;
}

/// @brief Initializes the problem_ based on an initial condition.
/// @param The initial guess at the transformation between the query frame and
/// the map frame.
void WindowOptimizationModule::resetProblem() {
  // make the depth loss function
  sharedDepthLossFunc_.reset(new steam::DcsLossFunc(2.0));

  // make the loss function, TODO: make this configurable, move to member var.
  sharedLossFunc_.reset(new steam::DcsLossFunc(2.0));

  // setup cost terms
  cost_terms_.reset(new steam::ParallelizedCostTermCollection());

  // setup cost terms for the depth
  depth_cost_terms_.reset(new steam::ParallelizedCostTermCollection());

  // set up the steam problem_.
  problem_.reset(new steam::OptimizationProblem());
}

/// @brief Adds a depth cost associated with this landmark to the depth cost
/// terms.
/// @param The landmark in question.
void WindowOptimizationModule::addDepthCost(
    steam::se3::LandmarkStateVar::Ptr landmark) {
  asrl::steam_extensions::RangeConditioningEval::Ptr errorfunc_range(
      new asrl::steam_extensions::RangeConditioningEval(landmark));
  double depth = landmark->getValue().hnormalized()[2];
  double weight = config_->depth_prior_weight / depth;
  steam::BaseNoiseModel<1>::Ptr rangeNoiseModel(new steam::StaticNoiseModel<1>(
      Eigen::Matrix<double, 1, 1>::Identity() * weight));
  steam::WeightedLeastSqCostTerm<1, 3>::Ptr depth_cost(
      new steam::WeightedLeastSqCostTerm<1, 3>(errorfunc_range, rangeNoiseModel,
                                               sharedDepthLossFunc_));
  depth_cost_terms_->add(depth_cost);
}

bool WindowOptimizationModule::verifyInputData(QueryCache &qdata,
                                               MapCache &mdata) {
  // make sure we have a landmark and pose map, and calibration.
  if (mdata.landmark_map.is_valid() == false ||
      mdata.pose_map.is_valid() == false ||
      qdata.rig_calibrations.is_valid() == false) {
    LOG(ERROR) << "WindowOptimizationModule::verifyInputData(): Input data for "
                  "windowed BA problem is not set! (Is the Windowed Recall "
                  "Module Running?)";
    return false;
  }

  // If there is nothing to optimize, then quit.
  if ((*mdata.pose_map).empty() || (*mdata.landmark_map).empty()) {
    LOG(ERROR) << "WindowOptimizationModule::verifyInputData(): No poses or "
                  "landmarks found. (Is the Windowed Recall Module Running?)";
    return false;
  } else
    return true;
}

bool WindowOptimizationModule::isLandmarkValid(const Eigen::Vector3d &point,
                                               MapCache &mdata) {
  // check depth
  if (point(2) > config_->max_point_depth ||
      point(2) < config_->min_point_depth) {
    return false;
  }

  // check the distance from the plane
  if (config_->perform_planarity_check &&
      mdata.plane_coefficients.is_valid() == true) {
    // estimate the distance of the point from the plane
    double dist = vision::estimatePlaneDepth(point, *mdata.plane_coefficients);

    // if it is beyond the maximum depth, it's invalid
    if (dist > config_->plane_distance) {
      return false;
    }
  }

  // if we got here, the landmark is valid
  return true;
}

bool WindowOptimizationModule::verifyOutputData(QueryCache &qdata,
                                                MapCache &mdata) {
  // attempt to fit a plane to the point cloud
  if (config_->perform_planarity_check) {
    // point cloud containers
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(2000);
    pcl::ModelCoefficients coefficients;
    pcl::PointIndices inliers;

    if (mdata.landmark_map.is_valid()) {
      LandmarkMap &lm_map = *mdata.landmark_map;

      // push the points into a PCL point cloud
      for (auto &landmark : lm_map) {
        if (landmark.second.steam_lm != nullptr &&
            (*landmark.second.valid) == true) {
          Eigen::Vector3d steam_point =
              landmark.second.steam_lm->getValue().hnormalized();
          cloud->points.push_back(
              pcl::PointXYZ(steam_point[0], steam_point[1], steam_point[2]));
        }
      }

      // attempt to fit a plane to the data
      if (vision::estimatePlane(cloud, config_->plane_distance, coefficients,
                                inliers) &&
          std::count_if(inliers.indices.begin(), inliers.indices.end(),
                        [](int i) { return i; }) > 100) {
        mdata.plane_coefficients =
            Eigen::Vector4f(coefficients.values[0], coefficients.values[1],
                            coefficients.values[2], coefficients.values[3]);
      } else {
        LOG(WARNING) << "SteamModule: Couldn't estimate map landmarks plane! "
                        "Inliers/Points: "
                     << std::count_if(inliers.indices.begin(),
                                      inliers.indices.end(),
                                      [](int i) { return i; })
                     << "/" << cloud->points.size();
      }
    }
  }

  // if the landmark map is valid, check for outliers from the plane or min/max
  // point depths
  if (mdata.landmark_map.is_valid()) {
    LandmarkMap &lm_map = *mdata.landmark_map;

    // do a sanity check on the points to ensure there are no gross outliers
    for (auto &landmark : lm_map) {
      if (landmark.second.steam_lm != nullptr &&
          (*landmark.second.valid) == true) {
        // get the steam point
        Eigen::Vector3d steam_point =
            landmark.second.steam_lm->getValue().hnormalized();

        // perform the validity check
        *landmark.second.valid = isLandmarkValid(steam_point, mdata);
      }
    }
  }

  // always return true for now
  return true;
}

void WindowOptimizationModule::updateCaches(QueryCache &qdata, MapCache &) {
  qdata.trajectory = trajectory_;
}

void WindowOptimizationModule::saveTrajectory(
    QueryCache &qdata, MapCache &mdata, const std::shared_ptr<Graph> &graph) {
  auto &poses = *mdata.pose_map;
  // if we used the backup LM solver, cast that instead of the main one.
  auto gn_solver =
      backup_lm_solver_used_
          ? std::dynamic_pointer_cast<steam::GaussNewtonSolverBase>(
                backup_lm_solver_)
          : std::dynamic_pointer_cast<steam::GaussNewtonSolverBase>(solver_);

  if (gn_solver == nullptr) {
    LOG(ERROR) << "This solver does not derive from The GaussNewtonSolverBase!";
    return;
  }

  // Set up the status message
  status_msgs::TrajectoryStatus status_msg;

  // record the optimization window, starting from the origin.
  auto pose_a_itr = poses.begin();
  auto pose_b_itr = poses.begin();
  pose_b_itr++;

  for (; pose_a_itr != poses.end() && pose_b_itr != poses.end();
       ++pose_a_itr, ++pose_b_itr) {
    // save off pose A
    auto vertex = graph->at(VertexId(pose_a_itr->first));
    auto *ba_pose = status_msg.mutable_optimization_window()->Add();
    ba_pose->set_id(vertex->id());
    ba_pose->set_locked(pose_a_itr->second.isLocked());
    ba_pose->set_interpolated(false);
    ba_pose->set_stamp(vertex->keyFrameTime().nanoseconds_since_epoch());
    EdgeTransform T_1_0 = pose_a_itr->second.tf_state_var->getValue();

    // extract the covariance
    if (pose_a_itr->second.isLocked() == false) {
      auto covariance =
          gn_solver->queryCovariance(pose_a_itr->second.tf_state_var->getKey());
      T_1_0.setCovariance(covariance);
    }
    *ba_pose->mutable_t_q_0() << T_1_0;

    // compute the delta in time between pose a and b
    auto time_a = graph->at(VertexId(pose_a_itr->first))
                      ->keyFrameTime()
                      .nanoseconds_since_epoch();
    auto time_b = graph->at(VertexId(pose_b_itr->first))
                      ->keyFrameTime()
                      .nanoseconds_since_epoch();
    auto time_delta = time_b - time_a;

    // sample the trajectory between a and b.
    for (auto idx = 0; idx < 5; ++idx) {
      int64_t query_time = time_a + ((time_delta / 5) * idx);
      auto *ba_pose = status_msg.mutable_optimization_window()->Add();
      ba_pose->set_interpolated(true);
      ba_pose->set_stamp(query_time);
      auto curr_eval =
          qdata.trajectory->getInterpPoseEval(steam::Time(query_time));
      EdgeTransform T_q_0 = curr_eval->evaluate();

      // if pose a and b are both unlocked, then interpolate the covariance
      // TODO: Re-enable once the steam library supports it.
      //      if(pose_a_itr->second.isLocked() == false &&
      //      pose_b_itr->second.isLocked() == false) {
      //        auto cov =
      //        qdata.trajectory->getCovariance(*gn_solver.get(),steam::Time(query_time));
      //        T_q_0.setCovariance(cov);
      //      }

      *ba_pose->mutable_t_q_0() << T_q_0;
    }
  }

  // save off the final pose...
  auto vertex = graph->at(VertexId(pose_a_itr->first));
  auto *ba_pose = status_msg.mutable_optimization_window()->Add();
  ba_pose->set_id(vertex->id());
  ba_pose->set_locked(pose_a_itr->second.isLocked());
  ba_pose->set_interpolated(false);
  ba_pose->set_stamp(vertex->keyFrameTime().nanoseconds_since_epoch());
  EdgeTransform T_1_0 = pose_a_itr->second.tf_state_var->getValue();
  if (pose_a_itr->second.isLocked() == false) {
    // extract the covariance
    auto covariance =
        gn_solver->queryCovariance(pose_a_itr->second.tf_state_var->getKey());
    T_1_0.setCovariance(covariance);
  }
  *ba_pose->mutable_t_q_0() << T_1_0;
  *ba_pose->mutable_t_q_0() << T_1_0;

  // Extrapolate into the future.
  int64_t future_stamp = vertex->keyFrameTime().nanoseconds_since_epoch();
  int64_t interval = 1e9 / 10;
  for (auto idx = 0; idx < 10; ++idx) {
    auto *ba_pose = status_msg.mutable_optimization_window()->Add();
    ba_pose->set_id(-1);
    ba_pose->set_locked(false);
    ba_pose->set_interpolated(true);
    ba_pose->set_stamp(future_stamp);
    auto curr_eval = trajectory_->getInterpPoseEval(steam::Time(future_stamp));
    EdgeTransform T_q_0 = curr_eval->evaluate();

    // TODO re-enable once the steam library supports it.
    // if pose a and b are both unlocked, then interpolate the covariance
    //    auto cov =
    //    qdata.trajectory->getCovariance(*gn_solver.get(),steam::Time(future_stamp));
    //    T_q_0.setCovariance(cov);

    *ba_pose->mutable_t_q_0() << T_q_0;
    future_stamp += interval;
  }

  // insert into the vertex.
  auto live_vertex = graph->at(*qdata.live_id);
  auto run = graph->run((*qdata.live_id).majorId());
  std::string stream_name("/results/refined_vo");
  if (!run->hasVertexStream(stream_name)) {
    run->registerVertexStream(stream_name, true);
  }
  live_vertex->insert<decltype(status_msg)>(stream_name, status_msg,
                                            *qdata.stamp);
}

/// @brief Update the graph with optimized transforms
void WindowOptimizationModule::updateGraph(QueryCache &qdata, MapCache &mdata,
                                           const std::shared_ptr<Graph> &graph,
                                           VertexId) {
  if (mdata.landmark_map.is_valid() == false ||
      mdata.pose_map.is_valid() == false || mdata.success.is_valid() == false ||
      *mdata.success == false) {
    return;
  }

  if (config_->save_trajectory) {
    saveTrajectory(qdata, mdata, graph);
  }

  auto &lm_map = *mdata.landmark_map;
  auto &poses = *mdata.pose_map;

  // if we used the backup LM solver, cast that instead of the main one.
  auto gn_solver =
      backup_lm_solver_used_
          ? std::dynamic_pointer_cast<steam::GaussNewtonSolverBase>(
                backup_lm_solver_)
          : std::dynamic_pointer_cast<steam::GaussNewtonSolverBase>(solver_);

  if (gn_solver == nullptr) {
    LOG(ERROR) << "This solver does not derive from The GaussNewtonSolverBase!";
    return;
  } else if (config_->solver_type == "LevenburgMarquardt" ||
             backup_lm_solver_used_) {
    // we need to explicitly ask the LM solver to update the covariance, but
    // this may throw
    auto lm_solver =
        std::dynamic_pointer_cast<steam::LevMarqGaussNewtonSolver>(gn_solver);
    try {
      lm_solver->solveCovariances();
    } catch (std::runtime_error &e) {
      LOG(ERROR) << "WindowOptimizationModule: Couldn't solve for covariance "
                    "in LM solver!"
                 << std::endl
                 << e.what();
    }
  }

  // update the edges in the graph.
  Eigen::Matrix<double, 6, 6> zero_matrix;
  zero_matrix.setZero();
  auto pose_a_itr = poses.begin();
  auto pose_b_itr = poses.begin();
  pose_b_itr++;
  for (; pose_b_itr != poses.end() && pose_a_itr != poses.end();
       pose_b_itr++, pose_a_itr++) {
    lgmath::se3::Transformation T_a_0 =
        pose_a_itr->second.tf_state_var->getValue();
    lgmath::se3::Transformation T_b_0 =
        pose_b_itr->second.tf_state_var->getValue();
    if (pose_b_itr->second.isLocked() == false) {
      if (pose_a_itr->first.majorId() != qdata.live_id->majorId()) {
        continue;
      }
      auto e_id = EdgeId(pose_a_itr->first, pose_b_itr->first,
                         asrl::pose_graph::Temporal);
      if (graph->contains(e_id)) {
        auto e = graph->at(e_id);
        lgmath::se3::TransformationWithCovariance T_b_a = T_b_0 / T_a_0;
        if (pose_a_itr->second.isLocked() == false) {
          std::vector<steam::StateKey> pose_keys{
              pose_a_itr->second.tf_state_var->getKey(),
              pose_b_itr->second.tf_state_var->getKey()};
          auto Cov_a0a0_b0b0 = gn_solver->queryCovarianceBlock(pose_keys);
          auto Tadj_b_a = T_b_a.adjoint();
          auto correlation = Tadj_b_a * Cov_a0a0_b0b0.at(0, 1);
          auto Cov_ba_ba =
              Cov_a0a0_b0b0.at(1, 1) - correlation - correlation.transpose() +
              Tadj_b_a * Cov_a0a0_b0b0.at(0, 0) * Tadj_b_a.transpose();
          T_b_a.setCovariance(Cov_ba_ba);
        } else {
          T_b_a.setCovariance(gn_solver->queryCovariance(
              pose_b_itr->second.tf_state_var->getKey()));
        }
        e->setTransform(T_b_a);
      } else {
        LOG(WARNING) << "Trying to update covariance of edge " << e_id
                     << ", which doesnt exist!";
        LOG(WARNING) << "++++++++++++++++++ Current Optimization problem "
                        "++++++++++++++++++++++++++++";
        for (auto itr = poses.begin(); itr != poses.end(); ++itr) {
          LOG(WARNING) << itr->first << " " << itr->second.isLocked();
        }
        LOG(WARNING) << "++++++++++++++++++ Current Optimization problem "
                        "++++++++++++++++++++++++++++";
      }
    }
  }

  VertexId first_unlocked;
  bool found_first_unlocked = false;
  // update the velocities in the graph
  for (auto &pose : poses) {
    if (pose.second.isLocked() == false) {
      auto new_velocity = pose.second.velocity->getValue();
      pose.second.proto_velocity->mutable_translational()->set_x(
          new_velocity(0, 0));
      pose.second.proto_velocity->mutable_translational()->set_y(
          new_velocity(1, 0));
      pose.second.proto_velocity->mutable_translational()->set_z(
          new_velocity(2, 0));
      pose.second.proto_velocity->mutable_rotational()->set_x(
          new_velocity(3, 0));
      pose.second.proto_velocity->mutable_rotational()->set_y(
          new_velocity(4, 0));
      pose.second.proto_velocity->mutable_rotational()->set_z(
          new_velocity(5, 0));
      if (found_first_unlocked == false) {
        first_unlocked = pose.first;
        found_first_unlocked = true;
      }
    }
  }

  // Update the landmarks in the graph
  for (auto &landmark : lm_map) {
    if (landmark.second.observations.size() >
        *landmark.second.num_vo_observations) {
      *landmark.second.num_vo_observations =
          landmark.second.observations.size();
    }
    // if this is a valid, unlocked landmark, then update its point/cov in the
    // graph.
    if (landmark.second.steam_lm != nullptr &&
        !landmark.second.steam_lm->isLocked() &&
        landmark.second.observations.size() > 1) {
      auto *robochunk_point = landmark.second.point;
      Eigen::Vector3d steam_point =
          landmark.second.steam_lm->getValue().hnormalized();
      robochunk_point->set_x(steam_point[0]);
      robochunk_point->set_y(steam_point[1]);
      robochunk_point->set_z(steam_point[2]);

      // check validity on the landmark, but only if the point was valid in the
      // first place
      if (*landmark.second.valid == true) {
        *landmark.second.valid = isLandmarkValid(steam_point, mdata);
      }
      /*
            // TODO: This is sooper dooper slow bud.
            // if this is the first unlocked pose.
            if(landmark.first.vertex == first_unlocked) {
              auto cov =
         gn_solver->queryCovariance(landmark.second.steam_lm->getKey()); auto
         *robochunk_cov = landmark.second.covariance; auto landmark_offset =
         landmark.first.index * 9;
              // Todo Eigen map for readability?
              //Eigen::Map<Eigen::Matrix<double,3,3> > ...

              for(int row = 0; row < 3; ++row) {
                for(int col = 0; col < 3; ++col) {
                  robochunk_cov->Set(landmark_offset + row*3 + col,
         cov(row,col));
                }
              }
            }
      */
    }
  }
  // reset to remove any old data from the problem setup
  resetProblem();
}

}  // namespace navigation
}  // namespace asrl
