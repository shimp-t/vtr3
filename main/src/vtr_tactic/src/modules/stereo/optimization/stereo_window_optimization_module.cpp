#include <steam/trajectory/SteamTrajPoseInterpEval.hpp>
#include <vtr_common/timing/simple_timer.hpp>
#include <vtr_steam_extensions/evaluator/range_conditioning_eval.hpp>
#include <vtr_steam_extensions/evaluator/scale_error_eval.hpp>
#include <vtr_steam_extensions/evaluator/tdcp_error_eval.hpp>
#include <vtr_tactic/modules/stereo/optimization/stereo_window_optimization_module.hpp>
#include <vtr_vision/geometry/geometry_tools.hpp>
#include <vtr_vision/messages/bridge.hpp>
#include <vtr_vision/types.hpp>

#if false
#include <asrl/messages/TrajectoryStatus.pb.h>
#include <asrl/messages/lgmath_conversions.hpp>
#endif

#if CASCADE
class CSVRow {
 public:
  std::string_view operator[](std::size_t index) const {
    return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
  }
  std::size_t size() const {
    return m_data.size() - 1;
  }
  void readNextRow(std::istream& str) {
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    while((pos = m_line.find(',', pos)) != std::string::npos) {
      m_data.emplace_back(pos);
      ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos   = m_line.size();
    m_data.emplace_back(pos);
  }
 private:
  std::string         m_line;
  std::vector<int>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data) {
  data.readNextRow(str);
  return str;
}
#endif

namespace vtr {
namespace tactic {
namespace stereo {

void StereoWindowOptimizationModule::configFromROS(
    const rclcpp::Node::SharedPtr &node, const std::string param_prefix) {
  SteamModule::configFromROS(node, param_prefix);
  window_config_ = std::make_shared<Config>();
  auto casted_config =
      std::static_pointer_cast<SteamModule::Config>(window_config_);
  *casted_config = *config_;  // copy over base config
  // clang-format off
  window_config_->depth_prior_enable = node->declare_parameter<bool>(param_prefix + ".depth_prior_enable", window_config_->depth_prior_enable);
  window_config_->depth_prior_weight = node->declare_parameter<double>(param_prefix + ".depth_prior_weight", window_config_->depth_prior_weight);
  window_config_->tdcp_cov = node->declare_parameter<double>(param_prefix + ".tdcp_cov", 0.01);
  window_config_->min_tdcp_terms = node->declare_parameter<int>(param_prefix + ".min_tdcp_terms", 3);
  window_config_->stereo_cov_multiplier = node->declare_parameter<double>(param_prefix + ".stereo_cov_multiplier", 1.0);
  // clang-format on

#if CASCADE
  //  read CSV file into data structure
  std::ifstream file("/home/ben/Desktop/cpo.csv");
  CSVRow row;
  bool first_row = true;
  while(file >> row) {
    if (first_row) {
      first_row = false;
      continue;
    }
    std::vector<double> row_vec;
    for (uint i = 0; i < row.size(); ++i) {
      std::string el_str = std::string(row[i]);
      if (!el_str.empty()) {
        row_vec.push_back(std::stod(el_str));
      }
    }
    cpo_estimates_.push_back(row_vec);
  }
#endif
}

std::shared_ptr<steam::OptimizationProblem>
StereoWindowOptimizationModule::generateOptimizationProblem(
    QueryCache &qdata, MapCache &, const std::shared_ptr<const Graph> &graph) {
  // get references to the relevent data.
  LandmarkMap &lm_map = *qdata.landmark_map;
  auto &poses = *qdata.pose_map;
  auto &tsv_transforms = *qdata.T_sensor_vehicle_map;
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
    throw std::runtime_error{"Monocular camera code not ported!"};
#if 0
    sharedMonoIntrinsics = toMonoSteamCalibration(calibration);
#endif
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
    Eigen::Vector3d lm_point(landmark.second.point.x, landmark.second.point.y,
                             landmark.second.point.z);

    // Extract the validity of the landmark
    bool map_point_valid = landmark.second.valid;

    // If the point and its depth is valid, then add it as a landmark.
    if (map_point_valid && isLandmarkValid(lm_point, qdata) == true &&
        landmark.second.observations.size() > 1) {
      landmark.second.steam_lm.reset(
          new steam::se3::LandmarkStateVar(lm_point));
      auto &steam_lm = landmark.second.steam_lm;

      // set the lock only if the map is initialized
      // if(*qdata.map_initialized == true) {
      steam_lm->setLock(lm_pose.isLocked());
      //}

      // add the depth prior
      if (window_config_->depth_prior_enable) {
        addDepthCost(steam_lm);
      }
    } else {
      landmark.second.steam_lm.reset();
      continue;
    }

    // get a reference to our new landmark.
    auto &steam_lm = landmark.second.steam_lm;
    // 3. go through the observations, and add cost terms for each.
    for (const LandmarkObs &obs : landmark.second.observations) {
      try {  // steam throws?

        // get the keypoints for this observation.
        auto obs_vertex =
            graph->fromPersistent(obs.origin_ref.from_id.persistent);
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
#if false
          meas_cov(2 * idx, 2 * idx) = cov[0];
          meas_cov(2 * idx + 1, 2 * idx + 1) = cov[3];
#else   // temporary way to easily scale vision costs while testing
          meas_cov(2 * idx, 2 * idx) = window_config_->stereo_cov_multiplier * cov[0];
          meas_cov(2 * idx + 1, 2 * idx + 1) =
              window_config_->stereo_cov_multiplier * cov[3];
#endif
          idx++;
        }

        // add to the noise models
        if (monocular) {
          throw std::runtime_error{"Monocular camera code not ported!"};
#if 0
          noise_mono.reset(new steam::StaticNoiseModelX(meas_cov));
#endif
        } else {
          noise_stereo.reset(new steam::StaticNoiseModel<4>(meas_cov));
        }

        // Construct the measurement vector for the current camera
        Eigen::MatrixXd data(obs.keypoints.size() * 2, 1);
        for (uint32_t idx = 0; idx < obs.keypoints.size(); idx++) {
          data(idx * 2) = obs.keypoints.at(idx).position.x;
          data(idx * 2 + 1) = obs.keypoints.at(idx).position.y;
        }

        if (monocular) {
          throw std::runtime_error{"Monocular camera code not ported!"};
#if 0
          // Construct error function for the current camera
          vtr::steam_extensions::MonoCameraErrorEval::Ptr errorfunc(
              new vtr::steam_extensions::MonoCameraErrorEval(
                  data, sharedMonoIntrinsics, T_obs_lm, steam_lm));
          // Construct cost term for the current camera
          steam::WeightedLeastSqCostTermX::Ptr cost(
              new steam::WeightedLeastSqCostTermX(errorfunc, noise_mono,
                                                  sharedLossFunc_));

          // finally, add the cost.
          vision_cost_terms_->add(cost);
#endif
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
          vision_cost_terms_->add(cost);
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
    throw std::runtime_error{"Monocular camera code not ported!"};
#if 0
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
      vtr::steam_extensions::ScaleErrorEval::Ptr scale_error_func(
          new vtr::steam_extensions::ScaleErrorEval(max_d,
                                                    max_d_tf_state_eval));

      // Create cost term and add to problem
      steam::WeightedLeastSqCostTermX::Ptr scale_cost(
          new steam::WeightedLeastSqCostTermX(scale_error_func,
                                              scaleUncertainty, scaleLossFunc));
      vision_cost_terms_->add(scale_cost);
    }
#endif
  }

  // add pose variables
  int jj = 0;

  double traj_start_time = 999999999999999.9;

  for (auto &pose : poses) {
    auto &steam_pose = pose.second;
    problem_->addStateVariable(steam_pose.tf_state_var);
    std::cout << "Added pose at t " << std::setprecision(12) << steam_pose.time.seconds() << std::setprecision(6) << " with current vel " << steam_pose.velocity->getValue().transpose();
    std::cout << "  Vel locked? " << steam_pose.velocity->isLocked() << std::endl;
    jj++;

    if (steam_pose.time.seconds() < traj_start_time)
      traj_start_time = steam_pose.time.seconds();
  }

  // Add landmark variables
  for (auto &landmark : lm_map) {
    if (landmark.second.steam_lm != nullptr &&
        (landmark.second.valid) == true) {
      problem_->addStateVariable(landmark.second.steam_lm);
    }
  }

  problem_->addCostTerm(vision_cost_terms_);
  if (window_config_->depth_prior_enable) {
    problem_->addCostTerm(depth_cost_terms_);
  }

  // add trajectory stuff
  if (window_config_->trajectory_smoothing == true) {
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
      if (window_config_->velocity_prior == true &&
          steam_pose.isLocked() == false && prior_added == false) {
        trajectory_->addVelocityPrior(steam_pose.time, velocity_prior_,
                                      velocity_prior_cov_);
      }
    }

    // Set up TDCP factors. We require a trajectory to interpolate so it happens in this if block

#if CASCADE   // get tdcp info from cpo_estimates_
    std::cout << "Live V " << *qdata.live_id << std::endl;

    if (qdata.tdcp_msgs.is_valid() && !qdata.tdcp_msgs->empty()) {    // will keep this condition so we can use tdcp_enable param still

      auto new_v = graph->at(*qdata.live_id);
      double curr_secs = new_v->keyFrameTime().nanoseconds_since_epoch * 1e-9;
      std::cout << "curr_secs: " << std::setprecision(12) << curr_secs << std::setprecision(6) << std::endl;
      for (uint i = 1; i < cpo_estimates_.size(); ++i) {
        // find time(s) corresponding to just before curr_sec
        const auto &row = cpo_estimates_[i];
        if (row[0] > curr_secs - 3.0 && row[0] < curr_secs) {
          // make transformation matrices out of it and previous row
          const auto &prev_row = cpo_estimates_[i - 1];

          if (prev_row[0] < traj_start_time) {
            std::cout << "Time a before trajectory start so not adding in. " << std::endl;
            continue;
          }

//          if (row[0] - prev_row[0] > 1.1)
//            continue;       // todo: do we want this to avoid large edges?

          Eigen::Matrix4d T_curr, T_prev;
          T_curr << row[8], row[12], row[16], row[20],  // these in gps frame
                    row[9], row[13], row[17], row[21],
                    row[10], row[14], row[18], row[22],
                    row[11], row[15], row[19], row[23];
          T_prev << prev_row[8], prev_row[12], prev_row[16], prev_row[20],
                    prev_row[9], prev_row[13], prev_row[17], prev_row[21],
                    prev_row[10], prev_row[14], prev_row[18], prev_row[22],
                    prev_row[11], prev_row[15], prev_row[19], prev_row[23];
          lgmath::se3::Transformation T_b0_meas_g(T_curr);
          lgmath::se3::Transformation T_a0_meas_g(T_prev);
          lgmath::se3::Transformation
              T_b0_meas = tf_gps_vehicle_->evaluate().inverse() * T_b0_meas_g;
          lgmath::se3::Transformation
              T_a0_meas = tf_gps_vehicle_->evaluate().inverse() * T_a0_meas_g;

          // compose to get relative transform -> measurement
          lgmath::se3::Transformation T_ba_meas = T_b0_meas * T_a0_meas.inverse();

          // get poseInterp at two times and compose -> TransformEval
          steam::se3::SteamTrajPoseInterpEval::ConstPtr
              T_b = trajectory_->getInterpPoseEval(steam::Time(row[0]));
          steam::se3::SteamTrajPoseInterpEval::ConstPtr
              T_a = trajectory_->getInterpPoseEval(steam::Time(prev_row[0]));
          steam::se3::TransformEvaluator::ConstPtr T_ba_state = steam::se3::composeInverse(T_b, T_a);

#if 1
          // add PoseError (TransformErrorEval)
          Eigen::Matrix<double, 6, 6> temp_cov = window_config_->tdcp_cov* Eigen::Matrix<double,6,6>::Identity();   // reusing tdcp_cov param in different way here
          steam::BaseNoiseModel<6>::Ptr
              cpo_noise_model(new steam::StaticNoiseModel<6>(temp_cov));
          steam::TransformErrorEval::Ptr cpo_error_func
              (new steam::TransformErrorEval(T_ba_meas, T_ba_state));
          steam::LossFunctionBase::Ptr cpo_loss_func(new steam::L2LossFunc());
          auto cpo_factor = steam::WeightedLeastSqCostTerm<6, 6>::Ptr(
              new steam::WeightedLeastSqCostTerm<6, 6>(
                  cpo_error_func,
                  cpo_noise_model,
                  cpo_loss_func));
#else
          // add PoseError (PositionErrorEval)
          Eigen::Matrix<double, 3, 3> temp_cov = config_->tdcp_cov* Eigen::Matrix<double,3,3>::Identity();   // reusing tdcp_cov param in different way here
          steam::BaseNoiseModel<3>::Ptr
              cpo_noise_model(new steam::StaticNoiseModel<3>(temp_cov));
          steam::PositionErrorEval::Ptr cpo_error_func
              (new steam::PositionErrorEval(T_ba_meas.r_ba_ina(), T_ba_state));
          steam::LossFunctionBase::Ptr cpo_loss_func(new steam::L2LossFunc());
          auto cpo_factor = steam::WeightedLeastSqCostTerm<3, 6>::Ptr(
              new steam::WeightedLeastSqCostTerm<3, 6>(
                  cpo_error_func,
                  cpo_noise_model,
                  cpo_loss_func));
#endif
          tdcp_cost_terms_->add(cpo_factor);
      }
    }

      problem_->addCostTerm(tdcp_cost_terms_);
#else
    // Note: if we don't have or want GPS, we just won't find that info in cache
    T_0g_statevar_.reset(new steam::se3::TransformStateVar(lgmath::se3::Transformation()));
    if (qdata.tdcp_msgs.is_valid() && !qdata.tdcp_msgs->empty()
        && qdata.T_0g_prior.is_valid()
        && qdata.T_0g_prior->second.covarianceSet()) {

      T_0g_statevar_->setValue(qdata.T_0g_prior->second);
      steam::se3::TransformEvaluator::ConstPtr
          T_0g(new steam::se3::TransformStateEvaluator(T_0g_statevar_));

      for (const auto &msg : *qdata.tdcp_msgs) {
        addTdcpCost(msg, T_0g, poses[qdata.T_0g_prior->first].tf_state_eval);
      }
      LOG(DEBUG) << "Found " << tdcp_cost_terms_->numCostTerms()
                 << " TDCP terms.";

      // if enough cost terms, add the costs and extra state to problem
      if (tdcp_cost_terms_->numCostTerms() >= window_config_->min_tdcp_terms) {
        problem_->addStateVariable(T_0g_statevar_);

        // add prior on global orientation at start of window
        Eigen::Matrix<double, 6, 6> temp_cov = qdata.T_0g_prior->second.cov();
        steam::BaseNoiseModel<6>::Ptr
            prior_noise_model(new steam::StaticNoiseModel<6>(temp_cov));
        steam::TransformErrorEval::Ptr prior_error_func
            (new steam::TransformErrorEval(qdata.T_0g_prior->second, T_0g));
        steam::LossFunctionBase::Ptr prior_loss_func(new steam::L2LossFunc());
        auto prior_factor = steam::WeightedLeastSqCostTerm<6, 6>::Ptr(
            new steam::WeightedLeastSqCostTerm<6, 6>(
                prior_error_func,
                prior_noise_model,
                prior_loss_func));

        global_prior_cost_term_->add(prior_factor);
        problem_->addCostTerm(global_prior_cost_term_);

        problem_->addCostTerm(tdcp_cost_terms_);
      }
#endif
      std::cout << "Initial Carrier Phase Cost:     "
                << tdcp_cost_terms_->cost() << "        Terms:  "   // debugging
                << tdcp_cost_terms_->numCostTerms() << std::endl;
      std::cout << "Initial Global Prior Cost:      "
                << global_prior_cost_term_->cost() << "        Terms:  "
                << global_prior_cost_term_->numCostTerms() << std::endl;
    }
    // Add smoothing terms
    trajectory_->appendPriorCostTerms(smoothing_cost_terms_);
    problem_->addCostTerm(smoothing_cost_terms_);
    std::cout << "Initial Smoothing Cost:         "
              << smoothing_cost_terms_->cost() << "        Terms:  "
              << smoothing_cost_terms_->numCostTerms() << std::endl;
  }
  std::cout << "Initial Vision Cost:            " << vision_cost_terms_->cost()
            << "        Terms:  " << vision_cost_terms_->numCostTerms()
            << std::endl;

  return problem_;
}

void StereoWindowOptimizationModule::resetProblem() {
  // make the depth loss function
  sharedDepthLossFunc_.reset(new steam::DcsLossFunc(2.0));

  // make the stereo loss function, TODO (old): make this configurable, move to member var.
  sharedLossFunc_.reset(new steam::DcsLossFunc(2.0));

  // make the TDCP loss function
  sharedTdcpLossFunc_.reset(new steam::L2LossFunc());

  // setup stereo cost terms
  vision_cost_terms_.reset(new steam::ParallelizedCostTermCollection());

  // setup WNOA costs
  smoothing_cost_terms_.reset(new steam::ParallelizedCostTermCollection());

  // setup cost terms for the depth
  depth_cost_terms_.reset(new steam::ParallelizedCostTermCollection());

  // setup cost terms for the TDCP
  tdcp_cost_terms_.reset(new steam::ParallelizedCostTermCollection());
  global_prior_cost_term_.reset(new steam::ParallelizedCostTermCollection());

  // set up the steam problem_.
  problem_.reset(new steam::OptimizationProblem());
}

void StereoWindowOptimizationModule::addDepthCost(
    steam::se3::LandmarkStateVar::Ptr landmark) {
  vtr::steam_extensions::RangeConditioningEval::Ptr errorfunc_range(
      new vtr::steam_extensions::RangeConditioningEval(landmark));
  double depth = landmark->getValue().hnormalized()[2];
  double weight = window_config_->depth_prior_weight / depth;
  steam::BaseNoiseModel<1>::Ptr rangeNoiseModel(new steam::StaticNoiseModel<1>(
      Eigen::Matrix<double, 1, 1>::Identity() * weight));
  steam::WeightedLeastSqCostTerm<1, 3>::Ptr depth_cost(
      new steam::WeightedLeastSqCostTerm<1, 3>(errorfunc_range, rangeNoiseModel,
                                               sharedDepthLossFunc_));
  depth_cost_terms_->add(depth_cost);
}

void StereoWindowOptimizationModule::addTdcpCost(const TdcpMsg::SharedPtr &msg,
                                                 const steam::se3::TransformEvaluator::ConstPtr &T_0g,
                                                 const steam::se3::TransformEvaluator::ConstPtr &T_0i) {

  // GPS measurement time vehicle pose wrt to start of trajectory
  steam::se3::SteamTrajPoseInterpEval::ConstPtr
      T_ai_v = trajectory_->getInterpPoseEval(steam::Time((int64_t) msg->t_a));
  steam::se3::SteamTrajPoseInterpEval::ConstPtr
      T_bi_v = trajectory_->getInterpPoseEval(steam::Time((int64_t) msg->t_b));
  // GPS measurement time vehicle pose wrt to start of window
  steam::se3::TransformEvaluator::ConstPtr
      T_a0_v = steam::se3::composeInverse(T_ai_v, T_0i);
  steam::se3::TransformEvaluator::ConstPtr
      T_b0_v = steam::se3::composeInverse(T_bi_v, T_0i);
  // GPS measurement time sensor pose wrt to start of window
  steam::se3::TransformEvaluator::ConstPtr
      T_a0_s = steam::se3::compose(tf_gps_vehicle_, T_a0_v);
  steam::se3::TransformEvaluator::ConstPtr
      T_b0_s = steam::se3::compose(tf_gps_vehicle_, T_b0_v);
  // change in sensor pose between the two GPS measurement times
  steam::se3::TransformEvaluator::ConstPtr
      T_ba = steam::se3::composeInverse(T_b0_s, T_a0_s);
  steam::se3::PositionEvaluator::ConstPtr
      r_ba_ina(new steam::se3::PositionEvaluator(T_ba));

  steam::se3::TransformEvaluator::ConstPtr
      T_ag = steam::se3::compose(T_a0_s, T_0g);

  // using constant covariance here for now
  steam::BaseNoiseModel<1>::Ptr tdcp_noise_model
      (new steam::StaticNoiseModel<1>(
          Eigen::Matrix<double, 1, 1>(window_config_->tdcp_cov)));

  // iterate through satellite pairs in msg and add TDCP costs
  for (const auto &pair : msg->pairs) {
    Eigen::Vector3d r_1a_ing_ata{pair.r_1a_a.x, pair.r_1a_a.y, pair.r_1a_a.z};
    Eigen::Vector3d r_1a_ing_atb{pair.r_1a_b.x, pair.r_1a_b.y, pair.r_1a_b.z};
    Eigen::Vector3d r_2a_ing_ata{pair.r_2a_a.x, pair.r_2a_a.y, pair.r_2a_a.z};
    Eigen::Vector3d r_2a_ing_atb{pair.r_2a_b.x, pair.r_2a_b.y, pair.r_2a_b.z};

    vtr::steam_extensions::TdcpErrorEval::Ptr tdcp_error
        (new vtr::steam_extensions::TdcpErrorEval(pair.phi_measured,
                                                  r_ba_ina,
                                                  T_ag,
                                                  r_1a_ing_ata,
                                                  r_1a_ing_atb,
                                                  r_2a_ing_ata,
                                                  r_2a_ing_atb));
    auto tdcp_factor = steam::WeightedLeastSqCostTerm<1,6>::Ptr(
        new steam::WeightedLeastSqCostTerm<1, 6>(tdcp_error,
                                                 tdcp_noise_model,
                                                 sharedTdcpLossFunc_));
    tdcp_cost_terms_->add(tdcp_factor);
  }
}

bool StereoWindowOptimizationModule::verifyInputData(QueryCache &qdata,
                                                     MapCache &) {
  // make sure we have a landmark and pose map, and calibration.
  if (qdata.landmark_map.is_valid() == false ||
      qdata.pose_map.is_valid() == false ||
      qdata.rig_calibrations.is_valid() == false) {
    LOG(ERROR)
        << "StereoWindowOptimizationModule::verifyInputData(): Input data for "
           "windowed BA problem is not set! (Is the Windowed Recall "
           "Module Running?)";
    return false;
  }

  // If there is nothing to optimize, then quit.
  if ((*qdata.pose_map).empty() || (*qdata.landmark_map).empty()) {
    LOG(ERROR)
        << "StereoWindowOptimizationModule::verifyInputData(): No poses or "
           "landmarks found. (Is the Windowed Recall Module Running?)";
    return false;
  } else
    return true;
}

bool StereoWindowOptimizationModule::isLandmarkValid(
    const Eigen::Vector3d &point, QueryCache &) {
  // check depth
  if (point(2) > window_config_->max_point_depth ||
      point(2) < window_config_->min_point_depth) {
    return false;
  }

  // check the distance from the plane
  if (window_config_->perform_planarity_check) {
    throw std::runtime_error{
        "planarity check not ported and tested - window opt"};
#if false
    if (qdata.plane_coefficients.is_valid() == true) {
      // estimate the distance of the point from the plane
      double dist =
          vision::estimatePlaneDepth(point, *qdata.plane_coefficients);

      // if it is beyond the maximum depth, it's invalid
      if (dist > window_config_->plane_distance) {
        return false;
      }
    }
#endif
  }

  // if we got here, the landmark is valid
  return true;
}

bool StereoWindowOptimizationModule::verifyOutputData(QueryCache &qdata,
                                                      MapCache &) {
  // attempt to fit a plane to the point cloud
  if (window_config_->perform_planarity_check) {
    throw std::runtime_error{
        "planarity check not ported and tested - window opt"};
#if false
    // point cloud containers
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(2000);
    pcl::ModelCoefficients coefficients;
    pcl::PointIndices inliers;

    if (qdata.landmark_map.is_valid()) {
      LandmarkMap &lm_map = *qdata.landmark_map;

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
      if (vision::estimatePlane(cloud, window_config_->plane_distance, coefficients,
                                inliers) &&
          std::count_if(inliers.indices.begin(), inliers.indices.end(),
                        [](int i) { return i; }) > 100) {
        qdata.plane_coefficients =
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
#endif
  }

  // if the landmark map is valid, check for outliers from the plane or min/max
  // point depths
  if (qdata.landmark_map.is_valid()) {
    LandmarkMap &lm_map = *qdata.landmark_map;

    // do a sanity check on the points to ensure there are no gross outliers
    for (auto &landmark : lm_map) {
      if (landmark.second.steam_lm != nullptr &&
          landmark.second.valid == true) {
        // get the steam point
        Eigen::Vector3d steam_point =
            landmark.second.steam_lm->getValue().hnormalized();

        // perform the validity check
        landmark.second.valid = isLandmarkValid(steam_point, qdata);
      }
    }
  }

  // always return true for now
  return true;
}

void StereoWindowOptimizationModule::updateCaches(QueryCache &qdata,
                                                  MapCache &) {
  qdata.trajectory = trajectory_;
}

#if false
void StereoWindowOptimizationModule::saveTrajectory(
    QueryCache &qdata, MapCache &mdata, const std::shared_ptr<Graph> &graph) {
  auto &poses = *qdata.pose_map;
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
  asrl::status_msgs::TrajectoryStatus status_msg;

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
  run->registerVertexStream(stream_name);
  live_vertex->insert<decltype(status_msg)>(stream_name, status_msg,
                                            *qdata.stamp);
}
#endif

void StereoWindowOptimizationModule::updateGraphImpl(QueryCache &qdata,
                                                     MapCache &,
                                                     const Graph::Ptr &graph,
                                                     VertexId) {
  if (qdata.landmark_map.is_valid() == false ||
      qdata.pose_map.is_valid() == false || qdata.success.is_valid() == false ||
      *qdata.success == false) {
    return;
  }

  // optimization debugging info
  std::cout << "Final Carrier Phase Cost:     " << tdcp_cost_terms_->cost()
            << "      Terms:  "<< tdcp_cost_terms_->numCostTerms() << std::endl;
  std::cout << "Final Global Prior Cost:      "
            << global_prior_cost_term_->cost() << "        Terms:  "
            << global_prior_cost_term_->numCostTerms() << std::endl;
  std::cout << "Final Smoothing Cost:         " << smoothing_cost_terms_->cost()
            << "      Terms:  " << smoothing_cost_terms_->numCostTerms()
            << std::endl;
  std::cout << "Final Vision Cost:            " << vision_cost_terms_->cost()
            << "      Terms:  " << vision_cost_terms_->numCostTerms()
            << std::endl;

  if (window_config_->save_trajectory) {
    throw std::runtime_error{
        "trajectory saving untested - windowed optimization"};
#if false
    saveTrajectory(qdata, mdata, graph);
#endif
  }

  auto &lm_map = *qdata.landmark_map;
  auto &poses = *qdata.pose_map;

  // if we used the backup LM solver, cast that instead of the main one.
  auto gn_solver =
      backup_lm_solver_used_
          ? std::dynamic_pointer_cast<steam::GaussNewtonSolverBase>(
                backup_lm_solver_)
          : std::dynamic_pointer_cast<steam::GaussNewtonSolverBase>(solver_);

  if (gn_solver == nullptr) {
    LOG(ERROR) << "This solver does not derive from The GaussNewtonSolverBase!";
    return;
  } else if (window_config_->solver_type == "LevenburgMarquardt" ||
             backup_lm_solver_used_) {
    // we need to explicitly ask the LM solver to update the covariance, but
    // this may throw
    auto lm_solver =
        std::dynamic_pointer_cast<steam::LevMarqGaussNewtonSolver>(gn_solver);
    try {
      lm_solver->solveCovariances();
    } catch (std::runtime_error &e) {
      LOG(ERROR)
          << "StereoWindowOptimizationModule: Couldn't solve for covariance "
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
    T_a_0.reproject(true);    // quick fix for lgmath orthonormal issue
    T_b_0.reproject(true);
    if (pose_b_itr->second.isLocked() == false) {
      if (pose_a_itr->first.majorId() != qdata.live_id->majorId()) {
        continue;
      }
      auto e_id =
          EdgeId(pose_a_itr->first, pose_b_itr->first, pose_graph::Temporal);
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
    if (pose.second.velocity->isLocked() == false) {
      auto v = graph->at(pose.first);
      auto v_vel =
          v->retrieveKeyframeData<vtr_messages::msg::Velocity>("_velocities");

      auto new_velocity = pose.second.velocity->getValue();
      v_vel->translational.x = new_velocity(0, 0);
      v_vel->translational.y = new_velocity(1, 0);
      v_vel->translational.z = new_velocity(2, 0);
      v_vel->rotational.x = new_velocity(3, 0);
      v_vel->rotational.y = new_velocity(4, 0);
      v_vel->rotational.z = new_velocity(5, 0);

      v->replace("_velocities", *v_vel, v->keyFrameTime());

      if (!found_first_unlocked && !pose.second.isLocked()) {
        first_unlocked = pose.first;
        found_first_unlocked = true;
      }
    }
  }

  // Update the landmarks in the graph
  std::map<VertexId, std::shared_ptr<vtr_messages::msg::RigLandmarks>>
      landmark_msgs;

  for (auto &landmark : lm_map) {
    VertexId vid = graph->fromPersistent(
        messages::copyPersistentId(landmark.first.persistent));

    if (!landmark_msgs.count(vid)) {
      auto v = graph->at(vid);
      auto v_lms = v->retrieveKeyframeData<vtr_messages::msg::RigLandmarks>(
          "front_xb3_landmarks");
      landmark_msgs.emplace(std::make_pair(vid, v_lms));
    }

    if (landmark.second.observations.size() >
        landmark.second.num_vo_observations) {
      landmark_msgs.at(vid)
          ->channels[landmark.first.channel]
          .num_vo_observations[landmark.first.index] =
          landmark.second.observations.size();
    }
    // if this is a valid, unlocked landmark, then update its point/cov in the
    // graph.
    if (landmark.second.steam_lm != nullptr &&
        !landmark.second.steam_lm->isLocked() &&
        landmark.second.observations.size() > 1) {
      Eigen::Vector3d steam_point =
          landmark.second.steam_lm->getValue().hnormalized();

      landmark_msgs.at(vid)
          ->channels[landmark.first.channel]
          .points[landmark.first.index]
          .x = steam_point[0];
      landmark_msgs.at(vid)
          ->channels[landmark.first.channel]
          .points[landmark.first.index]
          .y = steam_point[1];
      landmark_msgs.at(vid)
          ->channels[landmark.first.channel]
          .points[landmark.first.index]
          .z = steam_point[2];

      // check validity on the landmark, but only if the point was valid in the
      // first place
      if (landmark_msgs.at(vid)
              ->channels[landmark.first.channel]
              .valid[landmark.first.index]) {
        landmark_msgs.at(vid)
            ->channels[landmark.first.channel]
            .valid[landmark.first.index] = isLandmarkValid(steam_point, qdata);
      }
#if 0
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
#endif
    }
  }

  for (auto &msg : landmark_msgs) {
    auto v = graph->at(msg.first);
    v->replace("front_xb3_landmarks", *(msg.second), v->keyFrameTime());
  }

  // save estimated global orientation if we're using it
  if (qdata.T_0g_prior.is_valid() && qdata.T_0g_prior->second.covarianceSet()) {

    // if we have a global_prior_cost_term then TDCP terms were used  // todo: this assumption is not necessarily correct ***
    if (global_prior_cost_term_->numCostTerms() > 0) {
      qdata.T_0g_prior->second = T_0g_statevar_->getValue();
      qdata.T_0g_prior->second.setCovariance(gn_solver->queryCovariance(
          T_0g_statevar_->getKey()));
    }

    // insert the global orientation estimate to pose graph
    vtr_messages::msg::LgTransform T_0g_msg;
    for (int i = 0; i < 6; ++i) {
      T_0g_msg.xi.push_back(qdata.T_0g_prior->second.vec()(i));
    }
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        T_0g_msg.cov.push_back(qdata.T_0g_prior->second.cov()(i, j));
      }
    }
    T_0g_msg.cov_set = true;
#if 0
    std::cout << "phi_0g cov: \n"
              << qdata.T_0g_prior->second.cov().bottomRightCorner(3, 3)
              << std::endl;
#endif

    std::string t0g_str = "gps_T_0g";
    graph->registerVertexStream<vtr_messages::msg::LgTransform>(qdata.T_0g_prior->first.majorId(),
                                                                t0g_str);
    auto v = graph->at(qdata.T_0g_prior->first);
    v->insert(t0g_str, T_0g_msg, v->keyFrameTime());

#if !CASCADE
    std::cout << "Saved prior on global orientation at "
              << qdata.T_0g_prior->first << std::endl;



    // save off T_0g estimate to CSV for simple plotting  // todo: temporary
    Eigen::Matrix3d C_vg = qdata.T_0g_prior->second.C_ba();
    Eigen::Matrix<double, 6, 1>
        vec = qdata.T_0g_prior->second.vec();     // temporary stuff
    double yaw = atan2(C_vg(1, 0), C_vg(0, 0));
    double pitch = atan2(-C_vg(2, 0),
                         sqrt(C_vg(2, 1) * C_vg(2, 1)
                                  + C_vg(2, 2) * C_vg(2, 2)));
    double roll = atan2(C_vg(2, 1), C_vg(2, 2));
#if 1
    std::cout << "YAW:  " << yaw;
    std::cout << "  PITCH: " << pitch;
    std::cout << "  ROLL: " << roll << std::endl;
//    std::cout << "vec: " << vec.transpose() << std::endl;
#endif
    ypr_estimates_.push_back(std::vector<double>{yaw, pitch, roll,
                                                 (double) v->id().minorId(),
                                                 (double) v->keyFrameTime().nanoseconds_since_epoch
                                                     * 1e-9});
#endif
  }

  if (graph->numberOfVertices() == 488) {     // todo: very temporary
    std::ofstream outstream;
    outstream.open("/home/ben/Desktop/yprs.csv");
    for (auto ypr : ypr_estimates_) {
      outstream << ypr[0] << "," << ypr[1] << "," << ypr[2] << "," << ypr[3] << "," << std::setprecision(12) << ypr[4] << std::setprecision(6) << "\n";
    }
  }

  // reset to remove any old data from the problem setup
  resetProblem();
}

}  // namespace stereo
}  // namespace tactic
}  // namespace vtr