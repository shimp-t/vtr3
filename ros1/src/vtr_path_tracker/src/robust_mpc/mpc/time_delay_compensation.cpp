#include <vtr/path_tracker/robust_mpc/mpc/time_delay_compensation.h>

namespace vtr{
namespace path_tracker{

MpcTimeDelayComp::MpcTimeDelayComp(){
  cmd_hist.clear();
}

MpcTimeDelayComp::~MpcTimeDelayComp(){
}

// Utilities
int MpcTimeDelayComp::get_size(void){
  return cmd_hist.size();
}

// Functions
void MpcTimeDelayComp::clear_hist(void){
  cmd_hist.clear();
}

bool MpcTimeDelayComp::add_hist_entry(const float & v_cmd, const float & w_cmd, const ros::Time & ctrl_time){

  if (cmd_hist.size() > 0 && cmd_hist.back().ctrl_time > ctrl_time){
    LOG(WARNING) << "Time delay comp: Trying to add ctrl hist older than already in list. This is not supported.";
    return false;

  } else if (ctrl_time > ros::Time::now()){
    LOG(WARNING) << "Time delay comp: Trying to add ctrl hist from future. This is not supported.";
    return false;

  } else {
    cmd_hist_entry new_entry;
    new_entry.v_cmd = v_cmd;
    new_entry.w_cmd = w_cmd;
    new_entry.ctrl_time = ctrl_time;
    cmd_hist.push_back(new_entry);
  }
  return true;
}

bool MpcTimeDelayComp::get_cmd_list(const ros::Time & t_1, const ros::Time & t_2,
                                    std::vector< float > & v_cmd_vec,
                                    std::vector< float > & w_cmd_vec,
                                    std::vector< float > & dt_time_vec){

  std::vector < ros::Time > ctrl_time_vec;

  v_cmd_vec.clear();
  w_cmd_vec.clear();
  ctrl_time_vec.clear();
  dt_time_vec.clear();

  /** Ensure request is valid **/
  if (t_2 < t_1){
    LOG(DEBUG) << "Time delay comp (mpc): t_2 must be greater than t_1.";
    return false;

  } else if (cmd_hist.size() == 0){
    // No historic entries in cmd_hist
    return false;

  } else if (t_1 < cmd_hist.front().ctrl_time){
    // Requesting data older than there is in the cmd hist
    if (t_1 < cmd_hist.front().ctrl_time - ros::Duration(0.75)){
      // Delay is normal at start of path repeat or right after returning from pause,
      // so only show warning if delay is excessive
      LOG(INFO) << t_1 << ' ' << cmd_hist.front().ctrl_time;
      LOG(WARNING) << "Time delay comp (mpc): requesting data older than is in cmd hist.";
    }
    return false;

  } else if (ros::Time::now() + ros::Duration(1) < t_2){
    LOG(WARNING) << "Time delay comp (mpc): request t_2 is more than 1s into the future.";
    //return false;
  }

  /** Get indices of entries **/
  //std::vector<int> ind_pos, ind_neg;
  int ind_m1 = 0;

  for (uint32_t i = 1; i < cmd_hist.size(); i++){
    if (t_1 > cmd_hist[i].ctrl_time){
      ind_m1 = ind_m1 + 1;
    } else {
      break;
    }
  }

  int ind_m2 = ind_m1;
  for (uint32_t i = ind_m1; i < cmd_hist.size(); i++){
    if (i+1 < cmd_hist.size() && cmd_hist[i+1].ctrl_time < t_2){
      // Advance
      ind_m2 = ind_m2 + 1;
    } else {
      break;
    }
  }

  /** Double check **/
  if (cmd_hist[ind_m1].ctrl_time > t_1){
    LOG(WARNING) << "Time delay comp: Oops, something went wrong getting cmd list (1).";
    return false;
  } else if (cmd_hist[ind_m2].ctrl_time > t_2){
    LOG(WARNING) << "Time delay comp: Oops, something went wrong getting cmd list (2).";
    return false;
  }

  int num_entries = ind_m2 - ind_m1 + 1;

  v_cmd_vec.resize(num_entries);
  w_cmd_vec.resize(num_entries);
  ctrl_time_vec.resize(num_entries);
  dt_time_vec.resize(num_entries);

  int index = ind_m1;
  ros::Duration dt_ros;
  for (int i = 0; i < num_entries; i++){
    v_cmd_vec[i] = cmd_hist[index].v_cmd;
    w_cmd_vec[i] = cmd_hist[index].w_cmd;

    if (i == 0){
      ctrl_time_vec[i] = t_1;
    } else {
      ctrl_time_vec[i] = cmd_hist[index].ctrl_time;
      dt_ros = ctrl_time_vec[i] - ctrl_time_vec[i-1];
      dt_time_vec[i-1] = dt_ros.toSec();
      if (dt_ros.toSec() > 1.0){
        LOG(WARNING) << "Time delay compensation expects dt values to be < 1.0s.";
      }
    }
    index = index + 1;
  }

  dt_ros = t_2 - ctrl_time_vec[num_entries-1];
  dt_time_vec[num_entries-1] = dt_ros.toSec();
  if (dt_ros.toSec() > 1.0){
    LOG(WARNING) << "Time delay compensation expects dt values to be < 1.0s.";
  }
  return true;
}

bool MpcTimeDelayComp::get_avg_cmd(const ros::Time & t_1, const ros::Time & t_2, float & v_cmd_avg, float & w_cmd_avg){

  v_cmd_avg = 0;
  w_cmd_avg = 0;

  std::vector< float > v_cmd_vec, w_cmd_vec, dt_time_vec;

  /** Get list of relevant cmd entries **/
  get_cmd_list(t_1, t_2, v_cmd_vec, w_cmd_vec, dt_time_vec);

  float d_total = 0;
  float theta_total = 0;
  float t_total = 0;

  /** Compute cmd averages **/
  for (uint32_t i = 0; i < v_cmd_vec.size(); i++){
    d_total = d_total + v_cmd_vec[i]*dt_time_vec[i];
    theta_total = theta_total + w_cmd_vec[i]*dt_time_vec[i];
    t_total = t_total + dt_time_vec[i];
  }

  if (t_total > 0.01){
    v_cmd_avg = d_total / t_total;
    w_cmd_avg = theta_total / t_total;
  } else {
    return false;
  }
  return true;
}

bool MpcTimeDelayComp::del_hist_older_than(const ros::Time & t_1){

  /** Ensure request is valid **/
  if (cmd_hist.size() == 0 || t_1 < cmd_hist.front().ctrl_time){
    // Nothing to do
    return false;
  }

  /** Get indices of entries **/
  int num_to_delete = 0;

  for (uint32_t i = 1; i < cmd_hist.size(); i++){
    if (t_1 > cmd_hist[i].ctrl_time){
      num_to_delete = num_to_delete + 1;
    } else {
      break;
    }
  }

  if (num_to_delete > 0){
    for (int i = 0; i < num_to_delete - 1; i++){
      cmd_hist.pop_front();
    }
  }

  return true;
}

} // Pathtracker namespace
} // vtr namespace