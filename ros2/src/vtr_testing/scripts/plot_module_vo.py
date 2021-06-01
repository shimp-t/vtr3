#!/usr/bin/env python

import csv
import os.path as osp

import math
import numpy as np
import argparse
from pyproj import Proj
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# difference between starts of Unix time (Jan.1/70) and GPS time (Jan.6/80)
UNIX_GPS_OFFSET = 315964800
LEAP_SECONDS = 18

sns.set_style("whitegrid")

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def safe_float(field):
  try:
    return float(field)
  except ValueError:
    return float('NaN')


def safe_int(field):
  try:
    return int(field)
  except ValueError:
    return 0


def read_gpgga(gga_path, gps_day, start_time=0.0, end_time=4999999999.9):
  """Read file of ASCII GPGGA messages and return measurements as array in UTM coordinates"""

  day_seconds = UNIX_GPS_OFFSET + gps_day * 24 * 3600

  proj_origin = (43.7822845, -79.4661581, 169.642048)  # hardcoding for now - to do: get from ground truth CSV

  projection = Proj(
    "+proj=etmerc +ellps=WGS84 +lat_0={0} +lon_0={1} +x_0=0 +y_0=0 +z_0={2} +k_0=1".format(proj_origin[0],
                                                                                           proj_origin[1],
                                                                                           proj_origin[2]))
  with open(gga_path, newline='') as result_file:
    reader = csv.reader(result_file, delimiter=',', quotechar='|')
    tmp = []
    distance_along_path = 0
    for i, row in enumerate(reader):
      if row[0] != "$GPGGA":
        continue

      lat_tmp = row[2]
      lat = safe_float(lat_tmp[0:2]) + safe_float(lat_tmp[2:]) / 60.0
      long_tmp = row[4]
      long = safe_float(long_tmp[0:3]) + safe_float(long_tmp[3:]) / 60.0
      if row[5] == 'W':
        long = -long
      z = safe_float(row[9])
      x, y = projection(long, lat)
      fix_type = safe_int(row[6])
      time_of_day = row[1]
      timestamp = day_seconds + safe_float(time_of_day[0:2]) * 3600.0 + safe_float(
        time_of_day[2:4]) * 60.0 + safe_float(time_of_day[4:])

      if start_time <= timestamp <= end_time:
        if len(tmp) > 0:
          prev_x = tmp[-1][1]
          prev_y = tmp[-1][2]
          dist_added = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
          distance_along_path += dist_added

        tmp.append([timestamp, x, y, z, fix_type, long, lat, distance_along_path])

  return np.array(tmp)


def main():
  parser = argparse.ArgumentParser(description='Plot integrated VO from ModuleVO')
  parser.add_argument('--results_path', '-r', type=str, help='Parent directory containing run files.',
                      default='~/ASRL/vtr3g_offline_test/results/run_000000')
  parser.add_argument('--groundtruth_dir', '-g', type=str, help='Path to directory with RTK ground truth (optional)',
                      default='~/CLionProjects/gpso/data/gpgga/')
  parser.add_argument('--groundtruth_file', '-f', type=str, help='File name of RTK ground truth (optional)',
                      default='feb15a_gga.ASC')
  args = parser.parse_args()

  results_path = osp.expanduser(args.results_path)
  gt_path = osp.join(osp.expanduser(args.groundtruth_dir), args.groundtruth_file)
  gt_available = osp.exists(gt_path)
  dataset = args.groundtruth_file[:6]

  with open(osp.join(results_path, "vo.csv"), newline='') as result_file:
    reader = csv.reader(result_file, delimiter=',', quotechar='|')
    tmp = []
    for i, row in enumerate(reader):
      if i == 0:
        continue
      else:
        tmp.append([float(i) for i in row[:22]])  # t, va, vb, r_v0_0, T_0v
        assert len(tmp[-1]) == 22

  results = np.array(tmp)
  print("Number of keyframes: ", results.shape[0])

  # Use poses and sensor vehicle transform to get position estimates of GPS receiver to compare to ground truth
  T_vs = np.eye(4)
  T_vs[0, 3] = 0.60
  T_vs[1, 3] = 0.00
  T_vs[2, 3] = 0.52
  tmp = []
  for row in results:
    T_0v = np.reshape(row[6:], (4, 4)).transpose()
    T_0s = T_0v @ T_vs
    tmp.append([row[0], row[1], row[2], T_0s[0, 3], T_0s[1, 3], T_0s[2, 3]])
  r = np.array(tmp)

  if gt_available:

    if dataset[:5] == "feb15":
      day = 2145 * 7 + 1  # Feb.15/21
    else:
      raise Exception("Unknown dataset - {0}".format(dataset))

    # read ground truth into array
    gt = read_gpgga(gt_path, day, start_time=r[0, 0], end_time=r[-1, 0])

    # rotate VO
    align_distance = 10.0
    gt_idx = np.argmax(gt[:, 7] > align_distance)  # find first time we've travelled at least align_distance
    align_time = gt[gt_idx, 0]
    r_idx = np.argmax(r[:, 0] > align_time)
    if r[r_idx, 0] - align_time > align_time - r[r_idx - 1, 0]:
      r_idx -= 1

    theta_gt = math.atan2(gt[gt_idx, 2] - gt[0, 2], gt[gt_idx, 1] - gt[0, 1])
    theta_r = math.atan2(r[r_idx, 4] - r[0, 4], r[r_idx, 3] - r[0, 3])
    theta = theta_r - theta_gt
    c = math.cos(theta)
    s = math.sin(theta)

    r_rot = np.copy(r)  # copy estimates into new rotated array
    r_rot[:, 3] = c * r[:, 3] + s * r[:, 4]
    r_rot[:, 4] = -s * r[:, 3] + c * r[:, 4]

  fig = plt.figure(1, figsize=[9, 5])
  ax = fig.add_subplot(111)
  if not gt_available:
    ax.plot(r[:, 3], r[:, 4], c='b')
  plt.title("Integrated VO - {0} Run".format(dataset))
  plt.xlabel("x [m]")
  plt.ylabel("y [m]")
  plt.axis('equal')

  tmp = []
  for i in range(len(r_rot) - 2):
    yaw = math.atan2(r_rot[i + 2, 4] - r_rot[i, 4], r_rot[i + 2, 3] - r_rot[i, 3])
    tmp.append(yaw)
  vo_yaws = np.array(tmp)

  if gt_available:
    ax.plot(r_rot[:, 3] - r_rot[0, 3], r_rot[:, 4] - r_rot[0, 4], c='r', label='Rotated Estimates')
    ax.plot(gt[:, 1] - gt[0, 1], gt[:, 2] - gt[0, 2], c='g', label='RTK Ground Truth')
    ax.scatter(r_rot[r_idx, 3] - r_rot[0, 3], r_rot[r_idx, 4] - r_rot[0, 4], c='r')
    ax.scatter(gt[gt_idx, 1] - gt[0, 1], gt[gt_idx, 2] - gt[0, 2], c='g')
    plt.legend()

    # interpolate ground truth to keyframe times
    tmp = []
    for i, row in enumerate(r_rot):
      if row[0] < gt[0, 0] or row[0] > gt[-1, 0]:  # check that time in range we have ground truth
        continue

      idx = np.argmax(gt[:, 0] > row[0])
      time_fraction = (row[0] - gt[idx - 1, 0]) / (gt[idx, 0] - gt[idx - 1, 0])
      interp_x = gt[idx - 1, 1] + time_fraction * (gt[idx, 1] - gt[idx - 1, 1])
      interp_y = gt[idx - 1, 2] + time_fraction * (gt[idx, 2] - gt[idx - 1, 2])
      interp_z = gt[idx - 1, 3] + time_fraction * (gt[idx, 3] - gt[idx - 1, 3])
      interp_d = gt[idx - 1, 7] + time_fraction * (gt[idx, 7] - gt[idx - 1, 7])

      tmp.append([row[0], row[1], row[2], interp_x, interp_y, interp_z, interp_d])

    gt_interp = np.array(tmp)

    last_r_idx = len(gt_interp) - len(r_rot) + 1      # hacky
    assert (last_r_idx <= -1)

    e_x = (gt_interp[:, 3] - gt_interp[0, 3]) - (r_rot[1:last_r_idx, 3] - r_rot[1, 3])
    e_y = (gt_interp[:, 4] - gt_interp[0, 4]) - (r_rot[1:last_r_idx, 4] - r_rot[1, 4])
    e_z = (gt_interp[:, 5] - gt_interp[0, 5]) - (r_rot[1:last_r_idx, 5] - r_rot[1, 5])
    e_planar = np.sqrt(np.square(e_x) + np.square(e_y))

    fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=[8, 8])
    fig2.subplots_adjust(left=0.10, bottom=0.06, right=0.96, top=0.93)
    ax2[0].plot(gt_interp[:, 6] - gt_interp[0, 6], e_x)  # x errors
    ax2[1].plot(gt_interp[:, 6] - gt_interp[0, 6], e_y)  # y errors
    ax2[2].plot(gt_interp[:, 6] - gt_interp[0, 6], e_planar, c='C0')  # planar errors

    ax2[0].set_title('Position Errors wrt Ground Truth - {0}'.format(dataset))
    ax2[2].set_xlabel('Distance Along Path (m)')
    ax2[0].set_ylabel('x Error (m)')
    ax2[0].set_ylim([-3, 3])
    ax2[1].set_ylabel('y Error (m)')
    ax2[1].set_ylim([-3, 3])
    ax2[2].set_ylabel('2D Position Error (m)')
    ax2[2].set_ylim([0, 4])

  plt.figure(3)   # temporary
  plt.title("VTR3 with TDCP")
  yprs = np.genfromtxt("/home/ben/Desktop/yprs.csv", delimiter=',')
  if len(yprs) > 2:
  #   plt.plot(yprs[:, 4] - yprs[0, 4], -yprs[:, 0], label='Yaw')
  #   plt.plot(yprs[:, 4] - yprs[0, 4], -yprs[:, 1], label='Pitch')
  #   plt.plot(yprs[:, 4] - yprs[0, 4], -yprs[:, 2], label='Roll')
  # plt.plot(yprs[:480, 4] - yprs[0, 4], vo_yaws[:480], label='Yaw from integrated VO', c='C3')   # messy
    plt.plot(yprs[:, 3] + 1, -yprs[:, 0], label='Yaw')
    plt.plot(yprs[:, 3] + 1, -yprs[:, 1], label='Pitch')
    plt.plot(yprs[:, 3] + 1, -yprs[:, 2], label='Roll')
  plt.plot(vo_yaws[:490], label='Yaw from integrated VO', c='C3')   # messy
  plt.xlabel("Vertex")
  plt.ylabel("Estimated Angle (rad)")
  plt.legend()

  plt.show()


if __name__ == '__main__':
  main()
