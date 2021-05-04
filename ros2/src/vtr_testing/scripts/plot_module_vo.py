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
        tmp.append([float(i) for i in row[:6]])
        assert len(tmp[-1]) == 6

  r = np.array(tmp)
  print("Number of vertices: ", r.shape[0])

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

  fig = plt.figure(1, figsize=[8, 3])
  ax = fig.add_subplot(111)
  if not gt_available:
    ax.plot(r[:, 3], r[:, 4], c='b')
  plt.title("Integrated VO - {0} Run".format(dataset))
  plt.xlabel("x [m]")
  plt.ylabel("y [m]")
  plt.axis('equal')

  if gt_available:
    ax.plot(r_rot[:, 3] - r_rot[0, 3], r_rot[:, 4] - r_rot[0, 4], c='r')
    plt.plot(gt[:, 1] - gt[0, 1], gt[:, 2] - gt[0, 2], c='g')

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    plt.title("Errors")

  plt.show()


if __name__ == '__main__':
  main()
