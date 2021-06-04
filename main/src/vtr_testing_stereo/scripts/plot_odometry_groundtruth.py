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
    parser = argparse.ArgumentParser(description='Plot integrated VO from odometry_gps')
    parser.add_argument('--results_path', '-r', type=str, help='Parent directory containing run files.',
                        default='${VTRTEMP}/testing/stereo/results_run_000000')
    parser.add_argument('--groundtruth_dir', '-g', type=str, help='Path to directory with RTK ground truth (optional)',
                        default='${VTRDATA}/feb_gps/groundtruth/')
    parser.add_argument('--groundtruth_file', '-f', type=str, help='File name of RTK ground truth (optional)',
                        default='feb15c_gga.ASC')
    args = parser.parse_args()

    results_path = osp.expanduser(osp.expandvars(args.results_path))
    gt_path = osp.join(osp.expanduser(osp.expandvars(args.groundtruth_dir)), args.groundtruth_file)
    gt_available = osp.exists(gt_path)
    dataset = args.groundtruth_file[:6]
    plot_xy_errors = False  # whether we want 3 subplots in error plot or just overall error

    result_files = ["vo_vis_c.csv", "vo_gps_c.csv", "cascade_c.csv"]
    run_colours = {result_files[0]: 'C3', result_files[1]: 'C0', result_files[2]: 'C4'}
    run_labels = {result_files[0]: 'Only Vision', result_files[1]: 'With GPS', result_files[2]: 'Cascaded'}
    rs = {}  # position estimates from each result/run                  # todo: better var names
    rs_rot = {}  # position estimates rotated to align with ground truth
    rs_rot_interp = {}  # rotated position estimates interpolated to ground truth times
    vo_yaws = {}  # approximate yaw at each point in rotated vo
    r_idxs = {}  # where in the position estimates the rotation point is

    for run in result_files:
        with open(osp.join(results_path, run), newline='') as result_file:
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
        rs[run] = np.array(tmp)

    if gt_available:

        if dataset[:5] == "feb15":
            day = 2145 * 7 + 1  # Feb.15/21
        else:
            raise Exception("Unknown dataset - {0}".format(dataset))

        # read ground truth into array (grabbing arbitrary result to get time interval)
        gt = read_gpgga(gt_path, day, start_time=(list(rs.values())[0])[0, 0], end_time=(list(rs.values())[0])[-1, 0])

        # rotate VO
        align_distance = 10.0
        gt_idx = np.argmax(gt[:, 7] > align_distance)  # find first time we've travelled at least align_distance
        align_time = gt[gt_idx, 0]

        for run, r in rs.items():
            r_idx = np.argmax(r[:, 0] > align_time)
            if r[r_idx, 0] - align_time > align_time - r[r_idx - 1, 0]:
                r_idx -= 1
            r_idxs[run] = r_idx

            theta_gt = math.atan2(gt[gt_idx, 2] - gt[0, 2], gt[gt_idx, 1] - gt[0, 1])
            theta_r = math.atan2(r[r_idx, 4] - r[0, 4], r[r_idx, 3] - r[0, 3])
            theta = theta_r - theta_gt
            c = math.cos(theta)
            s = math.sin(theta)

            rs_rot[run] = np.copy(r)  # copy estimates into new rotated array
            rs_rot[run][:, 3] = c * r[:, 3] + s * r[:, 4]
            rs_rot[run][:, 4] = -s * r[:, 3] + c * r[:, 4]

    fig = plt.figure(1, figsize=[9, 5])
    ax = fig.add_subplot(111)
    if not gt_available:
        for run, r in rs.items():
            ax.plot(r[:, 3], r[:, 4], c=run_colours[run])
    plt.title("Integrated VO - {0} Run".format(dataset))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')

    for run, r_rot in rs_rot.items():
        tmp = []
        for i in range(len(r_rot) - 2):
            yaw = math.atan2(r_rot[i + 2, 4] - r_rot[i, 4], r_rot[i + 2, 3] - r_rot[i, 3])
            tmp.append(yaw)
        vo_yaws[run] = np.array(tmp)

    if gt_available:
        for run, r_rot in rs_rot.items():
            ax.plot(r_rot[:, 3] - r_rot[0, 3], r_rot[:, 4] - r_rot[0, 4], c=run_colours[run],
                    label='Rotated Estimates - {0}'.format(run_labels[run]))
            ax.scatter(r_rot[r_idxs[run], 3] - r_rot[0, 3], r_rot[r_idxs[run], 4] - r_rot[0, 4], c=run_colours[run])
        ax.plot(gt[:, 1] - gt[0, 1], gt[:, 2] - gt[0, 2], c='C2', label='RTK Ground Truth')
        ax.scatter(gt[gt_idx, 1] - gt[0, 1], gt[gt_idx, 2] - gt[0, 2], c='C2')
        plt.legend()

        for run, r_rot in rs_rot.items():
            # interpolate ground truth to keyframe times
            tmp = []
            for i, row in enumerate(gt):
                if row[0] < r_rot[0, 0] or row[0] > r_rot[-1, 0]:  # check that time in range we have ground truth
                    continue

                idx = np.argmax(r_rot[:, 0] > row[0])
                time_fraction = (row[0] - r_rot[idx - 1, 0]) / (r_rot[idx, 0] - r_rot[idx - 1, 0])
                interp_x = r_rot[idx - 1, 3] + time_fraction * (r_rot[idx, 3] - r_rot[idx - 1, 3])
                interp_y = r_rot[idx - 1, 4] + time_fraction * (r_rot[idx, 4] - r_rot[idx - 1, 4])
                interp_z = r_rot[idx - 1, 5] + time_fraction * (r_rot[idx, 5] - r_rot[idx - 1, 5])
                d = row[7]

                tmp.append([row[0], row[1], row[2], interp_x, interp_y, interp_z, d])  # first 3 elements unused

            rs_rot_interp[run] = np.array(tmp)

        if plot_xy_errors:
            fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=[8, 8])
            fig2.subplots_adjust(left=0.10, bottom=0.06, right=0.96, top=0.93)
        else:
            fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=[8, 3])
            fig2.subplots_adjust(left=0.08, bottom=0.14, right=0.98, top=0.90)
        for run, r_rot_interp in rs_rot_interp.items():
            assert (len(r_rot_interp) == len(gt))

            e_x = (r_rot_interp[:, 3] - r_rot_interp[0, 3]) - (gt[:, 1] - gt[0, 1])
            e_y = (r_rot_interp[:, 4] - r_rot_interp[0, 4]) - (gt[:, 2] - gt[0, 2])
            e_z = (r_rot_interp[:, 5] - r_rot_interp[0, 5]) - (gt[:, 3] - gt[0, 3])
            e_planar = np.sqrt(np.square(e_x) + np.square(e_y))

            if plot_xy_errors:
                ax2[0].plot(r_rot_interp[:, 6] - r_rot_interp[0, 6], e_x, c=run_colours[run])  # x errors
                ax2[1].plot(r_rot_interp[:, 6] - r_rot_interp[0, 6], e_y, c=run_colours[run])  # y errors
            else:
                ax2.plot(r_rot_interp[:, 6] - r_rot_interp[0, 6], e_planar, c=run_colours[run],
                         label=run_labels[run])  # planar errors

        if plot_xy_errors:
            ax2[0].set_title('Position Errors wrt Ground Truth - {0}'.format(dataset))
            ax2[2].set_xlabel('Distance Along Path (m)')
            ax2[0].set_ylabel('x Error (m)')
            ax2[0].set_ylim([-3, 3])
            ax2[1].set_ylabel('y Error (m)')
            ax2[1].set_ylim([-3, 3])
            ax2[2].set_ylabel('2D Position Error (m)')
            ax2[2].set_ylim([0, 3])
        else:
            ax2.set_title('Position Errors wrt Ground Truth - {0}'.format(dataset))
            ax2.set_xlabel('Distance Along Path (m)')
            ax2.set_ylabel('2D Position Error (m)')
            ax2.set_ylim([0, 3])
        plt.legend()

        plt.figure(3, figsize=[8, 4])  # temporary
        plt.title("VTR3 with TDCP")
        yprs = []
        try:
            yprs = np.genfromtxt("/home/ben/Desktop/yprs.csv", delimiter=',')
        except IOError:
            print("No angle estimates found.")
        if len(yprs) > 2:
            plt.plot(yprs[:, 3] + 1, -yprs[:, 0], label='Yaw Estimated', c='C4')
            plt.plot(yprs[:, 3] + 1, -yprs[:, 1], label='Pitch Estimated', c='C1')
            plt.plot(yprs[:, 3] + 1, -yprs[:, 2], label='Roll Estimated', c='C2')
        plt.plot(vo_yaws[result_files[0]][:490], label='Yaw from integrated VO', c='C5')  # todo: messy
        plt.xlabel("Vertex")
        plt.ylabel("Estimated Angle (rad)")
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
