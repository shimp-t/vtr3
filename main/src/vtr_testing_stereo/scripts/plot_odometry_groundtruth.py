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


def read_cpo(cpo_path, start_time=0.0, end_time=4999999999.9):
    estimates = np.genfromtxt(cpo_path, delimiter=',', skip_header=1)
    estimates = estimates[estimates[:, 0] >= start_time, :]
    estimates = estimates[estimates[:, 0] <= end_time, :]
    return estimates


def read_gpgga(gga_path, gps_day, start_time=0.0, end_time=4999999999.9):
    """Read file of ASCII GPGGA messages and return measurements as array in UTM coordinates"""

    day_seconds = UNIX_GPS_OFFSET + gps_day * 24 * 3600

    proj_origin = (43.7822845, -79.4661581, 169.642048)  # hardcoding for now - todo: get from ground truth CSV
    projection = Proj(
        "+proj=etmerc +ellps=WGS84 +lat_0={0} +lon_0={1} +x_0=0 +y_0=0 +z_0={2} +k_0=1".format(proj_origin[0],
                                                                                               proj_origin[1],
                                                                                               proj_origin[2]))
    with open(gga_path, newline='') as result_file:
        reader = csv.reader(result_file, delimiter=',', quotechar='|')
        tmp = []
        distance_along_path = 0
        for i, row in enumerate(reader):
            if row[0] == "$GPGGA":
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
            elif row[0] == "$NAVSAT":
                lat = safe_float(row[3])
                long = safe_float(row[4])
                z = safe_float(row[5]) + 37.2       # todo: altitude measured differently in NavSatFix vs $GPGGA
                x, y = projection(long, lat)
                fix_type = 4 if safe_float(row[6]) < 0.2 else 1         # rough
                timestamp = safe_float(row[1]) + 1e-9 * safe_float(row[2])

                # TODO: is this fair? rounding down to 10th of a second for comparison
                timestamp = 0.1 * math.floor(timestamp * 10.0)

                if start_time <= timestamp <= end_time:
                    if len(tmp) > 0:
                        prev_x = tmp[-1][1]
                        prev_y = tmp[-1][2]
                        dist_added = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        distance_along_path += dist_added

                    tmp.append([timestamp, x, y, z, fix_type, long, lat, distance_along_path])

    return np.array(tmp)


def interpolate_to_gt_times(gt, r):
    tmp = []
    for i, row in enumerate(gt):
        if row[0] < r[0, 0] or row[0] > r[-1, 0]:  # check that time in range we have ground truth
            continue

        idx = np.argmax(r[:, 0] > row[0])
        time_fraction = (row[0] - r[idx - 1, 0]) / (r[idx, 0] - r[idx - 1, 0])
        interp_vertex = r[idx - 1, 2] + time_fraction * (r[idx, 2] - r[idx - 1, 2])
        interp_x = r[idx - 1, 3] + time_fraction * (r[idx, 3] - r[idx - 1, 3])
        interp_y = r[idx - 1, 4] + time_fraction * (r[idx, 4] - r[idx - 1, 4])
        interp_z = r[idx - 1, 5] + time_fraction * (r[idx, 5] - r[idx - 1, 5])
        d = row[7]

        tmp.append([row[0], interp_x, interp_y, interp_z, d, 0,
                    interp_vertex])  # time, x, y, z, dist_along_path, yaw(TBD), ~vertex

    return np.array(tmp)


def rotate_all_runs(align_time, gt, gt_idx, vo_r_idxs, vo_rs_interp, vo_rs_rot_interp):
    for run, r in vo_rs_interp.items():
        r_idx = np.argmax(r[:, 0] >= align_time)
        vo_r_idxs[run] = r_idx

        theta_gt = math.atan2(gt[gt_idx, 2] - gt[0, 2], gt[gt_idx, 1] - gt[0, 1])
        theta_r = math.atan2(r[r_idx, 2] - r[0, 2], r[r_idx, 1] - r[0, 1])
        theta = theta_r - theta_gt
        c = math.cos(theta)
        s = math.sin(theta)

        vo_rs_rot_interp[run] = np.copy(r)  # copy estimates into new rotated array
        vo_rs_rot_interp[run][:, 1] = c * (r[:, 1] - r[0, 1]) + s * (r[:, 2] - r[0, 2])
        vo_rs_rot_interp[run][:, 2] = -s * (r[:, 1] - r[0, 1]) + c * (r[:, 2] - r[0, 2])


def estimate_yaws(r_rot_int):
    for i in range(len(r_rot_int) - 2):
        yaw = math.atan2(r_rot_int[i + 2, 2] - r_rot_int[i, 2], r_rot_int[i + 2, 1] - r_rot_int[i, 1])
        r_rot_int[i + 1, 5] = yaw
    r_rot_int[0, 5] = r_rot_int[1, 5]  # use neighbouring yaw estimate for endpoints
    r_rot_int[-1, 5] = r_rot_int[-2, 5]


def estimate_yaws_cpo(cpo_rot):
    for i in range(len(cpo_rot) - 2):
        yaw = math.atan2(cpo_rot[i + 2, 3] - cpo_rot[i, 3], cpo_rot[i + 2, 2] - cpo_rot[i, 2])
        cpo_rot[i + 1, 1] = yaw
    cpo_rot[0, 1] = cpo_rot[1, 1]  # use neighbouring yaw estimate for endpoints
    cpo_rot[-1, 1] = cpo_rot[-2, 1]


# result_files = ["vo.csv"]
result_files = ["5a_all.csv"]
run_colours = {result_files[0]: 'C3', "cpo": 'C1'}
run_labels = {result_files[0]: 'VO', "cpo": 'GPS Odometry'}


def main():
    # ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='Plot integrated VO from odometry_gps')
    parser.add_argument('--results_path', '-r', type=str, help='Parent directory containing run files.',
                        default='${VTRTEMP}/testing/stereo/results_run_000000')
    parser.add_argument('--groundtruth_dir', '-g', type=str, help='Path to directory with RTK ground truth (optional)',
                        # default='${VTRDATA}/june16-gt/')
                        default='${VTRDATA}/july5/gt/')
    parser.add_argument('--groundtruth_file', '-f', type=str, help='File name of RTK ground truth (optional)',
                        # default='june16b.csv')
                        default='july5a.csv')
    args = parser.parse_args()

    results_path = osp.expanduser(osp.expandvars(args.results_path))
    gt_path = osp.join(osp.expanduser(osp.expandvars(args.groundtruth_dir)), args.groundtruth_file)
    assert (osp.exists(gt_path))
    dataset = args.groundtruth_file[:7]

    # OPTIONS
    plot_xy_errors = False  # whether we want 3 subplots in error plot or just overall error
    plot_vehicle_frame_errors = False
    tdcp_period = 0.1
    start_trim = 2  # seconds to trim off start
    end_trim = 2
    align_distance = 10.0
    # cpo_path = osp.expanduser("~/Desktop/cpo_16b.csv")
    cpo_path = osp.expanduser("/home/ben/CLionProjects/ros2-ws/src/cpo_analysis/data/estimates/cpo.csv")
    cpo_available = osp.exists(cpo_path)

    # RESULT DICTIONARIES
    vo_rs = {}  # position estimates from each result/run                  # todo: better var names
    vo_rs_interp = {}  # position estimates interpolated to ground truth times
    vo_rs_rot_interp = {}  # interpolated position estimates rotated to align with ground truth frame
    vo_r_idxs = {}  # where in the position estimates the align point is
    cp_rs = {}  # the same stuff but for the "GPS edges"
    cp_rs_interp = {}
    cp_rs_rot_interp = {}
    cp_r_idxs = {}

    # READ IN TRANSFORMS FROM VO
    for run in result_files:
        results = np.genfromtxt(osp.join(results_path, run), delimiter=',', skip_header=1)
        print("Number of keyframes: ", results.shape[0])

        # Use poses and sensor vehicle transform to get position estimates of GPS receiver to compare to ground truth
        T_vs = np.eye(4)
        T_vs[0, 3] = 0.60
        T_vs[1, 3] = 0.00
        T_vs[2, 3] = 0.52
        tmp = []
        tmp2 = []
        for row in results:
            T_0v = np.reshape(row[6:22], (4, 4)).transpose()
            T_0s = T_0v @ T_vs
            tmp.append([row[0], row[1], row[2], T_0s[0, 3], T_0s[1, 3], T_0s[2, 3]])
            T_0v_cp = np.reshape(row[25:41], (4, 4)).transpose()
            T_0s_cp = T_0v_cp @ T_vs
            #            t, vid_maj, vid_min, x, y, z, success
            tmp2.append([row[0], row[1], row[2], T_0s_cp[0, 3], T_0s_cp[1, 3], T_0s_cp[2, 3], row[41]])
        vo_rs[run] = np.array(tmp)
        cp_rs[run] = np.array(tmp2)

    # GET TIME INTERVAL WE WANT TO WORK WITH
    first_time = math.ceil(vo_rs[result_files[0]][0, 0]) + start_trim
    last_time = math.floor(vo_rs[result_files[0]][-1, 0]) - end_trim

    # todo - find spot to check tf_gps_set flag in cp_rs but only in interval

    # READ GROUND TRUTH
    if dataset[:5] == "feb15":
        day = 2145 * 7 + 1  # Feb.15/21
    elif dataset[:4] == "june" or dataset[:4] == "july":
        day = 0         # don't need to know day to decode NavSatFix ground truth
    else:
        raise Exception("Unknown dataset - {0}".format(dataset))
    gt = read_gpgga(gt_path, day, start_time=first_time, end_time=last_time)

    # INTERPOLATE ESTIMATES AT KEYFRAME TIMES TO GROUND TRUTH TIMES
    for run, r in vo_rs.items():
        vo_rs_interp[run] = interpolate_to_gt_times(gt, r)
    for run, r in cp_rs.items():
        cp_rs_interp[run] = interpolate_to_gt_times(gt, r)

    # USE GROUND TRUTH TO ALIGN/ROTATE EACH VO RUN
    gt_idx = np.argmax(gt[:, 7] > align_distance)  # find first time we've travelled at least align_distance
    align_time = round(gt[gt_idx, 0])       # a little hacky (assumes we have ground truth at integer second times)
    print("Align time: {0}".format(align_time))

    rotate_all_runs(align_time, gt, gt_idx, vo_r_idxs, vo_rs_interp, vo_rs_rot_interp)
    rotate_all_runs(align_time, gt, gt_idx, cp_r_idxs, cp_rs_interp, cp_rs_rot_interp)

    # SET UP OVERHEAD PLOT
    fig = plt.figure(1, figsize=[9, 5])
    ax = fig.add_subplot(111)
    plt.title("Integrated VO - {0} Run".format(dataset))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')

    # READ AND OVERHEAD PLOT CPO ESTIMATES FROM CSV
    if cpo_available:
        cpo_estimates = read_cpo(cpo_path, start_time=first_time, end_time=last_time)
        if cpo_estimates[0, 0] > first_time:
            print("Warning: CPO estimates start after ground truth and VO start time. Consider increasing start_trim.")

        rotate_cpo = True
        if rotate_cpo:
            cpo_r_idx = np.argmax(cpo_estimates[:, 0] >= align_time)

            theta_gt = math.atan2(gt[gt_idx, 2] - gt[0, 2], gt[gt_idx, 1] - gt[0, 1])
            theta_r = math.atan2(cpo_estimates[cpo_r_idx, 3] - cpo_estimates[0, 3],
                                 cpo_estimates[cpo_r_idx, 2] - cpo_estimates[0, 2])
            theta = theta_r - theta_gt
            c = math.cos(theta)
            s = math.sin(theta)

            cpo_estimates_rot = np.copy(cpo_estimates)  # copy estimates into new rotated array
            cpo_estimates_rot[:, 2] = c * (cpo_estimates[:, 2] - cpo_estimates[0, 2]) + s * (cpo_estimates[:, 3] - cpo_estimates[0, 3])
            cpo_estimates_rot[:, 3] = -s * (cpo_estimates[:, 2] - cpo_estimates[0, 2]) + c * (cpo_estimates[:, 3] - cpo_estimates[0, 3])

            plt.plot(cpo_estimates_rot[:, 2] - cpo_estimates_rot[0, 2],
                     cpo_estimates_rot[:, 3] - cpo_estimates_rot[0, 3], label='GPS Odometry - Rotated', c='C1')
            ax.scatter(cpo_estimates_rot[cpo_r_idx, 2] - cpo_estimates_rot[0, 2],
                       cpo_estimates_rot[cpo_r_idx, 3] - cpo_estimates_rot[0, 3], c='C6')
        else:
            plt.plot(cpo_estimates[:, 2] - cpo_estimates[0, 2], cpo_estimates[:, 3] - cpo_estimates[0, 3],
                     label=run_labels["cpo"], c=run_colours["cpo"])

    # OVERHEAD PLOT THE ROTATED ESTIMATES
    for run, r_rot_int in vo_rs_rot_interp.items():
        ax.plot(r_rot_int[:, 1] - r_rot_int[0, 1], r_rot_int[:, 2] - r_rot_int[0, 2], c=run_colours[run],
                label='Rotated Estimates - {0}'.format(run_labels[run]))
        ax.scatter(r_rot_int[vo_r_idxs[run], 1] - r_rot_int[0, 1], r_rot_int[vo_r_idxs[run], 2] - r_rot_int[0, 2],
                   c=run_colours[run])
    for run, r_rot_int in cp_rs_rot_interp.items():
        ax.plot(r_rot_int[:, 1] - r_rot_int[0, 1], r_rot_int[:, 2] - r_rot_int[0, 2], c='k',
                label='GPS Edges'.format(run_labels[run]))
        ax.scatter(r_rot_int[vo_r_idxs[run], 1] - r_rot_int[0, 1], r_rot_int[vo_r_idxs[run], 2] - r_rot_int[0, 2],
                   c='k')
    ax.plot(gt[:, 1] - gt[0, 1], gt[:, 2] - gt[0, 2], c='C2', label='RTK Ground Truth')
    ax.scatter(gt[gt_idx, 1] - gt[0, 1], gt[gt_idx, 2] - gt[0, 2], c='C2')
    plt.legend()

    # SETUP ERROR PLOTS
    if plot_xy_errors or plot_vehicle_frame_errors:
        fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=[8, 8])
        fig2.subplots_adjust(left=0.10, bottom=0.06, right=0.96, top=0.93)
    else:
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=[8, 3])
        fig2.subplots_adjust(left=0.08, bottom=0.14, right=0.98, top=0.90)

    # CALCULATE AND PLOT ERRORS
    for run, r_rot_interp in vo_rs_rot_interp.items():
        assert (len(r_rot_interp) == len(gt))

        e_x = (r_rot_interp[:, 1] - r_rot_interp[0, 1]) - (gt[:, 1] - gt[0, 1])
        e_y = (r_rot_interp[:, 2] - r_rot_interp[0, 2]) - (gt[:, 2] - gt[0, 2])
        e_z = (r_rot_interp[:, 3] - r_rot_interp[0, 3]) - (gt[:, 3] - gt[0, 3])
        e_planar = np.sqrt(np.square(e_x) + np.square(e_y))

        if plot_xy_errors:
            ax2[0].plot(r_rot_interp[:, 4] - r_rot_interp[0, 4], e_x, c=run_colours[run])  # x errors
            ax2[1].plot(r_rot_interp[:, 4] - r_rot_interp[0, 4], e_y, c=run_colours[run])  # y errors
            ax2[2].plot(r_rot_interp[:, 4] - r_rot_interp[0, 4], e_planar, c=run_colours[run],
                        label=run_labels[run])  # planar errors
        elif plot_vehicle_frame_errors:
            # USE ESTIMATED YAWS TO CONVERT ERRORS TO VEHICLE FRAME (LONGITUDINAL/LATERAL)
            e_long = []
            e_lat = []
            for i, row in enumerate(r_rot_interp):
                yaw = r_rot_interp[i, 5]
                e_long.append(e_x[i] * math.cos(yaw) + e_y[i] * math.sin(yaw))
                e_lat.append(e_x[i] * -math.sin(yaw) + e_y[i] * math.cos(yaw))
            ax2[0].plot(r_rot_interp[:, 4] - r_rot_interp[0, 4], e_long, c=run_colours[run])  # x errors
            ax2[1].plot(r_rot_interp[:, 4] - r_rot_interp[0, 4], e_lat, c=run_colours[run])  # y errors
            ax2[2].plot(r_rot_interp[:, 4] - r_rot_interp[0, 4], e_planar, c=run_colours[run],
                        label=run_labels[run])  # planar errors
        else:
            ax2.plot(r_rot_interp[:, 4] - r_rot_interp[0, 4], e_planar, c=run_colours[run],
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
    elif plot_vehicle_frame_errors:
        ax2[0].set_title('Position Errors wrt Ground Truth - {0}'.format(dataset))
        ax2[2].set_xlabel('Distance Along Path (m)')
        ax2[0].set_ylabel('Longitudinal Error (m)')
        ax2[0].set_ylim([-3, 3])
        ax2[1].set_ylabel('Lateral Error (m)')
        ax2[1].set_ylim([-3, 3])
        ax2[2].set_ylabel('2D Position Error (m)')
        ax2[2].set_ylim([0, 3])
    else:
        ax2.set_title('Position Errors wrt Ground Truth - {0}'.format(dataset))
        ax2.set_xlabel('Distance Along Path (m)')
        ax2.set_ylabel('2D Position Error (m)')
        ax2.set_ylim([0, 3])

    if cpo_available:
        if rotate_cpo:
            cpo_estimates = cpo_estimates_rot  # todo: note - didn't bother copying code to plot both. switch here

        tmp = []
        for row in gt:
            idx_np = np.where(abs(cpo_estimates[:, 0] - row[0]) < 0.001)        # this off with new data
            if idx_np[0].size != 0:
                idx = safe_int(idx_np[0][0])
                tmp.append([cpo_estimates[idx, 0],  # GPS ref. timestamp
                            row[1],  # ground truth x (down-sampled)
                            row[2],  # "" y
                            row[3],  # "" z
                            (cpo_estimates[idx, 2] - cpo_estimates[0, 2]) - (row[1] - gt[0, 1]),  # estimate error x
                            (cpo_estimates[idx, 3] - cpo_estimates[0, 3]) - (row[2] - gt[0, 2]),  # "" y
                            (cpo_estimates[idx, 4] - cpo_estimates[0, 4]) - (row[3] - gt[0, 3]),  # "" z
                            row[7],  # distance along path
                            ])
        cpo_errors = np.array(tmp)
        if plot_xy_errors:
            ax2[0].plot(cpo_errors[:, 7] - cpo_errors[0, 7], cpo_errors[:, 4], c=run_colours["cpo"])  # x errors
            ax2[1].plot(cpo_errors[:, 7] - cpo_errors[0, 7], cpo_errors[:, 5], c=run_colours["cpo"])  # y errors
            ax2[2].plot(cpo_errors[:, 7] - cpo_errors[0, 7],
                        np.sqrt(cpo_errors[:, 4] ** 2 + cpo_errors[:, 5] ** 2), c=run_colours["cpo"],
                        label=run_labels["cpo"])  # planar errors
            for i, row in enumerate(cpo_errors):  # GPS availability
                if i < 1:
                    continue
                if cpo_errors[i, 0] - cpo_errors[i - 1, 0] > tdcp_period + 0.04:
                    ax2[0].plot([cpo_errors[i - 1, 7], cpo_errors[i, 7]], [0, 0], c='k')
                    ax2[1].plot([cpo_errors[i - 1, 7], cpo_errors[i, 7]], [0, 0], c='k')
                    ax2[2].plot([cpo_errors[i - 1, 7], cpo_errors[i, 7]], [0.01, 0.01], c='k')
        elif plot_vehicle_frame_errors:
            print("TODO")  # todo: use yaws to get these?
        else:
            ax2.plot(cpo_errors[:, 7] - cpo_errors[0, 7], np.sqrt(cpo_errors[:, 4] ** 2 + cpo_errors[:, 5] ** 2),
                     c=run_colours["cpo"], label=run_labels["cpo"])  # planar errors
            for i, row in enumerate(cpo_errors):  # GPS availability
                if i < 1:
                    continue
                if cpo_errors[i, 0] - cpo_errors[i - 1, 0] > tdcp_period + 0.04:
                    ax2.plot([cpo_errors[i - 1, 7], cpo_errors[i, 7]], [0.01, 0.01], c='k')

    plt.legend()

    # ESTIMATE YAW AT EACH VERTEX OF VO RUNS USING BEFORE/AFTER VERTEX
    plt.figure(3)
    plt.title("Heading vs Distance Along Path")
    for run, r_rot_int in vo_rs_rot_interp.items():
        estimate_yaws(r_rot_int)
        plt.plot(r_rot_int[:, 4] - r_rot_int[0, 4], r_rot_int[:, 5], c='C3')
    for run, r_rot_int in cp_rs_rot_interp.items():
        estimate_yaws(r_rot_int)
        plt.plot(r_rot_int[:, 4] - r_rot_int[0, 4], r_rot_int[:, 5], c='k')
    if cpo_available:
        estimate_yaws_cpo(cpo_estimates_rot)
        plt.plot(cpo_errors[:, 7] - cpo_errors[0, 7], cpo_estimates_rot[:, 1], c='C1')

    plt.show()


if __name__ == '__main__':
    main()
