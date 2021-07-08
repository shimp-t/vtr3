#!/usr/bin/env python

import csv
import os.path as osp

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from plot_odometry_groundtruth import read_gpgga

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# set sizes
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize


def read_vo(vo_dir, vo_file="vo.csv"):
    with open(osp.join(vo_dir, vo_file), newline='') as resultfile:
        spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')
        tmp = []
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                tmp.append([float(i) for i in row[3:6]])
                assert len(tmp[-1]) == 3
    return np.array(tmp)


def read_vo_transforms(data_dir, vo_file="vo.csv"):
    """Extract integrated VO transformations from teach run. Used to transform loc results"""
    vo_transforms = {}

    with open(osp.join(data_dir, vo_file), newline='') as resultfile:
        spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')

        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                tmp = [float(i) for i in row[6:22]]
                assert len(tmp) == 16
                vo_transforms[int(row[2])] = np.array(tmp).reshape((4, 4), order='F')    # csv is column-major

    return vo_transforms


def read_loc(teach_dir, repeat_dir, loc_file="loc.csv", teach_vo_file="vo.csv"):
    """Robot position from composing integrated VO and localization"""

    vo_transforms = read_vo_transforms(teach_dir, teach_vo_file)

    with open(osp.join(repeat_dir, loc_file), newline='') as resultfile:
        spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')
        r_world_set = np.empty((0, 4))
        r_qm_in_m_set = np.empty((0, 4))
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                r_loc = np.empty([4, 1])
                r_loc[0] = float(row[6])
                r_loc[1] = float(row[7])
                r_loc[2] = float(row[8])
                r_loc[3] = 1.0

                map_T = vo_transforms[int(row[4])]
                r_world = np.matmul(map_T, r_loc)
                r_world_set = np.append(r_world_set, r_world.T, axis=0)
                r_qm_in_m_set = np.append(r_qm_in_m_set, r_loc.T, axis=0)

    return r_world_set, r_qm_in_m_set


def read_loc_sensor(teach_dir, repeat_dir, T_sv, loc_file="loc.csv", teach_vo_file="vo.csv"):
    """Returns positional localization estimate in the (moving) sensor frame"""

    teach_data = np.genfromtxt(osp.join(teach_dir, teach_vo_file), delimiter=',', skip_header=1)

    with open(osp.join(repeat_dir, loc_file), newline='') as resultfile:
        spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')

        r_qsms_in_ms_set = np.empty((0, 9))
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                tmp = [float(i) for i in row[9:25]]
                T_qv_mv = np.array(tmp).reshape((4, 4), order='F')    # csv is column-major
                T_qs_ms = T_sv @ T_qv_mv @ np.linalg.inv(T_sv)
                T_ms_qs = np.linalg.inv(T_qs_ms)
                r_qs_ms = np.empty([9, 1])
                teach_time_idx = np.argmax(teach_data[:, 2] == float(row[4]))  # find time corresponding to map vertex
                r_qs_ms[0] = teach_data[teach_time_idx, 0]
                r_qs_ms[1] = float(row[0])   # repeat timestamp
                r_qs_ms[2] = T_ms_qs[0, 3]   # x estimated
                r_qs_ms[3] = T_ms_qs[1, 3]   # y estimated
                r_qs_ms[4] = T_ms_qs[2, 3]   # z estimated
                r_qs_ms[5] = 0   # x ground truth (TBD)
                r_qs_ms[6] = 0   # y ground truth (TBD)
                r_qs_ms[7] = 0   # z ground truth (TBD)
                r_qs_ms[8] = 0   # cumulative distance travelled based on teach ground truth (TBD)

                r_qsms_in_ms_set = np.append(r_qsms_in_ms_set, r_qs_ms.T, axis=0)

    return r_qsms_in_ms_set


def interpolate_and_rotate(gt_repeat, gt_teach, r_loc_in_gps_frame):
    first_teach_idx = 0
    for i, row in enumerate(r_loc_in_gps_frame):
        t_teach = row[0]
        t_repeat = row[1]

        if t_teach < gt_teach[0, 0] or t_teach > gt_teach[-1, 0]:
            raise ValueError
        if t_repeat < gt_repeat[0, 0] or t_repeat > gt_repeat[-1, 0]:
            raise ValueError

        teach_idx = np.argmax(gt_teach[:, 0] > t_teach)
        if i == 0:
            first_teach_idx = teach_idx

        t_0_t = gt_teach[teach_idx - 1, 0]
        t_1_t = gt_teach[teach_idx, 0]
        ratio_t = (t_teach - t_0_t) / (t_1_t - t_0_t)
        x_t = ratio_t * gt_teach[teach_idx - 1, 1] + (1 - ratio_t) * gt_teach[teach_idx, 1]
        y_t = ratio_t * gt_teach[teach_idx - 1, 2] + (1 - ratio_t) * gt_teach[teach_idx, 2]
        z_t = ratio_t * gt_teach[teach_idx - 1, 3] + (1 - ratio_t) * gt_teach[teach_idx, 3]

        theta_mg = math.atan2(gt_teach[teach_idx + 4, 2] - gt_teach[teach_idx - 5, 2],
                              gt_teach[teach_idx + 4, 1] - gt_teach[teach_idx - 5, 1])

        repeat_idx = np.argmax(gt_repeat[:, 0] > t_repeat)
        t_0_r = gt_repeat[repeat_idx - 1, 0]
        t_1_r = gt_repeat[repeat_idx, 0]
        ratio_r = (t_repeat - t_0_r) / (t_1_r - t_0_r)
        x_r = ratio_r * gt_repeat[repeat_idx - 1, 1] + (1 - ratio_r) * gt_repeat[repeat_idx, 1]
        y_r = ratio_r * gt_repeat[repeat_idx - 1, 2] + (1 - ratio_r) * gt_repeat[repeat_idx, 2]
        z_r = ratio_r * gt_repeat[repeat_idx - 1, 3] + (1 - ratio_r) * gt_repeat[repeat_idx, 3]

        dx_g = x_r - x_t
        dy_g = y_r - y_t
        dz_g = z_r - z_t
        # rotate localizations into vehicle frame (roughly)
        r_loc_in_gps_frame[i, 5] = math.cos(theta_mg) * dx_g + math.sin(theta_mg) * dy_g
        r_loc_in_gps_frame[i, 6] = -1 * math.sin(theta_mg) * dx_g + math.cos(theta_mg) * dy_g
        r_loc_in_gps_frame[i, 7] = dz_g

        r_loc_in_gps_frame[i, 8] = gt_teach[teach_idx, 7] - gt_teach[first_teach_idx, 7]


def main():

    teach_dir = osp.expanduser("~/ASRL/temp/testing/stereo/results_run_000000")
    repeat_dir = osp.expanduser("~/ASRL/temp/testing/stereo/results_run_000001")

    teach_vo_files = {0: "vo0_vis-exp2.csv", 1: "vo0_gps-exp2.csv"}
    repeat_loc_files = {0: "loc1_vis-exp2.csv", 1: "loc1_gps-exp2.csv"}
    colours = {0: ('C1', 'orange'), 1: ('C2', 'g')}
    labels = {0: "Vision Prior", 1: "GPS Prior"}
    r_teach = {}
    r_repeat = {}
    r_qm = {}

    for i in range(2):
        r_teach[i] = read_vo(teach_dir, vo_file=teach_vo_files[i])
        r_repeat[i], r_qm[i] = read_loc(teach_dir, repeat_dir, loc_file=repeat_loc_files[i], teach_vo_file=teach_vo_files[i])

    print("Number of points in first teach: ", r_teach[0].shape[0])
    print("Number of points in first repeat: ", r_repeat[0].shape[0])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.axis('equal')

    ax.plot(r_teach[0][:, 0], r_teach[0][:, 1], label='Teach', c='C7')
    for i, run in r_repeat.items():
        ax.plot(run[:, 0], run[:, 1], label='Repeat - {0}'.format(labels[i]), c=colours[i][0])

    plt.title('Overhead View of Integrated VO and Localization')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.legend()

    # CALCULATE AND PLOT ERRORS OF LOCALIZATION ESTIMATES WRT GROUND TRUTH
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=[8, 6])
    # fig3, ax3 = plt.subplots(nrows=2, ncols=1, figsize=[8, 6])
    # fig3.subplots_adjust(left=0.10, bottom=0.06, right=0.96, top=0.93)
    fig4 = plt.figure(4, figsize=[9, 4])

    # Read ground truth
    groundtruth_dir = '${VTRDATA}/june16-gt/'
    teach_gt_file = 'june16b.csv'
    repeat_gt_file = 'june16a.csv'
    gt_teach_path = osp.join(osp.expanduser(osp.expandvars(groundtruth_dir)), teach_gt_file)
    gt_repeat_path = osp.join(osp.expanduser(osp.expandvars(groundtruth_dir)), repeat_gt_file)
    gt_teach = read_gpgga(gt_teach_path, 0)
    gt_repeat = read_gpgga(gt_repeat_path, 0)
    T_sv = np.eye(4)
    T_sv[0, 3] = -0.60
    T_sv[1, 3] = 0.00
    T_sv[2, 3] = -0.52

    for i in range(2):
        # Get localization estimates and times in GPS receiver frame
        r_loc_in_gps_frame = read_loc_sensor(teach_dir, repeat_dir, T_sv, loc_file=repeat_loc_files[i], teach_vo_file=teach_vo_files[i])
        interpolate_and_rotate(gt_repeat, gt_teach, r_loc_in_gps_frame)

        plt.figure(2)
        ax2[0].plot(r_qm[i][:, 0], label='x - {0}'.format(labels[i]), c=colours[i][0])
        ax2[1].plot(r_qm[i][:, 1], label='y - {0}'.format(labels[i]), c=colours[i][0])
        # plt.plot(r_qm[i][:, 2], label='z - {0}'.format(labels[i]), c=colours[i][2])

        # plt.figure(3)
        # ax3[0].plot(r_loc_in_gps_frame[:, 8], r_loc_in_gps_frame[:, 2], c=colours[i][1], label='x - estimated')  # todo - fix colours
        # ax3[0].plot(r_loc_in_gps_frame[:, 8], r_loc_in_gps_frame[:, 5], c='k', label='x - gt')
        # ax3[0].plot(r_loc_in_gps_frame[:, 8], r_loc_in_gps_frame[:, 2] - r_loc_in_gps_frame[:, 5], c=colours[i][0], label='x - error')
        # ax3[1].plot(r_loc_in_gps_frame[:, 8], r_loc_in_gps_frame[:, 3], c=colours[i][2], label='y - estimated')
        # ax3[1].plot(r_loc_in_gps_frame[:, 8], r_loc_in_gps_frame[:, 6], c='k', label='y - gt')
        # ax3[1].plot(r_loc_in_gps_frame[:, 8], r_loc_in_gps_frame[:, 3] - r_loc_in_gps_frame[:, 6], c=colours[i][2], label='y - error')

        plt.figure(4)
        plt.plot(r_loc_in_gps_frame[:, 8], r_loc_in_gps_frame[:, 3] - r_loc_in_gps_frame[:, 6], c=colours[i][0], label=labels[i])

    plt.figure(2)
    ax2[0].set_title('Estimated Path-Tracking Errors')
    ax2[0].set_ylim([-1.2, 1.2])
    ax2[1].set_ylim([-1.2, 1.2])
    ax2[0].set_ylabel("Longitudinal Error (m)")
    ax2[1].set_ylabel("Lateral Error (m)")
    plt.xlabel('Distance Along Path (m)')
    plt.legend()

    # plt.figure(3)
    # plt.title("Estimated and Ground Truth Path-Tracking Errors in x and y")
    # ax3[0].legend()
    # ax3[1].legend()

    plt.figure(4)
    plt.title("Localization Estimate Errors")
    plt.ylim([-0.4, 0.4])
    plt.ylabel("Lateral Error wrt Ground Truth (m)")
    plt.xlabel("Distance Along Path (m)")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
