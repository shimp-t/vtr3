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

plt.rc('axes', labelsize=12, titlesize=14)
plt.rcParams["font.family"] = "serif"


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
        r_qm_in_m_set = np.empty((0, 8))
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                r_loc = np.empty([8, 1])
                r_loc[0] = float(row[6])
                r_loc[1] = float(row[7])
                r_loc[2] = float(row[8])
                r_loc[3] = 1.0
                r_loc[4] = row[0]  # repeat vertex timestamp
                r_loc[5] = row[2]  # repeat vertex minor id
                r_loc[6] = row[4]  # teach vertex minor id
                r_loc[7] = row[5]  # localization successful?

                map_T = vo_transforms[int(row[4])]
                r_world = np.matmul(map_T, r_loc[:4])
                r_world_set = np.append(r_world_set, r_world.T, axis=0)
                r_qm_in_m_set = np.append(r_qm_in_m_set, r_loc.T, axis=0)

    return r_world_set, r_qm_in_m_set


def read_loc_sensor(teach_dir, repeat_dir, T_sv, loc_file="loc.csv", teach_vo_file="vo.csv"):
    """Returns positional localization estimate in the (moving) sensor frame"""

    teach_data = np.genfromtxt(osp.join(teach_dir, teach_vo_file), delimiter=',', skip_header=1)

    with open(osp.join(repeat_dir, loc_file), newline='') as resultfile:
        spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')

        r_qsms_in_ms_set = np.empty((0, 10))
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                tmp = [float(i) for i in row[9:25]]
                T_qv_mv = np.array(tmp).reshape((4, 4), order='F')    # csv is column-major
                T_qs_ms = T_sv @ T_qv_mv @ np.linalg.inv(T_sv)
                T_ms_qs = np.linalg.inv(T_qs_ms)
                r_qs_ms = np.empty([10, 1])
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
                r_qs_ms[9] = row[5]   # localization successful

                r_qsms_in_ms_set = np.append(r_qsms_in_ms_set, r_qs_ms.T, axis=0)

    return r_qsms_in_ms_set


def interpolate_and_rotate(gt_repeat, gt_teach, r_loc_in_gps_frame):
    """Interpolates RTK ground truth for both localization (repeat - teach) to keyframe times and roughly rotates it
    into the local vehicle frame"""

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
        # rotate ground truth localizations from ENU into vehicle frame (roughly)
        r_loc_in_gps_frame[i, 5] = math.cos(theta_mg) * dx_g + math.sin(theta_mg) * dy_g
        r_loc_in_gps_frame[i, 6] = -1 * math.sin(theta_mg) * dx_g + math.cos(theta_mg) * dy_g
        r_loc_in_gps_frame[i, 7] = dz_g

        r_loc_in_gps_frame[i, 8] = gt_teach[teach_idx, 7] - gt_teach[first_teach_idx, 7]


def main():

    teach_dir = osp.expanduser("~/ASRL/temp/testing/stereo/results_run_000000")
    repeat_dir = osp.expanduser("~/ASRL/temp/testing/stereo/results_run_000001")

    # teach_vo_files = {0: "vo0_vis-exp2b.csv", 1: "vo.csv"}
    # repeat_loc_files = {0: "loc1_vis-exp2b.csv", 1: "loc.csv"}
    teach_vo_files = {0: "vo0-exp1-vis.csv", 1: "vo0-exp1-gps.csv"}
    repeat_vo_files = {0: "vo1-exp1-vis.csv", 1: "vo1-exp1-gps.csv"}
    repeat_loc_files = {0: "loc1-exp1-vis.csv", 1: "loc1-exp1-gps.csv"}
    # teach_vo_files = {0: "vo0-exp2-vis.csv", 1: "vo0-exp2-gps.csv"}
    # repeat_vo_files = {0: "vo1-exp2-vis.csv", 1: "vo1-exp2-gps.csv"}
    # repeat_loc_files = {0: "loc1-exp2-vis.csv", 1: "loc1-exp2-gps.csv"}
    # teach_vo_files = {0: "vo0-exp3-vis.csv", 1: "vo0-exp3-gps.csv"}
    # repeat_vo_files = {0: "vo1-exp3-vis.csv", 1: "vo1-exp3-gps.csv"}
    # repeat_loc_files = {0: "loc1-exp3-vis.csv", 1: "loc1-exp3-gps.csv"}
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

    fig1 = plt.figure(1, figsize=[9, 3.5])
    ax = fig1.add_subplot(111)
    fig1.subplots_adjust(left=0.10, bottom=0.15, right=0.96, top=0.90)
    plt.axis('equal')

    ax.plot(r_teach[0][:, 0], r_teach[0][:, 1], label='Teach', c='C7')
    for i, run in r_repeat.items():
        ax.plot(run[:, 0], run[:, 1], label='Repeat - {0}'.format(labels[i]), c=colours[i][0])
        # ax.scatter(run[:, 0], run[:, 1], label='Repeat - {0}'.format(labels[i]), c=colours[i][0])

    plt.title('Overhead View of Integrated VO and Localization')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.legend()

    # CALCULATE AND PLOT ERRORS OF LOCALIZATION ESTIMATES WRT GROUND TRUTH
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=[8, 6])
    fig3, ax3 = plt.subplots(nrows=2, ncols=1, figsize=[9, 5])
    fig3.subplots_adjust(left=0.10, bottom=0.10, right=0.96, top=0.92, hspace=0.30)

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
        ax2[0].plot(r_loc_in_gps_frame[:, 8], r_qm[i][:, 0], label='x - {0}'.format(labels[i]), c=colours[i][0])  # x-axis distance for final plots
        ax2[1].plot(r_loc_in_gps_frame[:, 8], r_qm[i][:, 1], label='y - {0}'.format(labels[i]), c=colours[i][0])
        # ax2[0].plot(r_qm[i][:, 4] - 1623800000, r_qm[i][:, 0], label='x - {0}'.format(labels[i]), c=colours[i][0])    # x-axis timestamp - 1623800000 for debugging
        # ax2[1].plot(r_qm[i][:, 4] - 1623800000, r_qm[i][:, 1], label='y - {0}'.format(labels[i]), c=colours[i][0])

        plt.figure(3)
        # plt.plot(r_loc_in_gps_frame[:, 8], r_loc_in_gps_frame[:, 2] - r_loc_in_gps_frame[:, 5], c=colours[i][0], label=labels[i])   # longitudinal errors (noisy)
        ax3[0].plot(r_loc_in_gps_frame[:, 8], abs(r_loc_in_gps_frame[:, 3] - r_loc_in_gps_frame[:, 6]), c=colours[i][0], label=labels[i])

    # plot sensor availability
    repeat_vo = np.genfromtxt(osp.join(repeat_dir, repeat_vo_files[1]), delimiter=',', skip_header=1)
    for j, row in enumerate(r_loc_in_gps_frame):
        if j == 0:
            continue
        assert(np.argmax(repeat_vo[:, 0] >= row[1]) == j)  # make sure times line up between repeat vo and loc csv files
        vo_estimated = repeat_vo[j, 3] != repeat_vo[j - 1, 3]  # todo: a little hacky but not saving VO success flag now
        c = 'C4' if vo_estimated else 'w'
        ax3[1].barh(1.5, r_loc_in_gps_frame[j, 8] - r_loc_in_gps_frame[j - 1, 8], height=1.0, left=r_loc_in_gps_frame[j - 1, 8], color=c, edgecolor=c)
        v_loc_successful = row[9]
        c = 'C1' if v_loc_successful else 'w'
        ax3[1].barh(0.5, r_loc_in_gps_frame[j, 8] - r_loc_in_gps_frame[j - 1, 8], height=1.0, left=r_loc_in_gps_frame[j - 1, 8], color=c, edgecolor=c)

    plt.figure(2)
    ax2[0].set_title('Estimated Path-Tracking Errors')
    ax2[0].set_ylim([-1.2, 1.2])
    ax2[1].set_ylim([-1.2, 1.2])
    ax2[0].set_ylabel("Longitudinal Error (m)")
    ax2[1].set_ylabel("Lateral Error (m)")
    plt.xlabel('Distance Along Path (m)')
    # plt.xlabel('Timestamp - 1623800000')
    plt.legend()

    ax3[0].set_title("Localization Estimate Errors wrt Ground Truth ")
    ax3[0].set_ylim([0, 0.4])
    ax3[0].set_ylabel("Absolute Lateral Error(m)")
    ax3[1].patch.set_visible(False)
    ax3[1].set_yticks([])
    ax3[1].set_ylim([0, 2])
    ax3[0].set_xlim([-2, 65])
    ax3[1].set_xlim([-2, 65])
    ax3[1].text(-8, 1.5, 'VO', fontsize=12)
    ax3[1].text(-8, 0.5, 'Vision', fontsize=12)
    ax3[0].set_xlabel("Distance Along Teach Path (m)")
    ax3[1].set_xlabel("Distance Along Teach Path (m)")
    ax3[0].legend()

    plt.show()


if __name__ == '__main__':
    main()
