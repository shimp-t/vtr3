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


def read_vo(vo_dir):
    with open(osp.join(vo_dir, "vo.csv"), newline='') as resultfile:
        spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')
        tmp = []
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                tmp.append([float(i) for i in row[3:6]])
                assert len(tmp[-1]) == 3
    return np.array(tmp)


def read_vo_transforms(data_dir):
    """Extract integrated VO transformations from teach run. Used to transform loc results"""
    vo_transforms = {}

    with open(osp.join(data_dir, "vo.csv"), newline='') as resultfile:
        spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')

        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                tmp = [float(i) for i in row[6:22]]
                assert len(tmp) == 16
                vo_transforms[int(row[2])] = np.array(tmp).reshape((4, 4), order='F')    # csv is column-major

    return vo_transforms


def read_loc(teach_dir, repeat_dir):
    """Robot position from composing integrated VO and localization"""

    vo_transforms = read_vo_transforms(teach_dir)

    with open(osp.join(repeat_dir, "loc.csv"), newline='') as resultfile:
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


def read_loc_sensor(teach_dir, repeat_dir, T_sv):
    """Returns positional localization estimate in the sensor frame"""

    teach_data = np.genfromtxt(osp.join(teach_dir, "vo.csv"), delimiter=',', skip_header=1)

    with open(osp.join(repeat_dir, "loc.csv"), newline='') as resultfile:
        spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')

        r_qsms_in_ms_set = np.empty((0, 8))
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            else:
                tmp = [float(i) for i in row[9:25]]
                T_qv_mv = np.array(tmp).reshape((4, 4), order='F')    # csv is column-major
                T_qs_ms = T_sv @ T_qv_mv @ np.linalg.inv(T_sv)
                T_ms_qs = np.linalg.inv(T_qs_ms)
                r_qs_ms = np.empty([8, 1])
                teach_time_idx = np.argmax(teach_data[:, 2] == float(row[4]))  # find time corresponding to map vertex
                r_qs_ms[0] = teach_data[teach_time_idx, 0]
                r_qs_ms[1] = float(row[0])   # repeat timestamp
                r_qs_ms[2] = T_ms_qs[0, 3]   # x
                r_qs_ms[3] = T_ms_qs[1, 3]   # y
                r_qs_ms[4] = T_ms_qs[2, 3]   # z
                r_qs_ms[5] = 0   # x gt (TBD)
                r_qs_ms[6] = 0   # y gt (TBD)
                r_qs_ms[7] = 0   # z gt (TBD)

                r_qsms_in_ms_set = np.append(r_qsms_in_ms_set, r_qs_ms.T, axis=0)

    return r_qsms_in_ms_set


def main():

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

    # Flags
    teach_dir = osp.expanduser("~/ASRL/temp/testing/stereo/")
    repeat_dir = osp.expanduser("~/ASRL/temp/testing/stereo/")

    r_teach = read_vo(teach_dir + "results_run_000000")

    r_repeat = {}
    r_qm = {}

    i = 1
    while osp.exists(repeat_dir + "results_run_" + str(i).zfill(6)):
        r_repeat[i], r_qm[i] = read_loc(teach_dir + "results_run_000000", repeat_dir + "results_run_" + str(i).zfill(6))
        i = i + 1

    print("Number of teach points: ", r_teach.shape[0])
    print("Number of points in first repeat: ", r_repeat[1].shape[0])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.axis('equal')
    ax.plot(r_teach[:, 0], r_teach[:, 1], label='Teach')

    for j, run in r_repeat.items():
        ax.plot(run[:, 0], run[:, 1], label='Repeat ' + str(j))

    plt.title('Overhead View of Integrated VO and Localization')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.legend()

    fig = plt.figure(2)
    latest = max(r_qm, key=int)
    plt.plot(r_qm[latest][:, 0], label='x')
    plt.plot(r_qm[latest][:, 1], label='y')
    plt.plot(r_qm[latest][:, 2], label='z')
    # for j, run in r_qm.items():
    #   plt.plot(run[:, 0], label='x_'+str(j), c='C0')      # plots all errors but messy
    # plt.plot(run[:, 1], label='y_'+str(j), c='C1')
    # plt.plot(run[:, 2], label='z_'+str(j), c='C2')
    plt.title('Estimated Path-Tracking Error')
    plt.ylabel('Distance (m)')
    plt.legend()

    # Calculate and plot errors of localization estimates wrt ground truth
    # Get localization estimates and times in GPS receiver frame
    T_sv = np.eye(4)
    r_loc_in_gps_frame = read_loc_sensor(teach_dir + "results_run_000000", repeat_dir + "results_run_000001", T_sv)

    # READ GROUND TRUTH
    groundtruth_dir = '${VTRDATA}/june16-gt/'
    teach_gt_file = 'june16b.csv'
    repeat_gt_file = 'june16a.csv'
    gt_teach_path = osp.join(osp.expanduser(osp.expandvars(groundtruth_dir)), teach_gt_file)
    gt_repeat_path = osp.join(osp.expanduser(osp.expandvars(groundtruth_dir)), repeat_gt_file)
    gt_teach = read_gpgga(gt_teach_path, 0)
    gt_repeat = read_gpgga(gt_repeat_path, 0)

    for i, row in enumerate(r_loc_in_gps_frame):
        t_teach = row[0]
        t_repeat = row[1]

        if t_teach < gt_teach[0, 0] or t_teach > gt_teach[-1, 0]:
            raise ValueError
        if t_repeat < gt_repeat[0, 0] or t_repeat > gt_repeat[-1, 0]:
            raise ValueError

        teach_idx = np.argmax(gt_teach[:, 0] > t_teach)
        t_0_t = gt_teach[teach_idx - 1, 0]
        t_1_t = gt_teach[teach_idx, 0]
        ratio_t = (t_teach - t_0_t) / (t_1_t - t_0_t)
        x_t = ratio_t * gt_teach[teach_idx - 1, 1] + (1 - ratio_t) * gt_teach[teach_idx, 1]
        y_t = ratio_t * gt_teach[teach_idx - 1, 2] + (1 - ratio_t) * gt_teach[teach_idx, 2]
        z_t = ratio_t * gt_teach[teach_idx - 1, 3] + (1 - ratio_t) * gt_teach[teach_idx, 3]

        theta_mg = math.atan2(gt_teach[teach_idx+4, 2] - gt_teach[teach_idx-5, 2], gt_teach[teach_idx+4, 1] - gt_teach[teach_idx-5, 1])

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
        r_loc_in_gps_frame[i, 6] = -1*math.sin(theta_mg) * dx_g + math.cos(theta_mg) * dy_g
        r_loc_in_gps_frame[i, 7] = dz_g

    fig3, ax3 = plt.subplots(nrows=2, ncols=1, figsize=[8, 6])
    fig3.subplots_adjust(left=0.10, bottom=0.06, right=0.96, top=0.93)
    plt.title("Estimated and Ground Truth Path-Tracking Errors in x and y")
    ax3[0].plot(r_loc_in_gps_frame[:, 2], c='C0', label='x - estimated')  # todo - clean up
    ax3[0].plot(r_loc_in_gps_frame[:, 5], c='C2', label='x - gt')
    ax3[0].plot(r_loc_in_gps_frame[:, 2] - r_loc_in_gps_frame[:, 5], c='C1', label='x - error')
    ax3[1].plot(r_loc_in_gps_frame[:, 3], c='C0', label='y - estimated')
    ax3[1].plot(r_loc_in_gps_frame[:, 6], c='C2', label='y - gt')
    ax3[1].plot(r_loc_in_gps_frame[:, 3] - r_loc_in_gps_frame[:, 6], c='C1', label='y - error')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
