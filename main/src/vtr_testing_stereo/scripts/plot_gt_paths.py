
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns
from plot_odometry_groundtruth import read_gpgga

sns.set_style("whitegrid")

plt.rc('axes', labelsize=12, titlesize=14)
plt.rcParams["font.family"] = "serif"


def main():

    # # Run 1
    # groundtruth_dir = '${VTRDATA}/july5/gt/'
    # teach_gt_file = 'july5b.csv'
    # repeat_gt_file = 'july5c.csv'
    # teach_interval = (1625504640.0, 1625504721.8)  # todo: more than 60 seconds?
    # repeat_interval = (1625513807.4, 1625513898.7)
    # fig1 = plt.figure(1, figsize=[4.5, 4])
    # fig1.subplots_adjust(left=0.15, bottom=0.13, right=0.95, top=0.92)
    # plt.title('Path 1')

    # # Run 2
    # groundtruth_dir = '${VTRDATA}/july5/gt/'
    # teach_gt_file = 'july5c.csv'
    # repeat_gt_file = 'july5b.csv'
    # teach_interval = (1625514007.5, 1625514097.4)
    # repeat_interval = (1625504821.1, 1625504905.8)
    # fig1 = plt.figure(1, figsize=[4.5, 4])
    # fig1.subplots_adjust(left=0.13, bottom=0.13, right=0.95, top=0.92)
    # plt.title('Path 2')

    # Run 3
    groundtruth_dir = '${VTRDATA}/june16-gt/'
    teach_gt_file = 'june16a.csv'
    repeat_gt_file = 'june16b.csv'
    teach_interval = (1623894860.7, 1623894920.7)
    repeat_interval = (1623895050.0, 1623895110.0)
    fig1 = plt.figure(1, figsize=[9, 3.5])
    fig1.subplots_adjust(left=0.10, bottom=0.15, right=0.96, top=0.90)
    plt.title('Path 3')

    gt_teach_path = osp.join(osp.expanduser(osp.expandvars(groundtruth_dir)), teach_gt_file)
    gt_repeat_path = osp.join(osp.expanduser(osp.expandvars(groundtruth_dir)), repeat_gt_file)
    gt_teach = read_gpgga(gt_teach_path, 0, teach_interval[0], teach_interval[1])
    gt_repeat = read_gpgga(gt_repeat_path, 0, repeat_interval[0], repeat_interval[1])

    plt.axis('equal')

    plt.plot(gt_teach[:, 1] - gt_teach[0, 1], gt_teach[:, 2] - gt_teach[0, 2], c='k', label='Teach Path')
    plt.plot(gt_repeat[:, 1] - gt_teach[0, 1], gt_repeat[:, 2] - gt_teach[0, 2], c='C3', label='Repeat Path')

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
