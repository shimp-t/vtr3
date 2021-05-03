#!/usr/bin/env python

import csv
import os.path as osp
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def main():

  parser = argparse.ArgumentParser(description='Plot integrated VO from ModuleVO')
  parser.add_argument('--results_path', '-r', type=str, help='Parent directory containing run files.',
                      default='~/ASRL/vtr3g_offline_test/results/run_000000')
  parser.add_argument('--groundtruth_path', '-g', type=str, help='Path to CSV file with RTK ground truth (optional)',
                      default='~/CLionProjects/gpso/data/gpgga/feb15a_gga.ASC',)    # todo: split into dir, file so can grab dataset
  args = parser.parse_args()

  results_path = osp.expanduser(args.results_path)
  gt_available = False
  gt_path = ''
  if osp.exists(osp.expanduser(args.groundtruth_path)):
    gt_path = osp.expanduser(args.groundtruth_path)
    gt_available = True

  with open(osp.join(results_path, "vo.csv"), newline='') as result_file:
    reader = csv.reader(result_file, delimiter=',', quotechar='|')
    tmp = []
    for i, row in enumerate(reader):
      if i == 0:
        continue
      else:
        tmp.append([float(i) for i in row[3:6]])
        assert len(tmp[-1]) == 3

  r = np.array(tmp)
  print("Number of vertices: ", r.shape[0])

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(r[:, 0], r[:, 1])
  plt.title("Integrated VO")
  plt.xlabel("x [m]")
  plt.ylabel("y [m]")
  plt.axis('equal')

  if gt_available:
    # todo: read ground truth and plot on same plot
    print('to do - add RTK ground truth')
    pass

  plt.show()


if __name__ == '__main__':
  main()
