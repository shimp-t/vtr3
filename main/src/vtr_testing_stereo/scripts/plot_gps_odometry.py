#!/usr/bin/env python

import csv
import os.path as osp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

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


def main():

  results_dir = osp.expanduser(osp.expandvars("${VTRTEMP}/testing/stereo/results_run_000000"))

  with open(osp.join(results_dir, "vo.csv"), newline='') as resultfile:
    spamreader = csv.reader(resultfile, delimiter=',', quotechar='|')
    tmp = []
    tmp_gps = []
    for i, row in enumerate(spamreader):
      if i == 0:
        continue
      else:
        tmp.append([float(i) for i in row[3:6]])
        tmp_gps.append([float(i) for i in row[22:26]])
        assert len(tmp[-1]) == 3

  r = np.array(tmp)           # start at different points so initial poses different
  r_gps = np.array(tmp_gps)
  print("Number of points: ", r.shape[0])

  # quick way to trim off part where we don't have GPS
  start_trim = np.argmax(r_gps[:, 0] != 0)
  r = r[start_trim:, :]
  r_gps = r_gps[start_trim:, :]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(r[:, 0] - r[0, 0], r[:, 1] - r[0, 1], label="VO edges")
  ax.plot(r_gps[:, 0] - r_gps[0, 0], r_gps[:, 1] - r_gps[0, 1], label="GPS edges")
  plt.axis('equal')
  plt.title("Integrated VO")
  plt.xlabel("x [m]")
  plt.ylabel("y [m]")
  plt.legend()
  plt.show()


if __name__ == '__main__':
  main()
