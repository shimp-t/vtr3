import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
import argparse
import os

def load_data(data_dir, num_repeats, ignore_runs, failed_runs):

    info = {}

    for i in range(1, num_repeats + 1):

        print(i)

        success_ind = []
        if i in failed_runs.keys():
            for j in range(len(failed_runs[i])):
                start = failed_runs[i][j][0]
                end = failed_runs[i][j][1]
                success_ind += list(range(start, end + 1))

        total_vo = 0
        total_loc = 0

        results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, i)
        
        if (not os.path.isdir(results_dir)) or (i in ignore_runs):
            continue

        info[i] = {'total': []}

        info_file_path = "{}/dist.csv".format(results_dir) 

        with open(info_file_path) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True

            dist_prev = 0.0
            dist_curr = 0.0
            cumulative = 0.0
            restart = False

            for row in csv_reader:

                if not first:

                    if 'timestamp' not in info[i].keys():
                        info[i]['timestamp'] = int(row[0])
                    
                    priv_id = int(row[2])

                    if len(success_ind) > 0:
                        if priv_id not in success_ind:
                            restart = True
                            continue

                    num_inliers = int(row[4])
                    assert(int(row[5]) == 0)
                    assert(int(row[6]) == 0)
                    success = True if num_inliers >= 6 else False

                    dist_prev = dist_curr
                    dist_curr = float(row[7])

                    if restart:
                        dist_prev = dist_curr
                        restart = False

                    # Started a new run
                    if dist_curr == 0.0:
                        dist_prev = 0.0

                    if success:
                        cumulative = 0.0
                    else:
                        dist_on_vo = dist_curr - dist_prev
                        cumulative += dist_on_vo
                        assert(dist_on_vo >= 0.0)
                        total_vo += 1
                        print(dist_on_vo)
                        print(cumulative)
                        print('============')

                    total_loc += 1
                    
                    assert(cumulative >= 0.0)

                    info[i]['total'] += [cumulative]

                first = False

        print("{}-{}-{}".format(i, total_vo, total_loc)) 

    return info


def plot_dist(dist, times, results_dir):

    plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    params = {'text.usetex' : True,
              'font.size' : 40,                   # Set font size to 11pt
              'axes.labelsize': 40,               # -> axis labels
              'legend.fontsize': 40,              # -> legends
              'xtick.labelsize' : 40,
              'ytick.labelsize' : 40,
              'font.family' : 'lmodern',
              'text.latex.unicode': True,
              }
    plt.rcParams.update(params) 

    
    f = plt.figure(figsize=(20, 12)) #
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_lines = []
    labels = []
    max_val = 0.0
    min_y = 1.0

    for i in range(len(dist)):

  
        if max(dist[i]) > max_val:
            max_val = max(dist[i])
        n_bins_vis_range = 50
        n_bins_total = 50
        # n_bins_total = int((n_bins_vis_range * max_val) / 696)

        values, base = np.histogram(dist[i], bins=n_bins_total)
        unity_values = values / values.sum()
        cumulative = np.cumsum(unity_values)

        min_c = min(cumulative)

        if (min_c > 0.0):

            p = plt.plot(base[:-1], cumulative, linewidth=5)
            plot_lines.append(p[0])
            # labels.append(times[i].strftime("%H:%M"))
            labels.append(times[i].strftime('%d.%m-%H:%M'))           

            if min_c < min_y:
                min_y = min(cumulative)

    plt.legend(plot_lines, labels, loc='lower right', prop={'size': 36})
    plt.xlim([0, max_val])
    plt.ylim([min_y, 1])
    plt.xticks(fontsize=38)
    plt.yticks(fontsize=38)
    plt.grid(True, which='both', axis='both', color='gray', linestyle='-', 
             linewidth=1)
    plt.xlabel(r'\textbf{Distance on VO (metres)}', fontsize=50)
    plt.ylabel(r'\textbf{CDF, keyframes}', fontsize=50)
    plt.savefig('{}/cdf_distance_vo.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/cdf_distance_vo.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/cdf_distance_vo.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()

    # Create the x-axis date labels
    # x_labels = []
    # for i in range(len(times)):
    #     x_labels.append(times[i].strftime('%d.%m-%H:%M'))

    # f = plt.figure(figsize=(20, 12)) #
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plot_lines = []
    # labels = []

    # # for key in dist.keys():

    # for i in range(0, len(dist['total'],), 2):
    #     print(min(dist[key][i]))
       
    #     # max_val = np.max(dist)
    #     n_bins_vis_range = 50
    #     n_bins_total = 50
    #     # n_bins_total = int((n_bins_vis_range * max_val) / 696)

    #     values, base = np.histogram(dist[key][i], bins=n_bins_total)
    #     unity_values = values / values.sum()
    #     cumulative = np.cumsum(unity_values)
    #     p = plt.plot(base[:-1], cumulative, linewidth=3)
    #     plot_lines.append(p[0])
    #     labels.append(x_labels[i])

    # plt.legend(plot_lines, labels, prop={'size': 36})
    # plt.xlim([0, 1])
    # # plt.ylim([0.7, 1])
    # plt.xticks(fontsize=38)
    # plt.yticks(fontsize=38)
    # plt.grid(True, which='both', axis='both', color='gray', linestyle='-', 
    #          linewidth=1)
    # plt.xlabel(r'\textbf{Total distance on VO (metres)}', fontsize=50)
    # plt.ylabel(r'\textbf{Cumulative distributon, keyframes}', fontsize=50)
    # # plt.title(r'\textbf{Cumulative distribution of repeats with distance driven on VO}', 
    # #            fontsize=50)
    # plt.savefig('{}/cdf_distance_vo_20_repeats_seasonal.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()

def plot_data(info, data_dir, failed_runs):

    dist = []
    times = [] 
    
    ind = 0
    for i in info.keys():
       
        dist.append(info[i]['total'])
        dt = datetime.datetime.fromtimestamp(info[i]["timestamp"] / 1e9)           
        times.append(dt)
 

    results_dir = "{}/graph.index/repeats".format(data_dir)

    plot_dist(dist, times, results_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')
    parser.add_argument('--numrepeats', default=None, type=int,
                        help='number of repeats (default: None)')

    args = parser.parse_args()

    failed_runs = {}

    # ignore_runs = [10, 15, 16] #exp2
    ignore_runs = [101]
   
    info = load_data(args.path, args.numrepeats, ignore_runs, failed_runs)

    plot_data(info, args.path, failed_runs);


