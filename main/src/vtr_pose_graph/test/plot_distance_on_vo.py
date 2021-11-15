import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
import argparse
import os

def load_data(data_dir, num_repeats, ignore_runs, path_indicesces, failed_runs):

    info = {}

    for i in range(1, num_repeats + 1):

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

        info[i] = {'total': [], 
                   'multis': [], 
                   'new1': [],
                   'dark': [], 
                   'new2': []}

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

                    total_loc += 1
                    
                    assert(cumulative >= 0.0)

                    info[i]['total'] += [cumulative]

                    if (priv_id < path_indices[0]) or \
                       ((priv_id >= path_indices[1]) and (priv_id < path_indices[2])):
                        info[i]['multis'] += [cumulative]
                    elif (priv_id >= path_indices[0]) and (priv_id < path_indices[1]):
                        info[i]['new1'] += [cumulative]
                    elif ((priv_id >= path_indices[2]) and(priv_id < path_indices[3])) or \
                         (priv_id >= path_indices[4]):
                        info[i]['dark'] += [cumulative]
                    elif (priv_id >= path_indices[3]) and (priv_id < path_indices[4]):
                        info[i]['new2'] += [cumulative]

                first = False

        print("{}-{}-{}".format(i, total_vo, total_loc)) 

    return info


def plot_dist(dist_all, dist, times, failed, failed_ind, success_ind, results_dir):

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

    
    label_names = {'total': 'Whole path',
                   'multis': 'Area in training data - off road',
                   'dark': 'Area in training data - on road',
                   'new1': 'Area outside training data - off road',
                   'new2': 'Area outside training data - on road'}

    f = plt.figure(figsize=(20, 12)) #
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_lines = []
    labels = []
    max_val = 0
    min_y = 0

    for key in dist_all.keys():
   
        if max(dist_all[key]) > max_val:
            max_val = max(dist_all[key])
        n_bins_vis_range = 50
        n_bins_total = 50
        # n_bins_total = int((n_bins_vis_range * max_val) / 696)

        values, base = np.histogram(dist_all[key], bins=n_bins_total)
        unity_values = values / values.sum()
        cumulative = np.cumsum(unity_values)
        p = plt.plot(base[:-1], cumulative, linewidth=5)
        plot_lines.append(p[0])
        labels.append(label_names[key])

        if key == 'new1':
            min_y = min(cumulative)

    plt.legend(plot_lines, labels, prop={'size': 36})
    plt.xlim([0, max_val])
    plt.ylim([min_y, 1])
    plt.xticks(fontsize=38)
    plt.yticks(fontsize=38)
    plt.grid(True, which='both', axis='both', color='gray', linestyle='-', 
             linewidth=1)
    plt.xlabel(r'\textbf{Total distance on VO (metres)}', fontsize=50)
    plt.ylabel(r'\textbf{Cumulative distributon, keyframes}', fontsize=50)
    # plt.title(r'\textbf{Cumulative distribution of repeats with distance driven on VO}', 
    #            fontsize=50)
    plt.savefig('{}/cdf_distance_vo_6_seasonal.png'.format(results_dir), 
                bbox_inches='tight', format='png')
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

    dist_all = {'total':[], 'multis':[], 'new1':[], 'dark':[], 'new2':[]}
    dist = {'total':[], 'multis':[], 'new1':[], 'dark':[], 'new2':[]}
    times = [] 
    failed = []
    failed_ind = []
    success_ind = []

    ind = 0
    for i in info.keys():

        # if i == month_switch:
        #     month_switch_ind = ind
        
        dt = datetime.datetime.fromtimestamp(info[i]["timestamp"] / 1e9)	

        for key in info[i].keys():
            if key != 'timestamp':
                dist_all[key] += info[i][key] 
                dist[key] += [info[i][key]] 
           
        times.append(dt)

        if i in failed_runs.keys():
            failed.append(True)
            failed_ind.append(ind)
        else:
            failed.append(False)
            success_ind.append(ind)

        ind += 1 

    results_dir = "{}/graph.index/repeats".format(data_dir)

    plot_dist(dist_all, dist, times, failed, failed_ind, success_ind, results_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')
    parser.add_argument('--numrepeats', default=None, type=int,
                        help='number of repeats (default: None)')

    args = parser.parse_args()

    ignore_runs = []

    # failed_runs = [55, 56, 60, 68]

    failed_runs = {51:[[1, 3973]], 
                   52:[[1, 2268]], 
                   55:[[1, 5236], [5395, 6955], [7018, 7155]], 
                   56:[[2, 6330], [6408, 7806]], 
                   60:[[3, 6946], [6958, 7806]], 
                   64:[[3, 3979]], 
                   68:[[1, 5189], [5414, 7807]]}

    # ignore_runs = [51, 52, 64]

    ignore_runs = []

    path_indices = [468, 5504, 6083, 6553, 7108]

    october = 46
    
    info = load_data(args.path, args.numrepeats, ignore_runs, 
                     path_indices, failed_runs)

    plot_data(info, args.path, failed_runs);


