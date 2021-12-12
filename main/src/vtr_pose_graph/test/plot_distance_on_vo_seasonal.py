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

    teach_time = datetime.datetime.combine(datetime.date(2021, 8, 14), 
                          datetime.time(13, 26))

    time_diffs = []

    # Difference in t.o.d
    # for i in range(len(times)):
    #     dt = times[i]
    #     dt = dt.replace(day=14)
    #     dt = dt.replace(month=8)
    #     time_diffs.append((dt - teach_time).total_seconds() / (60.0 * 60.0))

    # Difference in days
    for i in range(len(times)):
        time_diffs.append((times[i] - teach_time).total_seconds() / (60.0 * 60.0 * 24.0))

    time_diffs_norm = []
    max_time_diff = max(time_diffs)
    min_time_diff = min(time_diffs)
    for i in range(len(times)):
        time_diffs_norm.append((time_diffs[i] - min_time_diff) / (max_time_diff - min_time_diff))

    cmap = matplotlib.cm.viridis
    
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

            p = plt.plot(base[:-1], cumulative, linewidth=5, color=cmap(time_diffs_norm[i]))
            plot_lines.append(p[0])       

            if min_c < min_y:
                min_y = min(cumulative)

    plt.xlim([0, max_val])
    plt.ylim([min_y, 1])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.grid(True, which='both', axis='both', color='gray', linestyle='-', 
             linewidth=1)
    plt.xlabel(r'\textbf{Distance on VO (metres)}', fontsize=32)
    plt.ylabel(r'\textbf{CDF, keyframes}', fontsize=32)
    plt.savefig('{}/cdf_distance_vo_days.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/cdf_distance_vo_days.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/cdf_distance_vo_days.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()


    fig, ax = plt.subplots(figsize=(1, 12))
    # fig.subplots_adjust(bottom=0.5)
    norm = matplotlib.colors.Normalize(vmin=min(time_diffs), vmax=max(time_diffs))

    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Hours')

    # ticklabels = cbar.ax.get_ymajorticklabels()
    # ticks = list(cbar.get_ticks())

    # # Append the ticks (and their labels) for minimum and the maximum value
    # cbar.set_ticks([min_time_diff, max_time_diff] + ticks)
    # # cbar.set_ticklabels([min_time_diff, max_time_diff] + ticklabels)

    plt.savefig('{}/cdf_vo_colorbar_seasonal_days.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/cdf_vo_colorbar_seasonal_days.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/cdf_vo_colorbar_seasonal_days.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()


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