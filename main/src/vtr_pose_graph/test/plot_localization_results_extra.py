import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
import argparse

def load_data(data_dir, num_repeats):

    info = {}
   
    for i in range(1, num_repeats + 1):

        info[i] = {"timestamp":[],
                    "live_id":[],
                    "priv_id":[],
                    "success":[],
                    "inliers_rgb":[],
                    "inliers_gray":[],
                    "inliers_cc":[],
                    "window_temporal_depth":[],
                    "window_num_vertices":[],
                    "comp_time":[]}

        results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, i)
        info_file_path = "{}/info.csv".format(results_dir) 

        with open(info_file_path) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True

            for row in csv_reader:

                if not first:
                    info[i]["timestamp"] += [int(row[0])]
                    info[i]["live_id"] += [row[1]]
                    info[i]["priv_id"] += [row[2]]
                    info[i]["success"] += [row[3]]
                    info[i]["inliers_rgb"] += [float(row[4])]
                    info[i]["inliers_gray"] += [float(row[5])]
                    info[i]["inliers_cc"] += [float(row[6])]
                    info[i]["window_temporal_depth"] += [row[7]]
                    info[i]["window_num_vertices"] += [row[8]]
                    info[i]["comp_time"] += [float(row[9])]

                first = False

        dt_start = datetime.datetime.fromtimestamp(info[i]["timestamp"][0] / 1e9) 
        dt_end = datetime.datetime.fromtimestamp(info[i]["timestamp"][-1] / 1e9) 
        dt_diff = (dt_end - dt_start).total_seconds() / 60.0
        print("{}-{}-{}-{}".format(i, dt_start.strftime('%H:%M'), dt_end.strftime('%H:%M'), round(dt_diff)))

    return info

def plot_box_linear(times, inliers, day1, ignore_labels, results_dir):

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

    ############### Plot box plot of inliers for each repeat ###################

    f = plt.figure(figsize=(30, 13))
    # f = plt.figure(figsize=(15, 6))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    times_sorted = times[:]
    inliers_map = {}
    day_map = {}
    for i in range(len(times)):
        inliers_map[times[i]] = inliers[i]
        day_map[times[i]] = day1[i]

    times_sorted.sort()

    inliers_day1 = []
    positions_day1 = [] 
    x_labels_day1 = []
    inliers_day2 = []
    positions_day2 = [] 
    x_labels_day2 = []
    
    x = date2num(times)
    first = min(times)

    ind = 0
    for i in range(len(times_sorted)):
        if day_map[times_sorted[i]]:
            inliers_day1.append(inliers_map[times_sorted[i]])
            x_labels_day1.append(times_sorted[i].strftime('%H:%M'))
            positions_day1.append(i)
        else:
            inliers_day2.append(inliers_map[times_sorted[i]])
            x_labels_day2.append(times_sorted[i].strftime('%H:%M'))
            positions_day2.append(i)
        
        ind += 1

  
    p1 = plt.boxplot(inliers_day1, 
                    positions=positions_day1,
                    sym='', 
                    labels=x_labels_day1, 
                    patch_artist=True,
                    boxprops={'facecolor':'teal', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p2 = plt.boxplot(inliers_day2, 
                    positions=positions_day2,
                    sym='', 
                    labels=x_labels_day2, 
                    patch_artist=True,
                    boxprops={'facecolor':'cornflowerblue', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')
           
    plt.xticks(rotation=-75)
    # plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=36) 
    # plt.yticks(fontsize=48) 
    plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=32) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=32)
    plt.xticks(fontsize=32) 
    plt.yticks(fontsize=32) 

    plt.xlim([-0.5, len(times) - 0.5])
    # plt.xlim([-15.0, max(positions_day1) + 15.0])

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Day 1: 11.11'),
                       matplotlib.lines.Line2D([0], [0], color='cornflowerblue', lw=4, 
                                            label='Day 2: 12.11')]                
    # plt.legend(handles=legend_elements, fontsize=36, loc='upper right');
    plt.legend(handles=legend_elements, fontsize=32, loc='upper right');

    plt.savefig('{}/inliers_box_extra_linear.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_extra_linear.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_extra_linear.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()


def plot_cdf(times_all, inliers_all, results_dir):

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

    # ########### Plot cumulative distribution of inliers for each run ###########
    f = plt.figure(figsize=(20, 12)) #
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_lines = []
    max_inliers = 0

    # Field
    # teach_time = datetime.datetime.combine(datetime.date(2021, 11, 11), 
    #                       datetime.time(9, 52))

    # Parking
    teach_time = datetime.datetime.combine(datetime.date(2021, 11, 11), 
                           datetime.time(11, 34))

    time_diffs = []
    for i in range(len(times_all)):
        time_diffs.append((times_all[i] - teach_time).total_seconds() / 3600.0)

    time_diffs_norm = []
    max_time_diff = max(time_diffs)
    min_time_diff = min(time_diffs)
    for i in range(len(times_all)):
        time_diffs_norm.append((time_diffs[i] - min_time_diff) / (max_time_diff - min_time_diff))

    cmap = matplotlib.cm.viridis

    for i in range(len(inliers_all)):
        
        max_val = np.max(inliers_all[i])
        if max_val > max_inliers:
            max_inliers = max_val
        n_bins_vis_range = 50
        n_bins_total = int((n_bins_vis_range * max_val) / 696)

        values, base = np.histogram(inliers_all[i], bins=n_bins_total)
        unity_values = values / values.sum()
        cumulative = np.cumsum(np.flip(unity_values))
        p = plt.plot(base[:-1], 1.0 - cumulative, linewidth=5, color=cmap(time_diffs_norm[i]))
        plot_lines.append(p[0])

    plt.axvline(x=6.0, color='red', linewidth='3', linestyle='--')

    # plt.legend(plot_lines, labels, prop={'size': 36})
    plt.xlim([max_inliers, 0])
    plt.ylim([0, 1])
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.grid(True, which='both', axis='both', color='gray', linestyle='-', 
             linewidth=1)
    plt.xlabel(r'\textbf{Number of inliers}', fontsize=32)
    plt.ylabel(r'\textbf{CDF, keyframes}', fontsize=32)
    plt.savefig('{}/inliers_cdf_extra.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_cdf_extra.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_cdf_extra.svg'.format(results_dir), 
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

    plt.savefig('{}/inliers_cdf_colorbar_extra.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_cdf_colorbar_extra.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_cdf_colorbar_extra.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()


def plot_data(info, ignore_runs, data_dir, ignore_labels):

    avg_inliers = []
    inliers = []
    inliers_all = []
    times = [] 
    times_all = []
    day1 = []
    day1_all = []
    ignore_labels_box = []

    for i in info.keys():

        dt = datetime.datetime.fromtimestamp(info[i]["timestamp"][0] / 1e9) 

        print("{} - {}".format(i, dt.strftime('%H:%M')))

        if dt.day == 12:
            if i not in ignore_runs:
                day1.append(False)
            day1_all.append(False)
        else:
            if i not in ignore_runs:
                day1.append(True)
            day1_all.append(True)

        if dt.day != 11:
            dt = dt.replace(day=11)

        if i not in ignore_runs:
            inliers.append(info[i]["inliers_rgb"])
            avg_inliers.append(sum(info[i]["inliers_rgb"]) / float(len(info[i]["inliers_rgb"])))
            times.append(dt)

            if i in ignore_labels:
                ignore_labels_box.append(True)
            else:
                ignore_labels_box.append(False)

        inliers_all.append(info[i]["inliers_rgb"])
        times_all.append(dt)            

    results_dir = "{}/graph.index/repeats".format(data_dir)

    plot_box_linear(times, inliers, day1, ignore_labels, results_dir)

    plot_cdf(times_all, inliers_all, results_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')
    parser.add_argument('--numrepeats', default=None, type=int,
                        help='number of repeats (default: None)')

    args = parser.parse_args()

    # ignore_runs = [4,6,31,34,35,36,39,40,10,16,21,19] #exp1, box width=16 ot 13?
    # ignore_runs = [4, 34, 35, 36, 39] # box width = 10
    ignore_runs = [] # box width = 6

    # ignore_labels = [4, 10, 30, 32, 34, 35, 36, 39]
    ignore_labels = [] 
    
    info = load_data(args.path, args.numrepeats)

    plot_data(info, ignore_runs, args.path, ignore_labels);