import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import datetime
import argparse
import os

def load_data(data_dir, num_repeats, ignore_runs, failed_runs):

    keyframe_info = {}
    
    for i in range(1, num_repeats + 1):

        results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, i)
        
        if (not os.path.isdir(results_dir)) or (i in ignore_runs):
            continue

        success_ind = []
        if i in failed_runs.keys():
            for j in range(len(failed_runs[i])):
                start = failed_runs[i][j][0]
                end = failed_runs[i][j][1]
                success_ind += list(range(start, end + 1))

        info_file_path = "{}/info.csv".format(results_dir) 

        with open(info_file_path) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True

            for row in csv_reader:

                if not first:

                    priv_id = int(row[2])

                    if (len(success_ind) > 0) and (priv_id not in success_ind):
                        continue

                    inliers = float(row[4])

                    if (priv_id > 0):

                        if priv_id in keyframe_info.keys():
                            keyframe_info[priv_id]['inliers_total'] += inliers
                            keyframe_info[priv_id]['inliers'] += [inliers]
                            keyframe_info[priv_id]['num_loc'] += 1
                        else:
                            keyframe_info[priv_id] = {'inliers_total':inliers,
                                                      'inliers': [inliers],  
                                                      'num_loc':1}

                first = False


    vo_file = "{}/graph.index/repeats/vo_teach/vo_poses_teach.csv".format(data_dir)
         
    with open(vo_file) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        first = True

        for row in csv_reader:

            if not first:

                priv_id = int(row[0])
                x = float(row[1])
                y = float(row[2])
                
                if (priv_id in keyframe_info.keys()):
                    keyframe_info[priv_id]['x'] = x
                    keyframe_info[priv_id]['y'] = y

            first = False

    return keyframe_info

def plot_inliers(keyframe_info, data_dir):

    results_dir = "{}/graph.index/repeats".format(data_dir)

    stats = {}

    for keyframe in keyframe_info.keys():
        keyframe_info[keyframe]['mean'] = keyframe_info[keyframe]['inliers_total']\
                                     / float(keyframe_info[keyframe]['num_loc'])

        results = np.quantile(np.asarray(keyframe_info[keyframe]['inliers']), q=[0.25, 0.5, 0.75], interpolation='nearest')                             
        stats[keyframe] = results

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
    
    keyframe_ind = list(keyframe_info.keys())
    keyframe_ind.sort()

    # Make a list of point pairs (p1, p2) that are stored in numpy arrays as
    # [[x1, y1], [x2, y2]]. Each list entry is then the start and end point of
    # the line segment. Also set the number of inliers for each segment.
    lines = []
    mean = []
    median = []
    q25 = []
    q75 = []
    x = []
    y = []

    min_val = 1000000
    max_val = 0
    for i in range(len(keyframe_ind)-1):
        k1 = keyframe_ind[i]
        k2 = keyframe_ind[i + 1]
        p1 = np.array((keyframe_info[k1]['x'], keyframe_info[k2]['x']))
        p2 = np.array((keyframe_info[k1]['y'], keyframe_info[k2]['y']))
        line = np.column_stack((p1,p2))
        lines.append(line)

        x.append(keyframe_info[k1]['x'])
        y.append(keyframe_info[k1]['y'])

        median.append(stats[k1][1])
        q25.append(stats[k1][0])
        q75.append(stats[k1][2])
        mean.append(keyframe_info[k1]['mean'])

        if stats[k1][0] < min_val:
            min_val = stats[k1][0]

        if stats[k1][2] > max_val:
            max_val = stats[k1][2]

    values = [q25, median, q75]
    names = ['Q1', 'Median', 'Q3']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(45, 12))

    count = 0
    for ax in axes.flat:

        # ax.set_xlim([-10, 125])   # exp2
        # ax.set_ylim([-88, 92])  # exp2
        # ax.set_xlim([-2, 106])
        # ax.set_ylim([-70, 20]) # exp1
        ax.axis('equal')
        # ax.axis('off')

        sc = ax.scatter(x, y, c=values[count], vmin=min_val, vmax=max_val, s=400, cmap=plt.cm.viridis)
        
        # line_segments = LineCollection(lines,
        #                                linewidths=(40.0),
        #                                linestyles='solid')

        # # Set up the colour bar
        # line_segments.set_array(np.asarray(values[count]))
        # ax.add_collection(line_segments)

        # line_segments.set_clim(0, 461) #exp2
        # line_segments.set_clim(0 , max(num_inliers))

        if count == 1:
            ax.set_xlabel(r'\textbf{x (m)}', fontsize=40)
        elif count == 0:
            ax.set_ylabel(r'\textbf{y (m)}', fontsize=40)
        # ax.tick_params(axis='x', labelsize=50)
        # ax.tick_params(axis='y', labelsize=50)
        # plt.sci(line_segments)

        if count == 1:
            ax.set_title(r'\textbf{Median}', fontsize=50)
        elif count == 0:
            ax.set_title(r'\textbf{Q1}', fontsize=50)
        else:
            ax.set_title(r'\textbf{Q3}', fontsize=50)
        
        count += 1

    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.83, 0.11, 0.02, 0.77])
    # fig.colorbar(im, cax=cbar_ax)

    axcb = fig.colorbar(sc, cax=cbar_ax)
    axcb.set_label(r'\textbf{Number of inliers}', fontsize=50)
    axcb.ax.tick_params(labelsize=48)

    plt.savefig('{}/inliers_path.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.close()

        
    # fig, ax = plt.subplots(figsize=(23, 20))
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # # ax.set_xlim([-10, 125])   # exp2
    # # ax.set_ylim([-88, 92])  # exp2
    # ax.set_xlim([-2, 106])
    # ax.set_ylim([-70, 20]) # exp1
    # ax.axis('equal')

    # line_segments = LineCollection(lines,
    #                                linewidths=(20.0),
    #                                linestyles='solid') # exp2
    # line_segments = LineCollection(lines,
    #                                linewidths=(40.0),
    #                                linestyles='solid')

    # # Set up the colour bar
    # line_segments.set_array(np.asarray(num_inliers))
    # ax.add_collection(line_segments)

    # line_segments.set_clim(0, 461) #exp2
    # line_segments.set_clim(0 , max(num_inliers))

    # axcb = fig.colorbar(line_segments)
    # axcb.set_label(r'\textbf{Mean number of inliers}', fontsize=60)
    # # cbarlabels = np.linspace(0, 470, num=10)
    # # cbarlabels = [0,100,200,300,400,500]
    # # axcb.set_ticks(cbarlabels)
    # # axcb.set_ticklabels(cbarlabels)
    # axcb.ax.tick_params(labelsize=48)   

    # ax.set_title(r'\textbf{Mean number of inliers for map keyframes along the path}', fontsize=50))
    # ax.set_xlabel(r'\textbf{x (m)}', fontsize=60)
    # ax.set_ylabel(r'\textbf{y (m)}', fontsize=60)
    # ax.tick_params(axis='x', labelsize=50)
    # ax.tick_params(axis='y', labelsize=50)
    # plt.sci(line_segments)

    # plt.savefig('{}/inliers_on_path.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')

    # plt.savefig('{}/inliers_on_path.pdf'.format(results_dir), 
    #             bbox_inches='tight', format='pdf')
    # plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')
    parser.add_argument('--numrepeats', default=None, type=int,
                        help='number of repeats (default: None)')

    args = parser.parse_args()

    # ignore_runs = [6,7,10,14,15,16,17,18,24] # exp2
    # ignore_runs = [4,6,31,34,35,36,39,40,10,16,21,19] #exp1
    ignore_runs = []

    # failed_runs = {51:[[1, 3973]], 
    #                52:[[1, 2268]], 
    #                55:[[1, 5236], [5395, 6955], [7018, 7155]], 
    #                56:[[2, 6330], [6408, 7806]], 
    #                60:[[3, 6946], [6958, 7806]], 
    #                64:[[3, 3979]], 
    #                68:[[1, 5189], [5414, 7807]]}
    
    failed_runs = {} 

    keyframe_info = load_data(args.path, args.numrepeats, ignore_runs, failed_runs)

    plot_inliers(keyframe_info, args.path);


