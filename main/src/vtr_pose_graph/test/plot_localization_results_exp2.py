import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
import argparse
import tikzplotlib

def load_data(data_dir_learned, data_dir_surf,  num_repeats, ignore_runs_learned, ignore_runs_surf, path_indices):

    info_learned = {}
    info_surf = {}
    path_segments = {'multis':{}, 'new1':{}, 'dark':{}, 'new2':{}}
   
    for i in range(1, num_repeats + 1):

        if i in ignore_runs_learned:
            continue

        info_learned[i] = {"timestamp":[],
                           "live_id":[],
                           "priv_id":[],
                           "success":[],
                           "inliers_rgb":[],
                           "inliers_gray":[],
                           "inliers_cc":[],
                           "window_temporal_depth":[],
                           "window_num_vertices":[],
                           "comp_time":[]}

        path_segments['multis'][i] = {"timestamp":[], "inliers_rgb":[]}
        path_segments['new1'][i] = {"timestamp":[], "inliers_rgb":[]}
        path_segments['dark'][i] = {"timestamp":[], "inliers_rgb":[]}
        path_segments['new2'][i] = {"timestamp":[], "inliers_rgb":[]}

        results_dir_learned = "{}/graph.index/repeats/{}/results".format(data_dir_learned, i)
        info_file_path_learned = "{}/info.csv".format(results_dir_learned) 
        
        with open(info_file_path_learned) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True

            for row in csv_reader:

                if not first:

                    priv_id = int(row[2])

                    info_learned[i]["timestamp"] += [int(row[0])]
                    info_learned[i]["live_id"] += [row[1]]
                    info_learned[i]["priv_id"] += [row[2]]
                    info_learned[i]["success"] += [row[3]]
                    info_learned[i]["inliers_rgb"] += [float(row[4])]
                    info_learned[i]["inliers_gray"] += [float(row[5])]
                    info_learned[i]["inliers_cc"] += [float(row[6])]
                    info_learned[i]["window_temporal_depth"] += [row[7]]
                    info_learned[i]["window_num_vertices"] += [row[8]]
                    info_learned[i]["comp_time"] += [float(row[9])]

                    if (priv_id < path_indices[0]) or \
                       ((priv_id >= path_indices[1]) and (priv_id < path_indices[2])):
                        path_segments['multis'][i]['timestamp'] += [int(row[0])]
                        path_segments['multis'][i]['inliers_rgb'] += [float(row[4])]
                    elif (priv_id >= path_indices[0]) and (priv_id < path_indices[1]):
                        path_segments['new1'][i]['timestamp'] += [int(row[0])]
                        path_segments['new1'][i]['inliers_rgb'] += [float(row[4])]
                    elif ((priv_id >= path_indices[2]) and(priv_id < path_indices[3])) or \
                         (priv_id >= path_indices[4]):
                        path_segments['dark'][i]['timestamp'] += [int(row[0])]
                        path_segments['dark'][i]['inliers_rgb'] += [float(row[4])]
                    elif (priv_id >= path_indices[3]) and (priv_id < path_indices[4]):
                        path_segments['new2'][i]['timestamp'] += [int(row[0])]
                        path_segments['new2'][i]['inliers_rgb'] += [float(row[4])]

                first = False

        info_surf[i] = {"timestamp":[],
                        "live_id":[],
                        "priv_id":[],
                        "success":[],
                        "inliers_rgb":[],
                        "inliers_gray":[],
                        "inliers_cc":[],
                        "window_temporal_depth":[],
                        "window_num_vertices":[],
                        "comp_time":[]}

        if i not in ignore_runs_surf:

            results_dir_surf = "{}/graph.index/repeats/{}/results".format(data_dir_surf, i)
            info_file_path_surf = "{}/info.csv".format(results_dir_surf) 

            with open(info_file_path_surf) as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=',')
                first = True

                for row in csv_reader:

                    if not first:
                        info_surf[i]["timestamp"] += [int(row[0])]
                        info_surf[i]["live_id"] += [row[1]]
                        info_surf[i]["priv_id"] += [row[2]]
                        info_surf[i]["success"] += [row[3]]
                        info_surf[i]["inliers_rgb"] += [float(row[4])]
                        info_surf[i]["inliers_gray"] += [float(row[5])]
                        info_surf[i]["inliers_cc"] += [float(row[6])]
                        info_surf[i]["window_temporal_depth"] += [row[7]]
                        info_surf[i]["window_num_vertices"] += [row[8]]
                        info_surf[i]["comp_time"] += [float(row[9])]

                    first = False
        else:
            info_surf[i]["timestamp"] = info_learned[i]["timestamp"]

        dt_start = datetime.datetime.fromtimestamp(info_learned[i]["timestamp"][0] / 1e9) 
        dt_end = datetime.datetime.fromtimestamp(info_learned[i]["timestamp"][-1] / 1e9) 
        dt_diff = (dt_end - dt_start).total_seconds() / 60.0
        print("{}-{}-{}-{}".format(i, dt_start.strftime('%H:%M'), dt_end.strftime('%H:%M'), round(dt_diff)))

    return info_learned, info_surf, path_segments


def plot_quantile(times_all, inliers_all, day_all, results_dir):

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

    times_sorted = times_all[:]
    inliers_map = {}
    day_map = {}
    for i in range(len(times_sorted)):
        inliers_map[times_sorted[i]] = inliers_all[i]
        day_map[times_sorted[i]] = day_all[i]

    times_sorted.sort()

    quart_25_day1 = []
    quart_75_day1 = []
    median_day1 = []
    time_day1 = []

    quart_25_day2 = []
    quart_75_day2 = []
    median_day2 = []
    time_day2 = []

    quart_25_day3 = []
    quart_75_day3 = []
    median_day3 = []
    time_day3 = []

    for i in range(len(times_sorted)):
        stats = np.quantile(np.asarray(inliers_map[times_sorted[i]]), q=[0.25, 0.5, 0.75], interpolation='nearest')
        if day_map[times_sorted[i]] == 1:
            quart_25_day1.append(stats[0])
            median_day1.append(stats[1])
            quart_75_day1.append(stats[2])
            time_day1.append(times_sorted[i])
        elif day_map[times_sorted[i]] == 2:
            quart_25_day2.append(stats[0])
            median_day2.append(stats[1])
            quart_75_day2.append(stats[2])  
            time_day2.append(times_sorted[i])
        else:
            quart_25_day3.append(stats[0])
            median_day3.append(stats[1])
            quart_75_day3.append(stats[2])  
            time_day3.append(times_sorted[i])

    f = plt.figure(figsize=(30, 12))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.plot(time_day1, median_day1, color='teal', marker='o', linewidth=3, markersize=10, label='Day 1: 15.08') 
    plt.fill_between(time_day1, quart_25_day1, quart_75_day1, alpha = .2,color = 'teal')

    plt.plot(time_day2, median_day2, color='cornflowerblue', marker='o', linewidth=3, markersize=10, label='Day 2: 16.08') 
    plt.fill_between(time_day2, quart_25_day2, quart_75_day2, alpha = .2,color = 'cornflowerblue')

    plt.plot(time_day3, median_day3, color='slateblue', marker='o', linewidth=3, markersize=10, label='Day 3: 20.08') 
    plt.fill_between(time_day3, quart_25_day3, quart_75_day3, alpha = .2,color = 'slateblue')

    plt.legend(fontsize=32)   
    
    myFmt = matplotlib.dates.DateFormatter('%H:%M')
    ax = plt.axes()
    ax.xaxis.set_major_formatter(myFmt)

    # plt.xlim([min(times) - datetime.timedelta(minutes=10), max(times) + datetime.timedelta(minutes=10)])
    plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=32) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=32)
    plt.xticks(fontsize=32) 
    plt.yticks(fontsize=32) 
    # plt.ylim([min(avg_inliers) - 10, max(avg_inliers) + 10])

    # legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, label='Day1: 03.08'),
    #                    matplotlib.lines.Line2D([0], [0], color='cornflowerblue', lw=4, label='Day2: 09.08')]                   
    # plt.legend(handles=legend_elements, fontsize=40);

    plt.savefig('{}/inliers_quantline_generalization.png'.format(results_dir), bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_quantile_generalization.pdf'.format(results_dir), bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_quantile_generalization.svg'.format(results_dir), bbox_inches='tight', format='svg')
    plt.close()


def plot_box(times, inliers, day, ignore_labels, results_dir):

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
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    inliers_day1 = []
    positions_day1 = [] 
    x_labels_day1 = []
    inliers_day2 = []
    positions_day2 = [] 
    x_labels_day2 = []
    inliers_day3 = []
    positions_day3 = [] 
    x_labels_day3 = []
    
    x = date2num(times)
    first = min(times)

    ind = 0
    for i in range(len(times)):
        if day[i] == 1:
            inliers_day1.append(inliers[i])
            if ignore_labels[i]:
                x_labels_day1.append('')
            else:
                x_labels_day1.append(times[i].strftime('%H:%M'))
            positions_day1.append((times[i] - first).seconds / 60.0)
        elif day[i] == 2:
            inliers_day2.append(inliers[i])
            if ignore_labels[i]:
                x_labels_day2.append('')
            else:
                x_labels_day2.append(times[i].strftime('%H:%M'))
            positions_day2.append((times[i] - first).seconds / 60.0)
        else:
            inliers_day3.append(inliers[i])
            if ignore_labels[i]:
                x_labels_day3.append('')
            else:
                x_labels_day3.append(times[i].strftime('%H:%M'))
            positions_day3.append((times[i] - first).seconds / 60.0)
        ind += 1

  
    p1 = plt.boxplot(inliers_day1, 
                    positions=positions_day1,
                    sym='', 
                    labels=x_labels_day1, 
                    patch_artist=True,
                    widths=6.0,
                    boxprops={'facecolor':'teal', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p2 = plt.boxplot(inliers_day2, 
                    positions=positions_day2,
                    sym='', 
                    labels=x_labels_day2, 
                    patch_artist=True,
                    widths=6.0,
                    boxprops={'facecolor':'cornflowerblue', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p2 = plt.boxplot(inliers_day3, 
                    positions=positions_day3,
                    sym='', 
                    labels=x_labels_day3, 
                    patch_artist=True,
                    widths=6.0,
                    boxprops={'facecolor':'slateblue', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})
  
    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')
           
    plt.xticks(rotation=-75)
    plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=50) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    plt.xticks(fontsize=32) 
    plt.yticks(fontsize=48) 
    plt.xlim([-15.0, max(positions_day1) + 15.0])

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Day 1: 15.08'),
                       matplotlib.lines.Line2D([0], [0], color='cornflowerblue', lw=4, 
                                            label='Day 2: 16.08'),
                       matplotlib.lines.Line2D([0], [0], color='slateblue', lw=4, 
                                            label='Day 3: 20.08')]                
    plt.legend(handles=legend_elements, fontsize=40, loc='upper right');

    plt.savefig('{}/inliers_box_generalization_all.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_generalization_all.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_generalization_all.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()

def plot_box_surf(times, inliers_learned, inliers_surf, day, ignore_labels, results_dir):

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
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    inliers_day1 = []
    positions_day1 = [] 
    x_labels_day1 = []
    inliers_day2 = []
    positions_day2 = [] 
    x_labels_day2 = []
    inliers_day3 = []
    positions_day3 = [] 
    x_labels_day3 = []
    inliers_surf_box = []
    positions_surf = [] 
    x_labels_surf = []
    
    x = date2num(times)
    first = min(times)

    ind = 0
    for i in range(len(times)):
        if day[i] == 1:
            inliers_day1.append(inliers_learned[i])
            if ignore_labels[i]:
                x_labels_day1.append('')
            else:
                x_labels_day1.append(times[i].strftime('%H:%M'))
            positions_day1.append((times[i] - first).seconds / 60.0)
        elif day[i] == 2:
            inliers_day2.append(inliers_learned[i])
            if ignore_labels[i]:
                x_labels_day2.append('')
            else:
                x_labels_day2.append(times[i].strftime('%H:%M'))
            positions_day2.append((times[i] - first).seconds / 60.0)
        else:
            inliers_day3.append(inliers_learned[i])
            if ignore_labels[i]:
                x_labels_day3.append('')
            else:
                x_labels_day3.append(times[i].strftime('%H:%M'))
            positions_day3.append((times[i] - first).seconds / 60.0)

        if len(inliers_surf[i]) > 0:
            inliers_surf_box.append(inliers_surf[i])
            x_labels_surf.append('')
            positions_surf.append(((times[i] - first).seconds / 60.0))

        ind += 1

  
    p1 = plt.boxplot(inliers_day1, 
                    positions=positions_day1,
                    sym='', 
                    labels=x_labels_day1, 
                    patch_artist=True,
                    widths=17.0,
                    boxprops={'facecolor':'teal', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p2 = plt.boxplot(inliers_day2, 
                    positions=positions_day2,
                    sym='', 
                    labels=x_labels_day2, 
                    patch_artist=True,
                    widths=17.0,
                    boxprops={'facecolor':'cornflowerblue', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p3 = plt.boxplot(inliers_day3, 
                    positions=positions_day3,
                    sym='', 
                    labels=x_labels_day3, 
                    patch_artist=True,
                    widths=17.0,
                    boxprops={'facecolor':'slateblue', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p4 = plt.boxplot(inliers_surf_box, 
                    positions=positions_surf,
                    sym='', 
                    labels=x_labels_surf, 
                    patch_artist=True,
                    widths=17.0,
                    boxprops={'facecolor':'gold', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})
  
    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')
           
    plt.xticks(rotation=-75)
    plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=50) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    plt.xticks(fontsize=32) 
    plt.yticks(fontsize=48) 
    plt.xlim([-15.0, max(positions_day1) + 15.0])

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Day 1: 15.08'),
                       matplotlib.lines.Line2D([0], [0], color='cornflowerblue', lw=4, 
                                            label='Day 2: 16.08'),
                       matplotlib.lines.Line2D([0], [0], color='slateblue', lw=4, 
                                            label='Day 3: 20.08'),
                       matplotlib.lines.Line2D([0], [0], color='gold', lw=4, 
                                            label='SURF')]                
    plt.legend(handles=legend_elements, fontsize=40, loc='upper left');

    plt.savefig('{}/inliers_box_generalization_surf_select.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_generalization_surf_select.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_generalization_surf_select.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()

def plot_box_surf_linear(times, inliers_learned, inliers_surf, day, ignore_labels, results_dir):

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

    f = plt.figure(figsize=(30, 13)) # org
    # f = plt.figure(figsize=(30, 6)) # narrow
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    times_sorted = times[:]
    inliers_map = {}
    inliers_map_surf = {}
    position_map = {}
    day_map = {}
    for i in range(len(times_sorted)):
        inliers_map[times[i]] = inliers_learned[i]
        inliers_map_surf[times[i]] = inliers_surf[i]
        # position_map[times_sorted[i]] = i
        day_map[times[i]] = day[i]

    times_sorted.sort()

    inliers_day1 = []
    positions_day1 = [] 
    x_labels_day1 = []
    inliers_day2 = []
    positions_day2 = [] 
    x_labels_day2 = []
    inliers_day3 = []
    positions_day3 = [] 
    x_labels_day3 = []
    inliers_surf_box = []
    positions_surf = [] 
    x_labels_surf = []
    
    x = date2num(times)
    first = min(times)

    ind = 0
    for i in range(len(times_sorted)):
        if day_map[times_sorted[i]] == 1:
            inliers_day1.append(inliers_map[times_sorted[i]])
            x_labels_day1.append(times_sorted[i].strftime('%H:%M'))
            positions_day1.append(i)
        elif day_map[times_sorted[i]] == 2:
            inliers_day2.append(inliers_map[times_sorted[i]])
            x_labels_day2.append(times_sorted[i].strftime('%H:%M'))
            positions_day2.append(i)
        else:
            inliers_day3.append(inliers_map[times_sorted[i]])
            x_labels_day3.append(times_sorted[i].strftime('%H:%M'))
            positions_day3.append(i)

        if len(inliers_map_surf[times_sorted[i]]) > 0:
            inliers_surf_box.append(inliers_map_surf[times_sorted[i]])
            x_labels_surf.append('')
            positions_surf.append(i)

        ind += 1

  
    p1 = plt.boxplot(inliers_day1, 
                    positions=positions_day1,
                    sym='', 
                    labels=x_labels_day1, 
                    patch_artist=True,
                    widths=0.6,
                    boxprops={'facecolor':'teal', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p2 = plt.boxplot(inliers_day2, 
                    positions=positions_day2,
                    sym='', 
                    labels=x_labels_day2, 
                    patch_artist=True,
                    widths=0.6,
                    boxprops={'facecolor':'cornflowerblue', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p3 = plt.boxplot(inliers_day3, 
                    positions=positions_day3,
                    sym='', 
                    labels=x_labels_day3, 
                    patch_artist=True,
                    widths=0.6,
                    boxprops={'facecolor':'slateblue', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p4 = plt.boxplot(inliers_surf_box, 
                    positions=positions_surf,
                    sym='', 
                    labels=x_labels_surf, 
                    patch_artist=True,
                    widths=0.6,
                    boxprops={'facecolor':'gold', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})
  
    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')
         
    plt.xticks(rotation=-75)

    # org
    # plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=36) 
    # plt.yticks(fontsize=48) 

    # narrow
    # plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=30)  # narrow
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=30)
    plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=32) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=32)
    # plt.xticks(fontsize=28) #narrow
    # plt.yticks(fontsize=28) 
    plt.xticks(fontsize=32) 
    plt.yticks(fontsize=32) 

    plt.xlim([-0.5, len(times) - 0.5])
    # plt.xlim([-15.0, max(positions_day1) + 15.0])

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Day 1: 15.08'),
                       matplotlib.lines.Line2D([0], [0], color='cornflowerblue', lw=4, 
                                            label='Day 2: 16.08'),
                       matplotlib.lines.Line2D([0], [0], color='slateblue', lw=4, 
                                            label='Day 3: 20.08'),
                       matplotlib.lines.Line2D([0], [0], color='gold', lw=4, 
                                            label='SURF')]                
    # plt.legend(handles=legend_elements, fontsize=36, loc='upper right'); #org
    plt.legend(handles=legend_elements, fontsize=32, loc='upper right'); 
    # plt.legend(handles=legend_elements, fontsize=28, loc='upper right'); # narrow

    plt.savefig('{}/inliers_box_generalization_surf_linear.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_generalization_surf_linear.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_generalization_surf_linear.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()

def plot_box_segments(inliers_segments, times, results_dir):

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

    # Create the x-axis date labels
    times_sorted = times[:]
    inliers_map_multis = {}
    inliers_map_new1 = {}
    inliers_map_dark = {}
    inliers_map_new2 = {}

    for i in range(len(times)):
        inliers_map_multis[times[i]] = inliers_segments['multis'][i]
        inliers_map_new1[times[i]] = inliers_segments['new1'][i]
        inliers_map_dark[times[i]] = inliers_segments['dark'][i]
        inliers_map_new2[times[i]] = inliers_segments['new2'][i]
        
    times_sorted.sort()

    x_labels = []
    for i in range(len(times_sorted)):
        x_labels.append(times_sorted[i].strftime('%H:%M'))

    inliers_multis = []
    inliers_dark = []
    inliers_new1 = []
    inliers_new2 = []
    
    for i in range(len(times_sorted)):
        inliers_multis.append(inliers_map_multis[times_sorted[i]])
        inliers_dark.append(inliers_map_dark[times_sorted[i]])
        inliers_new1.append(inliers_map_new1[times_sorted[i]])
        inliers_new2.append(inliers_map_new2[times_sorted[i]])

    ############### Plot box plot of inliers for each repeat ###################
    
    ## Plot for off-road ##
    f = plt.figure(figsize=(30, 13))
    # f = plt.figure(figsize=(30, 6))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    num_repeats = len(x_labels)

    print(len(inliers_map_multis[times_sorted[i]]))
    print(len(x_labels))

    p1 = plt.boxplot(inliers_multis, 
                    positions=list(range(0, 2 * num_repeats, 2)),
                    sym='', 
                    labels=x_labels, 
                    patch_artist=True,
                    boxprops={'facecolor':'teal', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p2 = plt.boxplot(inliers_new1,
                    positions=list(range(1, 2 * num_repeats, 2)), 
                    sym='', 
                    labels=[''] * num_repeats, 
                    patch_artist=True,
                    boxprops={'facecolor':'mediumturquoise', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})
    
    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')

    plt.xticks(rotation=-80)
    # plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=38) 
    # plt.yticks(fontsize=48) 

    plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=32) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=32)
    plt.xticks(fontsize=32) 
    plt.yticks(fontsize=23) 

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Area in training data'),
                       matplotlib.lines.Line2D([0], [0], color='mediumturquoise', lw=4, 
                                            label='Area outside training data')]                
    # plt.legend(handles=legend_elements, fontsize=36, loc='upper right');
    plt.legend(handles=legend_elements, fontsize=32, loc='upper right');

    plt.savefig('{}/inliers_box_offroad_generalization_linear.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_offroad_generalization_linear.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_offroad_generalization_linear.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()

    
    ## Plot for on-road ##
    f = plt.figure(figsize=(30, 13))
    # f = plt.figure(figsize=(30, 6))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    p1 = plt.boxplot(inliers_dark, 
                    positions=list(range(0, 2 * num_repeats, 2)),
                    sym='', 
                    labels=x_labels, 
                    patch_artist=True,
                    boxprops={'facecolor':'teal', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})

    p2 = plt.boxplot(inliers_new2,
                    positions=list(range(1, 2 * num_repeats, 2)), 
                    sym='', 
                    labels=[''] * num_repeats, 
                    patch_artist=True,
                    boxprops={'facecolor':'mediumturquoise', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})
    
    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')

    plt.xticks(rotation=-80)
    # plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=38) 
    # plt.yticks(fontsize=48) 

    plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=32) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=32)
    plt.xticks(fontsize=32) 
    plt.yticks(fontsize=32) 

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Area in training data'),
                       matplotlib.lines.Line2D([0], [0], color='mediumturquoise', lw=4, 
                                            label='Area outside training data')]                
    # plt.legend(handles=legend_elements, fontsize=36, loc='upper right');
    plt.legend(handles=legend_elements, fontsize=32, loc='upper right');

    plt.savefig('{}/inliers_box_onroad_generalization_linear.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_onroad_generalization_linear.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_onroad_generalization_linear.svg'.format(results_dir), 
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

    teach_time = datetime.datetime.combine(datetime.date(2021, 8, 15), 
                          datetime.time(13, 26))

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
    plt.savefig('{}/inliers_cdf_generalization.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_cdf_generalization.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_cdf_generalization.svg'.format(results_dir), 
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

    plt.savefig('{}/inliers_cdf_colorbar_generalization.png'.format(results_dir), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_cdf_colorbar_generalization.pdf'.format(results_dir), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_cdf_colorbar_generalization.svg'.format(results_dir), 
                bbox_inches='tight', format='svg')
    plt.close()


def plot_bar(avg_inliers_learned, avg_inliers_surf, times_learned, times_surf, day, results_dir):

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

    ############# Plot bar plot of average inliers for each repeat #############

    x = date2num(times_learned)
    
    f = plt.figure(figsize=(30, 12))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    min_surf = 1000
    
    for k in range(len(avg_inliers_learned)):
        if avg_inliers_surf[k] == 0:
            p1 = plt.bar(x[k], avg_inliers_learned[k], width=0.008) # 0.015 #0.001
        else:
            p1 = plt.bar(x[k] - 0.004, avg_inliers_learned[k], width=0.008) # 0.015 #0.001
    
        if day[k] == 1:
            p1[0].set_color('teal')
        elif day[k] == 2:
            p1[0].set_color('cornflowerblue')
        else:
            p1[0].set_color('navy')

        if avg_inliers_surf[k] != 0:
            p2 = plt.bar(x[k] + 0.004, avg_inliers_surf[k], width=0.008) # 0.015 #0.001

            if avg_inliers_surf[k] < min_surf:
                min_surf = avg_inliers_surf[k]
    
        p2[0].set_color('gray')

    myFmt = matplotlib.dates.DateFormatter('%H:%M')
    ax = plt.axes()
    ax.xaxis.set_major_formatter(myFmt)   

    plt.xlim([min(times_learned) - datetime.timedelta(minutes=10), max(times_learned) + datetime.timedelta(minutes=10)])
    plt.xlabel(r'\textbf{Repeat time (hh:mm)}', fontsize=50) 
    plt.ylabel(r'\textbf{Mean number of inliers}', fontsize=50)
    plt.xticks(fontsize=48) 
    plt.yticks(fontsize=48) 
    plt.ylim([max(min_surf - 10, 0), max(avg_inliers_learned) + 10])
                
    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, label='Day1: 15.08'),
                       matplotlib.lines.Line2D([0], [0], color='cornflowerblue', lw=4, label='Day2: 16.08'),
                       matplotlib.lines.Line2D([0], [0], color='navy', lw=4, label='Day3: 20.08'),
                       matplotlib.lines.Line2D([0], [0], color='gray', lw=4, label='SURF')]                   
    plt.legend(handles=legend_elements, fontsize=40);

    plt.savefig('{}/inliers_bar_generalization.png'.format(results_dir), dpi=1000, bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_bar_generalization.pdf'.format(results_dir), dpi=1000, bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_bar_generalization.svg'.format(results_dir), dpi=1000, bbox_inches='tight', format='svg')
    plt.close()

def plot_data(info_learned, info_surf, info_segments, ignore_labels, data_dir_learned, data_dir_surf):

    avg_inliers_learned = []
    inliers_learned = []
    avg_inliers_surf = []
    inliers_surf = []
    times_learned = []
    times_surf = [] 
    day = []
    ignore_labels_box = []
    inliers_segments = {'multis':[], 'new1':[], 'dark':[], 'new2':[]} 

    for i in info_surf.keys():

        if info_surf[i]["inliers_gray"] == []:
            avg_inliers_surf.append(0)
            inliers_surf.append([])
        else:
            avg_inliers_surf.append((sum(info_surf[i]["inliers_gray"]) + sum(info_surf[i]["inliers_cc"])) / float(len(info_surf[i]["inliers_gray"])))
            inliers_surf.append(info_surf[i]["inliers_gray"] + info_surf[i]["inliers_cc"])

        dt = datetime.datetime.fromtimestamp(info_surf[i]["timestamp"][0] / 1e9)

        # Cheat to get all the plot bars for one day
        if dt.day != 15:
            dt = dt.replace(day=15)
       
        times_surf.append(dt)

    for i in info_learned.keys():

        avg_inliers_learned.append(sum(info_learned[i]["inliers_rgb"]) / float(len(info_learned[i]["inliers_rgb"])))
        inliers_learned.append(info_learned[i]["inliers_rgb"])

        for key in info_segments.keys():
            inliers_segments[key] += [np.asarray(info_segments[key][i]['inliers_rgb'])] 
            
        dt = datetime.datetime.fromtimestamp(info_learned[i]["timestamp"][0] / 1e9)	

        if dt.day == 16:
            day.append(2)
        elif dt.day == 15:
            day.append(1)
        else:
            day.append(3)

        # Cheat to get all the plot bars for one day
        if dt.day != 15:
            dt = dt.replace(day=15)
       
        times_learned.append(dt)

        if i in ignore_labels:
            ignore_labels_box.append(True)
        else:
            ignore_labels_box.append(False)

    results_dir = "{}/graph.index/repeats".format(data_dir_learned)

    plot_cdf(times_learned, inliers_learned, results_dir)

    # plot_bar(avg_inliers_learned, avg_inliers_surf, times_learned, times_surf, day, results_dir)

    # plot_box(times_learned, inliers_learned, day, ignore_labels_box, results_dir)

    # plot_box_surf(times_learned, inliers_learned, inliers_surf, day, ignore_labels_box, results_dir)

    plot_box_surf_linear(times_learned, inliers_learned, inliers_surf, day, ignore_labels_box, results_dir)

    plot_box_segments(inliers_segments, times_learned, results_dir)

    plot_quantile(times_learned, inliers_learned, day, results_dir)

    # plot_inliers(avg_inliers_learned, avg_inliers_surf, times_learned, times_surf, colours_learned, colours_surf, results_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')
    parser.add_argument('--path_surf', default=None, type=str,
                        help='path to results dir (default: None)')
    parser.add_argument('--numrepeats', default=None, type=int,
                        help='number of repeats (default: None)')

    args = parser.parse_args()

    # ignore_runs = [5,6,9,10,14, 15, 16, 17, 18, 23] # exp2 wrong
    # ignore_runs_learned = [6,7,10,14,15,16,17,18,24] # exp2 w=17
    ignore_runs_learned = [10,15,16] # exp2 w =6 ?
    ignore_runs_surf = [6,7,8,10,11,12,13,15,16,17,20,21,22,29] # exp2 surf
    # ignore_labels = [6, 7, 17, 18, 24]
    ignore_labels = []

    path_indices = [468, 5504, 6083, 6553, 7108]

   
    info_learned, info_surf, info_segments = load_data(args.path, 
                                                       args.path_surf, 
                                                       args.numrepeats, 
                                                       ignore_runs_learned, 
                                                       ignore_runs_surf, 
                                                       path_indices)

    # plot_data(info_learned, info_surf, info_segments, ignore_labels, args.path, args.path_surf);
