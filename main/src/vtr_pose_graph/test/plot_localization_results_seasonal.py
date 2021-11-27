import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
import argparse
import os

def load_data(data_dir, start, end, ignore_runs, path_indicesces, failed_runs):

    info = {}
    path_segments = {'multis':{}, 'new1':{}, 'dark':{}, 'new2':{}}

    for i in range(start, end + 1):


        results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, i)
        
        if (not os.path.isdir(results_dir)) or (i in ignore_runs):
            continue

        success_ind = []
        if i in failed_runs.keys():
            for j in range(len(failed_runs[i])):
                start = failed_runs[i][j][0]
                end = failed_runs[i][j][1]
                success_ind += list(range(start, end + 1))

        path_segments['multis'][i] = {"timestamp":[], "inliers_rgb":[]}
        path_segments['new1'][i] = {"timestamp":[], "inliers_rgb":[]}
        path_segments['dark'][i] = {"timestamp":[], "inliers_rgb":[]}
        path_segments['new2'][i] = {"timestamp":[], "inliers_rgb":[]}

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

        info_file_path = "{}/info.csv".format(results_dir) 

        with open(info_file_path) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True

            for row in csv_reader:

                if not first:

                    priv_id = int(row[2])

                    if (len(success_ind) > 0) and (priv_id not in success_ind):
                        continue

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

        dt = datetime.datetime.fromtimestamp(info[i]["timestamp"][0] / 1e9) 
        print("{}-{}".format(i, dt))

    return info, path_segments

def plot_inliers_segments(avg_inliers_segments, inliers_segments, times, failed, 
                          results_dir, month, success_ind, fail_ind):

    # Create the x-axis date labels
    x_labels = []
    for i in range(len(times)):
        x_labels.append(times[i].strftime('%d.%m-%H:%M'))

    ############### Plot box plot of inliers for each repeat ###################

    ## Plot for off road
    inliers_success_multis = []
    inliers_fail_multis = []
    positions_success_multis = []
    positions_fail_multis = []

    inliers_success_new1 = []
    inliers_fail_new1 = []
    positions_success_new1 = []
    positions_fail_new1 = []    
    
    x_labels_success = []
    x_labels_fail = []
    
    ind = 0
    for i in range(len(times)):
        if (i in success_ind):
            inliers_success_multis.append(inliers_segments['multis'][i])
            inliers_success_new1.append(inliers_segments['new1'][i])
            x_labels_success.append(times[i].strftime('%d-%H:%M'))
            positions_success_multis.append(ind)
            positions_success_new1.append(ind+1)
            ind += 2
    
        if (i in fail_ind):
            inliers_fail_multis.append(inliers_segments['multis'][i])
            inliers_fail_new1.append(inliers_segments['new1'][i])
            x_labels_fail.append(times[i].strftime('%d-%H:%M'))
            positions_fail_multis.append(ind)
            positions_fail_new1.append(ind+1)
            ind += 2

    f = plt.figure(figsize=(30, 12))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    if len(inliers_success_multis) > 0:
    
        p1 = plt.boxplot(inliers_success_multis, 
                        positions=positions_success_multis,
                        sym='', 
                        labels=x_labels_success, 
                        patch_artist=True,
                        boxprops={'facecolor':'teal', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})

    if len(inliers_success_new1) > 0:

        p2 = plt.boxplot(inliers_success_new1, 
                        positions=positions_success_new1,
                        sym='', 
                        labels=[''] * len(positions_success_new1), 
                        patch_artist=True,
                        boxprops={'facecolor':'mediumturquoise', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})

    if len(inliers_fail_multis) > 0:
    
        p3 = plt.boxplot(inliers_fail_multis,
                        positions=positions_fail_multis, 
                        sym='', 
                        labels=x_labels_fail, 
                        patch_artist=True,
                        boxprops={'facecolor':'firebrick', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})

    if len(inliers_fail_new1) > 0:

        p4 = plt.boxplot(inliers_fail_new1,
                    positions=positions_fail_new1, 
                    sym='', 
                    labels=[''] * len(positions_fail_new1), 
                    patch_artist=True,
                    boxprops={'facecolor':'lightcoral', 'linewidth':3},
                    whiskerprops={'color':'black', 'linewidth':3},
                    medianprops={'color':'black', 'linewidth':3})
    
    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')

    plt.xticks(rotation=-80)
    plt.xlabel(r'\textbf{Repeat time (day-hh:mm)}', fontsize=50) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    plt.xticks(fontsize=38) # oct 32 
    plt.yticks(fontsize=48) 

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Area in training data (success/fail)'),
                       matplotlib.lines.Line2D([0], [0], color='mediumturquoise', lw=4, 
                                            label='Area outside training data (success/fail)')]                
    plt.legend(handles=legend_elements, fontsize=32, loc='upper right');

    plt.savefig('{}/inliers_box_offroad_seasonal_{}.png'.format(results_dir, month), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_offroad_seasonal_{}.pdf'.format(results_dir, month), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_offroad_seasonal_{}.svg'.format(results_dir, month), 
                bbox_inches='tight', format='svg')
    plt.close()

    ## Plot for on road ##
    inliers_success_dark = []
    inliers_fail_dark = []
    positions_success_dark = []
    positions_fail_dark = []

    inliers_success_new2 = []
    inliers_fail_new2 = []
    positions_success_new2 = []
    positions_fail_new2 = []    
    
    x_labels_success = []
    x_labels_fail = []
    
    ind = 0
    for i in range(len(times)):
        if (i in success_ind):
            inliers_success_dark.append(inliers_segments['dark'][i])
            inliers_success_new2.append(inliers_segments['new2'][i])
            x_labels_success.append(times[i].strftime('%d-%H:%M'))
            positions_success_dark.append(ind)
            positions_success_new2.append(ind+1)
            ind += 2
    
        if (i in fail_ind):
            inliers_fail_dark.append(inliers_segments['dark'][i])
            inliers_fail_new2.append(inliers_segments['new2'][i])
            x_labels_fail.append(times[i].strftime('%d-%H:%M'))
            positions_fail_dark.append(ind)
            positions_fail_new2.append(ind+1)
            ind += 2

    f = plt.figure(figsize=(30, 12))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    if len(inliers_success_dark) > 0:
    
        p1 = plt.boxplot(inliers_success_dark, 
                        positions=positions_success_dark,
                        sym='', 
                        labels=x_labels_success, 
                        patch_artist=True,
                        boxprops={'facecolor':'teal', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})

    if len(inliers_success_new2) > 0:

        p2 = plt.boxplot(inliers_success_new2, 
                        positions=positions_success_new2,
                        sym='', 
                        labels=[''] * len(positions_success_new2), 
                        patch_artist=True,
                        boxprops={'facecolor':'mediumturquoise', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})

    if len(inliers_fail_dark) > 0:

        p3 = plt.boxplot(inliers_fail_dark,
                        positions=positions_fail_dark, 
                        sym='', 
                        labels=x_labels_fail, 
                        patch_artist=True,
                        boxprops={'facecolor':'firebrick', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})

    if len(inliers_fail_new2) > 0:

        p4 = plt.boxplot(inliers_fail_new2,
                        positions=positions_fail_new2, 
                        sym='', 
                        labels=[''] * len(positions_fail_new2), 
                        patch_artist=True,
                        boxprops={'facecolor':'lightcoral', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})
    
    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')

    plt.xticks(rotation=-80)
    plt.xlabel(r'\textbf{Repeat time (day-hh:mm)}', fontsize=50) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    plt.xticks(fontsize=38) #oct 32 
    plt.yticks(fontsize=48) 

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Area in training data (success/fail)'),
                       matplotlib.lines.Line2D([0], [0], color='mediumturquoise', lw=4, 
                                            label='Area outside training data (success/fail)')]                
    plt.legend(handles=legend_elements, fontsize=32, loc='upper right');

    plt.savefig('{}/inliers_box_onroad_seasonal_{}.png'.format(results_dir, month), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_onroad_seasonal_{}.pdf'.format(results_dir, month), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_onroad_seasonal_{}.svg'.format(results_dir, month), 
                bbox_inches='tight', format='svg')
    plt.close()

    ############# Plot bar plot of average inliers for each repeat #############
    
    # ## September, off road ##
    # f = plt.figure(figsize=(30, 12))
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # num_repeats = len(x_labels_sep)
    # p1 = plt.bar(list(range(0, 2 * num_repeats, 2)), 
    #              avg_inliers_segments['multis'][:month_switch_ind],
    #              tick_label=x_labels_sep)
    # p2 = plt.bar(list(range(1, 2 * num_repeats, 2)), 
    #              avg_inliers_segments['new1'][:month_switch_ind])
    
    # for i in range(len(p1)):
    #     p1[i].set_color('teal')
    #     p2[i].set_color('mediumturquoise')

    # plt.xticks(rotation=-80)
    # plt.xlabel(r'\textbf{Repeat time (day-hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=30) 
    # plt.yticks(fontsize=48) 
    # plt.ylim([min(avg_inliers_segments['new1'][:month_switch_ind]) - 10, 
    #           max(avg_inliers_segments['multis'][:month_switch_ind]) + 10])
    # plt.title(r'\textbf{Mean matched feature inliers for each repeat - off road - September}', 
    #           fontsize=50)

    # legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
    #                                         label='Area in training data'),
    #                    matplotlib.lines.Line2D([0], [0], color='mediumturquoise', lw=4, 
    #                                         label='Area outside training data')]                
    # plt.legend(handles=legend_elements, fontsize=36, loc='upper right');

    # plt.savefig('{}/inliers_mean_offroad_september.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()

    # ## October, off road ##
    # f = plt.figure(figsize=(30, 12))
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # num_repeats = len(x_labels_oct)
    # p1 = plt.bar(list(range(0, 2 * num_repeats, 2)), 
    #              avg_inliers_segments['multis'][month_switch_ind:],
    #              tick_label=x_labels_oct)
    # p2 = plt.bar(list(range(1, 2 * num_repeats, 2)), 
    #              avg_inliers_segments['new1'][month_switch_ind:])
    
    # for i in range(len(p1)):
    #     if failed[month_switch_ind:][i]:
    #         p1[i].set_color('firebrick')
    #         p2[i].set_color('lightcoral')
    #     else:
    #         p1[i].set_color('teal')
    #         p2[i].set_color('mediumturquoise')

    # plt.xticks(rotation=-80)
    # plt.xlabel(r'\textbf{Repeat time (day-hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=30) 
    # plt.yticks(fontsize=48) 
    # plt.ylim([min(avg_inliers_segments['new1'][month_switch_ind:]) - 10, 
    #           max(avg_inliers_segments['multis'][month_switch_ind:]) + 10])
    # plt.title(r'\textbf{Mean matched feature inliers for each repeat - off road - October}', 
    #           fontsize=50)

    # legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
    #                                         label='Area in training data (success/fail)'),
    #                    matplotlib.lines.Line2D([0], [0], color='mediumturquoise', lw=4, 
    #                                         label='Area outside training data (success/fail)')]                
    # plt.legend(handles=legend_elements, fontsize=36, loc='upper right');

    # plt.savefig('{}/inliers_mean_offroad_october.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()

    # ## September, on road ##
    # f = plt.figure(figsize=(30, 12))
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # num_repeats = len(x_labels_sep)
    # p1 = plt.bar(list(range(0, 2 * num_repeats, 2)), 
    #              avg_inliers_segments['dark'][:month_switch_ind],
    #              tick_label=x_labels_sep)
    # p2 = plt.bar(list(range(1, 2 * num_repeats, 2)), 
    #              avg_inliers_segments['new2'][:month_switch_ind])
    
    # for i in range(len(p1)):
    #     p1[i].set_color('teal')
    #     p2[i].set_color('mediumturquoise')

    # plt.xticks(rotation=-80)
    # plt.xlabel(r'\textbf{Repeat time (day-hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=30) 
    # plt.yticks(fontsize=48) 
    # plt.ylim([min(avg_inliers_segments['dark'][:month_switch_ind]) - 10, 
    #           max(avg_inliers_segments['new2'][:month_switch_ind]) + 10])
    # plt.title(r'\textbf{Mean matched feature inliers for each repeat - on road - September}', 
    #           fontsize=50)

    # legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
    #                                         label='Area in training data'),
    #                    matplotlib.lines.Line2D([0], [0], color='mediumturquoise', lw=4, 
    #                                         label='Area outside training data')]                
    # plt.legend(handles=legend_elements, fontsize=36, loc='upper right');

    # plt.savefig('{}/inliers_mean_onroad_september.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()

    # ## October, off road ##
    # f = plt.figure(figsize=(30, 12))
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # num_repeats = len(x_labels_oct)
    # p1 = plt.bar(list(range(0, 2 * num_repeats, 2)), 
    #              avg_inliers_segments['dark'][month_switch_ind:],
    #              tick_label=x_labels_oct)
    # p2 = plt.bar(list(range(1, 2 * num_repeats, 2)), 
    #              avg_inliers_segments['new2'][month_switch_ind:])
    
    # for i in range(len(p1)):
    #     if failed[month_switch_ind:][i]:
    #         p1[i].set_color('firebrick')
    #         p2[i].set_color('lightcoral')
    #     else:
    #         p1[i].set_color('teal')
    #         p2[i].set_color('mediumturquoise')

    # plt.xticks(rotation=-80)
    # plt.xlabel(r'\textbf{Repeat time (day-hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=30) 
    # plt.yticks(fontsize=48) 
    # plt.ylim([min(avg_inliers_segments['dark'][month_switch_ind:]) - 10, 
    #           max(avg_inliers_segments['new2'][month_switch_ind:]) + 10])
    # plt.title(r'\textbf{Mean matched feature inliers for each repeat - on road - October}', 
    #           fontsize=50)

    # legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
    #                                         label='Area in training data (success/fail)'),
    #                    matplotlib.lines.Line2D([0], [0], color='mediumturquoise', lw=4, 
    #                                         label='Area outside training data (success/fail)')]                
    # plt.legend(handles=legend_elements, fontsize=36, loc='upper right');

    # plt.savefig('{}/inliers_mean_onroad_october.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()


def plot_inliers(avg_inliers, times, inliers, colours, failed, fail_ind, 
                 success_ind, results_dir, month, selected_runs):

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

    inliers_success = []
    inliers_fail = []
    positions_success = []
    positions_fail = []  
    
    x_labels_success = []
    x_labels_fail = []
   
    ind = 0
    for i in range(len(times)):
        if (i in success_ind):
            inliers_success.append(inliers[i])
            x_labels_success.append(times[i].strftime('%d-%H:%M'))
            positions_success.append(ind)
            ind += 1
    
        if (i in fail_ind):
            inliers_fail.append(inliers[i])
            x_labels_fail.append(times[i].strftime('%d-%H:%M'))
            positions_fail.append(ind)
            ind += 1

    if len(inliers_success) > 0:
    
        p1 = plt.boxplot(inliers_success, 
                        positions=positions_success,
                        sym='', 
                        labels=x_labels_success, 
                        patch_artist=True,
                        boxprops={'facecolor':'teal', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})

    if len(inliers_fail) > 0:

        p2 = plt.boxplot(inliers_fail, 
                        positions=positions_fail,
                        sym='', 
                        labels=x_labels_fail, 
                        patch_artist=True,
                        boxprops={'facecolor':'firebrick', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})
    
    plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')
            
    plt.xticks(rotation=-75)
    plt.xlabel(r'\textbf{Repeat time (day-hh:mm)}', fontsize=50) 
    plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    plt.xticks(fontsize=38) #oct 32 
    plt.yticks(fontsize=48) 

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
                                            label='Successful run'),
                       matplotlib.lines.Line2D([0], [0], color='firebrick', lw=4, 
                                            label='Run with failure')]                
    plt.legend(handles=legend_elements, fontsize=32, loc='upper right');

    plt.savefig('{}/inliers_box_seasonal_{}.png'.format(results_dir, month), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_box_seasonal_{}.pdf'.format(results_dir, month), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_box_seasonal_{}.svg'.format(results_dir, month), 
                bbox_inches='tight', format='svg')
    plt.close()

    ## November ##
    # f = plt.figure(figsize=(30, 13))
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])

    # inliers_success = []
    # inliers_fail = []
    # positions_success = []
    # positions_fail = []  
    
    # x_labels_success = []
    # x_labels_fail = []
    
    # ind = 0
    # for i in range(len(times)):
    #     if (i in success_ind) and (i >= month_switch_ind_nov):
    #         inliers_success.append(inliers[i])
    #         x_labels_success.append(times[i].strftime('%d-%H:%M'))
    #         positions_success.append(ind)
    #         ind += 1
    
    #     if (i in fail_ind) and (i >= month_switch_ind_nov):
    #         inliers_fail.append(inliers[i])
    #         x_labels_fail.append(times[i].strftime('%d-%H:%M'))
    #         positions_fail.append(ind)
    #         ind += 1
    
    # p1 = plt.boxplot(inliers_success, 
    #                 positions=positions_success,
    #                 sym='', 
    #                 widths=0.5,
    #                 labels=x_labels_success, 
    #                 patch_artist=True,
    #                 boxprops={'facecolor':'teal', 'linewidth':3},
    #                 whiskerprops={'color':'black', 'linewidth':3},
    #                 medianprops={'color':'black', 'linewidth':3})

    # p2 = plt.boxplot(inliers_fail, 
    #                 positions=positions_fail,
    #                 widths=0.5,
    #                 sym='', 
    #                 labels=x_labels_fail, 
    #                 patch_artist=True,
    #                 boxprops={'facecolor':'firebrick', 'linewidth':3},
    #                 whiskerprops={'color':'black', 'linewidth':3},
    #                 medianprops={'color':'black', 'linewidth':3})
    
    # plt.axhline(y=6.0, color='red', linewidth='2', linestyle='--')
            
    # plt.xticks(rotation=-75)
    # plt.xlabel(r'\textbf{Repeat time (day-hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=30) 
    # plt.yticks(fontsize=48) 

    # legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
    #                                         label='Successful run'),
    #                    matplotlib.lines.Line2D([0], [0], color='firebrick', lw=4, 
    #                                         label='Run with failure')]                
    # plt.legend(handles=legend_elements, fontsize=32, loc='upper right');

    # plt.savefig('{}/inliers_box_november_seasonal.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.savefig('{}/inliers_box_november_seasonal.pdf'.format(results_dir), 
    #             bbox_inches='tight', format='pdf')
    # plt.savefig('{}/inliers_box_november_seasonal.svg'.format(results_dir), 
    #             bbox_inches='tight', format='svg')
    # plt.close()

    ############# Plot bar plot of average inliers for each repeat #############
    
    ## September##
    # f = plt.figure(figsize=(30, 12))
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])

    # x_labels = []
    # for i in range(len(times)):
    #     x_labels.append(times[i].strftime('%d-%H:%M'))
    
    # p = plt.bar(x_labels[:month_switch_ind], avg_inliers[:month_switch_ind])

    # for i in range(len(p)):
    #     p[i].set_color('teal')

    # plt.xticks(rotation=-75)
    # plt.xlabel(r'\textbf{Repeat time (day.month-hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=38) 
    # plt.yticks(fontsize=48) 
    # plt.ylim([min(avg_inliers) - 10, max(avg_inliers) + 10])
    # plt.title(r'\textbf{Mean matched feature inliers - September}', 
    #           fontsize=50)

    # legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
    #                                         label='Successful run')] 
    # plt.legend(handles=legend_elements, fontsize=32, loc='upper right'); 

    # plt.savefig('{}/inliers_mean_september_seasonal.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()

    # failed_oct = []
    # for i in range(len(times)):
    #     if (i >= month_switch_ind):
    #         failed_oct.append(failed[i])

    # f = plt.figure(figsize=(30, 12))
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # p = plt.bar(x_labels[month_switch_ind:], avg_inliers[month_switch_ind:])

    # for i in range(len(p)):
    #     if failed_oct[i]:
    #         p[i].set_color('firebrick')
    #     else:
    #         p[i].set_color('teal')

    # plt.xticks(rotation=-75)
    # plt.xlabel(r'\textbf{Repeat time (day.month-hh:mm)}', fontsize=50) 
    # plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
    # plt.xticks(fontsize=38) 
    # plt.yticks(fontsize=48) 
    # plt.ylim([min(avg_inliers) - 10, max(avg_inliers) + 10])
    # plt.title(r'\textbf{Mean matched feature inliers - October', 
    #           fontsize=50)

    # legend_elements = [matplotlib.lines.Line2D([0], [0], color='teal', lw=4, 
    #                                         label='Successful run'),
    #                    matplotlib.lines.Line2D([0], [0], color='firebrick', lw=4, 
    #                                         label='Run with failure')] 
    # plt.legend(handles=legend_elements, fontsize=32, loc='upper right'); 

    # plt.savefig('{}/inliers_mean_october_seasonal.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()

def plot_cdf(times_all, inliers_all, results_dir, month):

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

    teach_time = datetime.datetime.combine(datetime.date(2021, 8, 14), 
                          datetime.time(13, 26))

    time_diffs = []

    # Difference in t.o.d
    for i in range(len(times_all)):
        dt = times_all[i]
        dt = dt.replace(day=14)
        dt = dt.replace(month=8)
        time_diffs.append((dt - teach_time).total_seconds() / (60.0 * 60.0))

    # Difference in days
    # for i in range(len(times_all)):
    #     time_diffs.append((times_all[i] - teach_time).total_seconds() / (60.0 * 60.0 * 24.0))

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
    plt.xticks(fontsize=38)
    plt.yticks(fontsize=38)
    plt.grid(True, which='both', axis='both', color='gray', linestyle='-', 
             linewidth=1)
    plt.xlabel(r'\textbf{Number of inliers}', fontsize=50)
    plt.ylabel(r'\textbf{CDF, keyframes}', fontsize=50)
    plt.savefig('{}/inliers_cdf_tod_seasonal_{}.png'.format(results_dir, month), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_cdf_tod_seasonal_{}.pdf'.format(results_dir, month), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_cdf_tod_seasonal_{}.svg'.format(results_dir, month), 
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

    plt.savefig('{}/inliers_cdf_tod_colorbar_seasonal_{}.png'.format(results_dir, month), 
                bbox_inches='tight', format='png')
    plt.savefig('{}/inliers_cdf_tod_colorbar_seasonal_{}.pdf'.format(results_dir, month), 
                bbox_inches='tight', format='pdf')
    plt.savefig('{}/inliers_cdf_tod_colorbar_seasonal_{}.svg'.format(results_dir, month), 
                bbox_inches='tight', format='svg')
    plt.close()


def plot_data(info, path_segments, data_dir, failed_runs, month, ignore_runs, selected_runs):

    avg_inliers = []
    avg_inliers_segments = {'multis':[], 'new1':[], 'dark':[], 'new2':[]}
    inliers = []
    inliers_segments = {'multis':[], 'new1':[], 'dark':[], 'new2':[]} 
    times = [] 
    avg_comp_time = []
    colours = {'failed':[]}
    failed = []
    failed_ind = []
    success_ind = []
    selected_ind = []

    ind = 0
    for i in info.keys():

        inliers.append(np.asarray(info[i]["inliers_rgb"]))
        avg_inliers.append(sum(info[i]["inliers_rgb"]) / float(len(info[i]["inliers_rgb"])))
        avg_comp_time.append(sum(info[i]["comp_time"]) / float(len(info[i]["comp_time"])))

        dt = datetime.datetime.fromtimestamp(info[i]["timestamp"][0] / 1e9)	

        for key in path_segments.keys():
            inliers_segments[key] += [np.asarray(path_segments[key][i]['inliers_rgb'])] 
            if len(path_segments[key][i]['inliers_rgb']) > 0:
                avg_inliers_segments[key] += \
                                [sum(path_segments[key][i]["inliers_rgb"]) / 
                                float(len(path_segments[key][i]["inliers_rgb"]))]
            else:
                avg_inliers_segments[key] += [0.0]
           
        times.append(dt)

        if i in failed_runs.keys():
            colours['failed'] = colours['failed'] + ['firebrick']
            failed.append(True)
            failed_ind.append(ind)
        else:
            colours['failed'] = colours['failed'] + ['teal']
            failed.append(False)
            success_ind.append(ind)

        if i in selected_runs:
            selected_ind.append(ind)

        ind += 1 

    results_dir = "{}/graph.index/repeats".format(data_dir)

    # plot_inliers(avg_inliers, times, inliers, colours, failed, failed_ind, 
    #              success_ind, results_dir, month, selected_ind)

    # plot_inliers_segments(avg_inliers_segments, inliers_segments, times, failed, 
    #                      results_dir, month, success_ind, failed_ind)

    # plot_cdf(times, inliers, results_dir, month)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')
    parser.add_argument('--start', default=None, type=int,
                        help='first repeats (default: None)')
    parser.add_argument('--end', default=None, type=int,
                        help='last repeats (default: None)')
    parser.add_argument('--month', default=None, type=str,
                        help='month (default: None)')

    args = parser.parse_args()

    ignore_runs = [101]

    selected_runs = [30, 43, 50, 54, 57, 60, 69]

    failed_runs = {51:[[1, 3973]], 
                   52:[[1, 2268]], 
                   55:[[1, 5236], [5395, 6955], [7018, 7155]], 
                   56:[[2, 6330], [6408, 7806]], 
                   60:[[3, 6946], [6958, 7806]], 
                   64:[[3, 3979]], 
                   68:[[1, 5189], [5414, 7807]],
                   72:[[1, 9000]],
                   76:[[1, 9000]],
                   78:[[1, 9000]],
                   84:[[1, 9000]],
                   85:[[1, 9000]],
                   95:[[1, 9000]],
                   102:[[1, 9000]],
                   104:[[1, 9000]]}

    path_indices = [468, 5504, 6083, 6553, 7108]
   
    info, path_segments = load_data(args.path, args.start, args.end, ignore_runs, path_indices, failed_runs)

    plot_data(info, path_segments, args.path, failed_runs, args.month, ignore_runs, selected_runs);

    #30/46/86 (sept/oct/nov)


