import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
import argparse
import os

def load_data(data_dir, num_repeats, ignore_runs):

    info = {}
    times = {}

    for i in range(1, num_repeats + 1):

        results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, i)
        fm_file_path = "{}/feature_matches.csv".format(results_dir)

        if (not os.path.isfile(fm_file_path)) or (i in ignore_runs):
            continue   

        print("run: {}".format(i))    

        info[i] = {"score_16":[],
                   "score_32":[],
                   "score_64":[],
                   "score_128":[],
                   "score_256":[],
                   "score_total":[]}

        bad_values = ['nan', '-nan', 'inf', '-inf']

        with open(fm_file_path) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True

            for row in csv_reader:

                if not first:

                    try:

                        if i not in times.keys():
                            times[i] = int(row[0])

                        if not row[5] in bad_values: 
                            info[i]["score_16"] += [float(row[5])]
                        
                        if not row[6] in bad_values:
                            info[i]["score_32"] += [float(row[6])]
                        
                        if not row[7] in bad_values:
                            info[i]["score_64"] += [float(row[7])]
                        
                        if not row[8] in bad_values:
                            info[i]["score_128"] += [float(row[8])]
                        
                        if not row[9] in bad_values:
                            info[i]["score_256"] += [float(row[9])]
                        
                        if not row[10] in bad_values:
                            info[i]["score_total"] += [float(row[10])]
                    except Exception as e:
                        print(e)
                        print('Could not read row: '.format(row))

                first = False

    return info, times


def plot_inliers(feature_info, times, results_dir):

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
    x_labels = []
    for i in times.keys():
        x_labels.append(times[i].strftime('%d.%m-%H:%M'))

    layer_labels = {'score_16': 'layer 1 (size 16)',
                    'score_32': 'layer 2 (size 32)',
                    'score_64': 'layer 3 (size 64)',
                    'score_128': 'layer 4 (size 128)',
                    'score_256': 'layer 5 (size 256)',
                    'score_total': 'all layers (size 496)'}

    ############### Plot box plot of inliers for each repeat ###################
    for layer in feature_info.keys():
        f = plt.figure(figsize=(30, 13))
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        p = plt.boxplot(feature_info[layer], 
                        sym='', 
                        labels=x_labels, 
                        patch_artist=True,
                        boxprops={'facecolor':'C0', 'linewidth':3},
                        whiskerprops={'color':'black', 'linewidth':3},
                        medianprops={'color':'black', 'linewidth':3})
        
        plt.xticks(rotation=-75)
        plt.xlabel(r'\textbf{Repeat time (day.month-hh:mm)}', fontsize=50) 
        plt.ylabel(r'\textbf{Number of inliers}', fontsize=50)
        plt.xticks(fontsize=38) 
        plt.yticks(fontsize=48) 
        plt.title(r'\textbf{Feature matching score - %s' % layer_labels[layer], fontsize=50)

        # legend_elements = [matplotlib.lines.Line2D([0], [0], color='C0', lw=4, 
        #                                            label='Success'),
        #                    matplotlib.lines.Line2D([0], [0], color='C1', lw=4, 
        #                                            label='Failed')]                
        # plt.legend(handles=legend_elements, fontsize=20);

        plt.savefig('{}/feature_{}_box_seasonal.png'.format(results_dir, layer), 
                    bbox_inches='tight', format='png')
        plt.close()

    ############# Plot bar plot of average inliers for each repeat #############
    # f = plt.figure(figsize=(30, 12))
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # p = plt.bar(x_labels, feature_info['score_total'])

    # plt.xticks(rotation=-75)
    # plt.xlabel('Repeat time (day.month-hh:mm)', fontsize=22, weight='bold') 
    # plt.ylabel('Number of inliers', fontsize=22, weight='bold')
    # plt.xticks(fontsize=20) 
    # plt.yticks(fontsize=20) 
    # plt.ylim([min(avg_inliers) - 10, max(avg_inliers) + 10])
    # plt.title('Mean matched feature inliers for each repeat', 
    #           fontsize=22, weight='bold')

    # # legend_elements = [matplotlib.lines.Line2D([0], [0], color='C0', lw=4, 
    # #                                            label='Success'),
    # #                    matplotlib.lines.Line2D([0], [0], color='C1', lw=4, 
    # #                                            label='Failed')]                
    # # plt.legend(handles=legend_elements, fontsize=20);

    # plt.savefig('{}/feature_score_total_mean_seasonal.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()

    # ########### Plot cumulative distribution of inliers for each run ###########
    # plt.figure(figsize=(20, 12)) #
    # f.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plot_lines = []
    # labels = []
    # max_inliers = 0

    # for i in range(len(inliers)):
        
    #     max_val = np.max(inliers[i])
    #     if max_val > max_inliers:
    #         max_inliers = max_val
    #     n_bins_vis_range = 50
    #     n_bins_total = int((n_bins_vis_range * max_val) / 696)

    #     values, base = np.histogram(inliers[i], bins=n_bins_total)
    #     unity_values = values / values.sum()
    #     cumulative = np.cumsum(np.flip(unity_values))
    #     p = plt.plot(base[:-1], 1.0 - cumulative, linewidth=3)
    #     plot_lines.append(p[0])
    #     labels.append(x_labels[i])

    # plt.axvline(x=20.0, color='red', linewidth='3', linestyle='--')

    # plt.legend(plot_lines, labels, prop={'size': 16})
    # plt.xlim([max_inliers, 0])
    # plt.ylim([0, 1])
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.grid(True, which='both', axis='both', color='gray', linestyle='-', 
    #          linewidth=1)
    # plt.xlabel('Number of inliers', fontsize=20, weight='bold')
    # plt.ylabel('CDF over keyframes', fontsize=20, weight='bold')
    # plt.title('Cumulative distribution of keyframes with number of inliers', 
    #            fontsize=20, weight='bold')
    # plt.savefig('{}/inliers_dist_seasonal.png'.format(results_dir), 
    #             bbox_inches='tight', format='png')
    # plt.close()

def plot_data(info, times, data_dir):

    feature_scores = {'score_16': [], 
                      'score_32':[], 
                      'score_64':[], 
                      'score_128':[], 
                      'score_256':[], 
                      'score_total':[]}
    
    ind = 0
    for i in info.keys():

        feature_scores['score_16'] += [np.asarray(info[i]['score_16'])]
        feature_scores['score_32'] += [np.asarray(info[i]['score_32'])]
        feature_scores['score_64'] += [np.asarray(info[i]['score_64'])]
        feature_scores['score_128'] += [np.asarray(info[i]['score_128'])]
        feature_scores['score_256'] += [np.asarray(info[i]['score_256'])]
        feature_scores['score_total'] += [np.asarray(info[i]['score_total'])]
        times[i] = datetime.datetime.fromtimestamp(times[i] / 1e9)	      

    results_dir = "{}/graph.index/repeats".format(data_dir)

    plot_inliers(feature_scores, times, results_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')
    parser.add_argument('--numrepeats', default=None, type=int,
                        help='number of repeats (default: None)')

    args = parser.parse_args()

    ignore_runs = []

    failed_runs = [44, 45, 51]
  
    info, times = load_data(args.path, args.numrepeats, ignore_runs)

    plot_data(info, times, args.path);


