import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import argparse
import math

def load_data(data_dir, tod, keyframes):

    info = {}
    stats = {}
    all_frames = {}

    # for key in keyframes:
    info = {'day':{'x':[], 'y':[], 'h':[], 'num_loc':0}, 'sunset':{'x':[], 'y':[], 'h':[], 'num_loc':0}, 
                     'night':{'x':[], 'y':[], 'h':[], 'num_loc':0}, 'sunrise':{'x':[], 'y':[], 'h':[], 'num_loc':0}}
        # stats[key] = {}

    for run in tod.keys():

        results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, run)
        info_file_path = "{}/obs.csv".format(results_dir) 

        with open(info_file_path) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True

            for row in csv_reader:

                if not first:

                    priv_id = int(row[2])

                    x = list(map(float, row[5::2]))
                    x_bin = [int(x_i // 16) for x_i in x]

                    y = list(map(float, row[6::2]))
                    y_bin = [int(y_i // 16) for y_i in y]

                    h_bin = [math.floor(y_i) for y_i in y]
                       
                    info[tod[run]]["x"] += x_bin
                    info[tod[run]]["y"] += y_bin
                    info[tod[run]]["h"] += h_bin
                    info[tod[run]]["num_loc"] += 1

                    stats[priv_id][run] = len(x)
                    
                first = False


    print(stats)
    keep_frames = []
    for keyframe in all_frames.keys():
        # Must have runs from all 4 tod
        if len(all_frames[keyframe].keys()) == 4:
            all_complete = 0
            for time in all_frames[keyframe].keys():
                # Must have at least n runs for each tod, exp2 n = 3, exp1 n = 5
                if len(all_frames[keyframe][time]) >= 5:
                    all_complete += 1
            
            if all_complete == 4:
                keep_frames.append(keyframe)

    keep_frames.sort()
    print(keep_frames)
    print(len(tod.keys()))

    return info

def plot_data(keypoint_coord, tod_num_runs, data_dir):

    results_dir = "{}/graph.index/repeats".format(data_dir)

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

    max_bin = 0
    bins_dict = {}
    for keyframe in keypoint_coord.keys():
        
        # Plot an image with colour based on number of featres that landed
        # in each bin.
        bins_list = []
        bins_h_list = []
        tod_list = ['sunrise', 'day', 'sunset', 'night']
        for tod in tod_list:

            x_bin = keypoint_coord[tod]['x']
            y_bin = keypoint_coord[tod]['y']
            h_bin = keypoint_coord[tod]['h']
            num_loc = keypoint_coord[tod]['num_loc']

            bins = np.zeros((384, 512))
            bins_h = np.zeros(384)

            for i in range(len(h_bin)):
                bins_h[h_bin[i]] += 1

            for i in range(len(x_bin)):
                start_x = x_bin[i] * 16
                end_x = start_x + 16

                start_y = y_bin[i] * 16
                end_y = start_y + 16

                bins[start_y:end_y, start_x:end_x] += 1

            bins = bins / num_loc
            bins_h = bins_h / 384.0

            if np.max(bins) > max_bin:
                max_bin = np.max(bins)

            bins_list.append(bins)
            bins_h_list.append(bins_h)

        bins_dict[keyframe] = [bins_list, bins_h_list]

    for keyframe in keypoint_coord.keys():

        count = 0
        # f = plt.figure(figsize=(15, 11))
        # f.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))

        for ax in axes.flat:

            # plot_str = "22{}".format(count)
            # plt.subplot(int(plot_str))
            # im = plt.imshow(bins_list[i], 
            #                 vmin=0, 
            #                 vmax=max_bin, 
            #                 cmap=plt.cm.BuPu_r)
            ax.plot(bins_dict[keyframe][1][count])
            # im = ax.imshow(bins_dict[keyframe][count], 
            #                vmin=0, 
            #                vmax=max_bin, 
            #                cmap=plt.cm.viridis)
            # plt.imshow(bins, cmap='jet')
            # plt.colorbar(im);
            # ax.set_title(tod_list[count].capitalize(), fontsize=36)
            # ax.axis('off')
            
            count += 1

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(im, cax=cbar_ax)
   
        plt.savefig('{}/height_prob_{}.png'.format(results_dir, keyframe), bbox_inches='tight', format='png')
        plt.close()
   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')

    args = parser.parse_args()

    # ignore_runs = [6,7,10,14,15,16,17,18,24] 

    # EXP 2
    # tod = {13: 'morning', 19: 'morning', 20: 'morning',
    #        21: 'morning', 22: 'morning', 
    #        1: 'day', 2: 'day', 3: 'day', 4: 'day',
    #        8: 'evening', 25: 'evening', 
    #        26: 'evening', 27: 'evening', 28: 'evening', 
    #        11: 'night', 12: 'night', 29: 'night'}

    # tod_num_runs = {'morning':5, 'day':4, 'evening':5, 'night':3}

    # # 5: 'day', 23: 'day', 9: 'evening'

    # keyframes = [14, 950, 1228, 1624, 2603, 2725, 3091, 3291, 3359, 3481, 3703, 4023, 4406, 4818, 5053, 5389, 5664, 5844, 5965, 6127, 6376, 6683, 6813, 6884, 7082, 7296, 7496]

    # EXP 1
    # morning until (not including) 10
    # evening from (including) 18
    # night from 21 - 05:40
    tod = {1: 'sunrise', 2: 'sunrise', 3: 'sunrise', 30: 'sunrise', 
           31: 'sunrise', 32: 'sunrise', 33: 'sunrise', 34: 'sunrise', 35: 'sunrise',  
           4: 'day', 5: 'day', 6: 'day', 7: 'day', 8: 'day', 9: 'day', 
           10: 'day', 11: 'day', 12: 'day', 13: 'day', 14: 'day', 15: 'day',
           16: 'day', 17: 'day', 36: 'day', 37: 'day', 38: 'day', 39: 'day', 40: 'day',
           18: 'sunset', 19: 'sunset', 20: 'sunset', 21: 'sunset', 22: 'sunset', 
           23: 'night', 24: 'night', 25: 'night', 26: 'night', 27: 'night', 
           28: 'night', 29: 'night'}

    tod_num_runs = {'sunrise':9, 'day':19, 'sunset':5, 'night':7}

    keyframes = [10, 200, 300, 396, 507, 604, 701,  801, 901, 1001]

    info = load_data(args.path, tod, keyframes)
   
    plot_data(info, tod_num_runs, args.path);
