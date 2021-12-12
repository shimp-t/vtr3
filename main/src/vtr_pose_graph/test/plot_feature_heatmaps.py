import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import argparse

def load_data(data_dir, tod, keyframes, n_runs):

    info = {}
    stats = {}
    all_frames = {}

    for key in keyframes:
        info[key] = {'day':{'x':[], 'y':[], 'num_loc':0}, 'sunset':{'x':[], 'y':[], 'num_loc':0}, 
                     'night':{'x':[], 'y':[], 'num_loc':0}, 'sunrise':{'x':[], 'y':[], 'num_loc':0}}
        stats[key] = {}

    for run in tod.keys():

        results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, run)
        info_file_path = "{}/obs.csv".format(results_dir) 

        with open(info_file_path) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            first = True

            for row in csv_reader:

                if not first:

                    priv_id = int(row[2])
                    # live_id = int(row[1])

                    if priv_id not in all_frames.keys():
                        all_frames[priv_id] = {tod[run]: [run]}
                    else:
                        if tod[run] not in all_frames[priv_id].keys():
                            all_frames[priv_id][tod[run]] = [run]
                        else:
                            if run not in all_frames[priv_id][tod[run]]:
                                all_frames[priv_id][tod[run]] += [run]

                    if priv_id in keyframes:

                        # print("{}-{}-{}-{}".format(priv_id, live_id, run, tod[run]))

                        x = list(map(float, row[5::2]))
                        x_bin = [int(x_i // 16) for x_i in x]

                        y = list(map(float, row[6::2]))
                        y_bin = [int(y_i // 16) for y_i in y]
                           
                        info[priv_id][tod[run]]["x"] += x_bin
                        info[priv_id][tod[run]]["y"] += y_bin
                        info[priv_id][tod[run]]["num_loc"] += 1

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
                if len(all_frames[keyframe][time]) >= n_runs:
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
        tod_list = ['sunrise', 'day', 'sunset', 'night']
        for tod in tod_list:

            x_bin = keypoint_coord[keyframe][tod]['x']
            y_bin = keypoint_coord[keyframe][tod]['y']
            num_loc = keypoint_coord[keyframe][tod]['num_loc']

            bins = np.zeros((384, 512))

            for i in range(len(x_bin)):
                start_x = x_bin[i] * 16
                end_x = start_x + 16

                start_y = y_bin[i] * 16
                end_y = start_y + 16

                bins[start_y:end_y, start_x:end_x] += 1

            bins = bins / num_loc

            if np.max(bins) > max_bin:
                max_bin = np.max(bins)

            bins_list.append(bins)

        bins_dict[keyframe] = bins_list

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
            im = ax.imshow(bins_dict[keyframe][count], 
                           vmin=0, 
                           vmax=max_bin, 
                           cmap=plt.cm.GnBu_r)
            # plt.imshow(bins, cmap='jet')
            # plt.colorbar(im);
            ax.set_title(tod_list[count].capitalize(), fontsize=36)
            ax.axis('off')
            
            count += 1

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
   
        plt.savefig('{}/feature_counts_{}.png'.format(results_dir, keyframe), bbox_inches='tight', format='png')
        plt.savefig('{}/feature_counts_{}.pdf'.format(results_dir, keyframe), bbox_inches='tight', format='pdf')
        plt.savefig('{}/feature_counts_{}.svg'.format(results_dir, keyframe), bbox_inches='tight', format='svg')
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

    # EXP 2, labels as per thesis table
    tod = {13:'sunrise', 14: 'sunrise', 19: 'sunrise', 20: 'sunrise',
           1: 'day', 2: 'day', 3: 'day', 4: 'day', 5: 'day', 6: 'day', 7: 'day',
           17: 'day', 18: 'day', 21: 'day', 22: 'day', 23: 'day', 24: 'day',          
           25: 'day', 26: 'day',
           8: 'sunset', 9: 'sunset', 27: 'sunset', 28: 'sunset', 
           11: 'night', 12: 'night', 29: 'night'}

    tod_num_runs = {'sunrise':4, 'day':15, 'sunset':4, 'night':3}

    # # 5: 'day', 23: 'day', 9: 'evening'

    keyframes = [14, 950, 1228, 1624, 2603, 2725, 3091, 3291, 3359, 3481, 3703, 4023, 4406, 4818, 5053, 5389, 5664, 5844, 5965, 6127, 6376, 6683, 6813, 6884, 7082, 7296, 7496]
    n_runs = 3

    # [15, 951, 1227, 1699,  2433, 2784, 3255, 3294, 3346, 3426, 3436, 3453, 3481, 4139, 4705, 4714, 4722, 4915, 4918, 5030, 5053, 5064, 5088, 5111, 5391, 5443, 5689, 5692, 5693, 5702, 5707, 5724, 5727, 5730, 5732, 5755, 5760, 5762, 5765, 5788, 5794, 5795, 5803, 5815, 5816, 5838, 5841, 5843, 5845, 5854, 5932, 6019, 6029, 6043, 6050, 6068, 6076, 6081, 6087, 6088, 6091, 6092, 6093, 6095, 6096, 6097, 6099, 6100, 6101, 6103, 6105, 6107, 6109, 6110, 6111, 6112, 6113, 6115, 6116, 6118, 6119, 6120, 6121, 6122, 6123, 6124, 6125, 6126, 6127, 6128, 6129, 6130, 6131, 6132, 6133, 6134, 6135, 6137, 6138, 6139, 6140, 6141, 6142, 6143, 6144, 6146, 6147, 6148, 6150, 6151, 6153, 6154, 6155, 6156, 6157, 6158, 6159, 6160, 6161, 6162, 6164, 6165, 6166, 6167, 6168, 6169, 6170, 6171, 6172, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6180, 6181, 6182, 6183, 6184, 6185, 6187, 6189, 6190, 6191, 6192, 6193, 6194, 6195, 6196, 6197, 6198, 6199, 6200, 6201, 6202, 6203, 6204, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 6212, 6214, 6215, 6216, 6217, 6218, 6219, 6220, 6221, 6222, 6223, 6224, 6225, 6226, 6229, 6231, 6232, 6233, 6234, 6235, 6236, 6237, 6238, 6239, 6240, 6241, 6242, 6244, 6245, 6247, 6248, 6249, 6250, 6251, 6252, 6253, 6254, 6256, 6257, 6259, 6260, 6261, 6262, 6263, 6264, 6265, 6266, 6267, 6268, 6269, 6270, 6271, 6272, 6273, 6274, 6275, 6276, 6277, 6278, 6279, 6280, 6281, 6282, 6283, 6285, 6286, 6287, 6288, 6289, 6291, 6292, 6293, 6295, 6296, 6297, 6298, 6299, 6300, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312, 6313, 6314, 6315, 6316, 6317, 6318, 6319, 6320, 6321, 6322, 6323, 6324, 6325, 6326, 6327, 6328, 6329, 6330, 6331, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6339, 6341, 6342, 6343, 6345, 6346, 6347, 6348, 6349, 6350, 6351, 6352, 6353, 6355, 6356, 6357, 6358, 6359, 6360, 6361, 6362, 6363, 6365, 6367, 6368, 6369, 6371, 6373, 6375, 6376, 6377, 6378, 6379, 6380, 6381, 6382, 6383, 6384, 6385, 6388, 6392, 6397, 6399, 6402, 6404, 6406, 6407, 6409, 6410, 6411, 6413, 6414, 6415, 6416, 6417, 6423, 6487, 6493, 6508, 6511, 6512, 6514, 6516, 6519, 6520, 6522, 6528, 6531, 6550, 6552, 6564, 6569, 6570, 6571, 6573, 6574, 6577, 6585, 6592, 6594, 6597, 6600, 6605, 6609, 6615, 6617, 6618, 6619, 6625, 6626, 6627, 6635, 6637, 6639, 6643, 6647, 6650, 6651, 6652, 6653, 6656, 6659, 6661, 6663, 6671, 6672, 6674, 6676, 6679, 6683, 6689, 6691, 6697, 6701, 6703, 6707, 6708, 6713, 6717, 6719, 6723, 6737, 6738, 6753, 6760, 6770, 6772, 6773, 6780, 6782, 6783, 6788, 6789, 6791, 6803, 6804, 6820, 6905, 6906, 6918, 6924, 6932, 6934, 6936, 6939, 6940, 6941, 6942, 6945, 6946, 6948, 6951, 6953, 6957, 6958, 6962, 6970, 6971, 6975, 6976, 6982, 6983, 6985, 6986, 6987, 6988, 6991, 6993, 6995, 6996, 6998, 7001, 7004, 7007, 7009, 7010, 7011, 7013, 7014, 7015, 7016, 7017, 7018, 7020, 7021, 7022, 7023, 7024, 7025, 7026, 7039, 7045, 7046, 7047, 7048, 7049, 7050, 7051, 7053, 7054, 7055, 7056, 7057, 7068, 7084, 7086, 7092, 7099, 7107, 7108, 7109, 7110, 7111, 7116, 7123, 7135, 7136, 7137, 7141, 7150, 7151, 7152, 7153, 7194, 7206, 7207, 7208, 7209, 7210, 7224, 7266, 7283, 7284, 7286, 7300, 7302, 7324, 7326, 7334, 7336, 7358, 7367, 7368, 7379, 7381, 7385, 7403, 7413, 7415, 7438, 7442, 7461, 7473, 7479, 7480, 7481, 7482, 7498, 7500, 7503, 7509, 7511, 7512, 7513, 7515, 7517, 7531, 7533, 7535, 7538, 7546, 7550, 7576, 7581, 7634, 7635, 7637, 7641, 7642, 7669, 7674, 7684, 7689, 7691, 7707, 7710, 7726, 7727, 7733, 7766, 7767]


    # Note for run 5 we extract features for up to vertex 5xxx, and not all the way to the end 
    # EXP 1, labels as per thesis table
    # tod = {1: 'sunrise', 2: 'sunrise', 3: 'sunrise', 30: 'sunrise', 
    #        31: 'sunrise', 32: 'sunrise', 33: 'sunrise', 34: 'sunrise', 35: 'sunrise',  
    #        4: 'day', 5: 'day', 6: 'day', 7: 'day', 8: 'day', 9: 'day', 
    #        10: 'day', 11: 'day', 12: 'day', 13: 'day', 14: 'day', 15: 'day',
    #        16: 'day', 17: 'day', 36: 'day', 37: 'day', 38: 'day', 39: 'day', 40: 'day',
    #        18: 'sunset', 19: 'sunset', 20: 'sunset', 21: 'sunset', 22: 'sunset', 
    #        23: 'night', 24: 'night', 25: 'night', 26: 'night', 27: 'night', 
    #        28: 'night', 29: 'night'}

    # tod_num_runs = {'sunrise':9, 'day':19, 'sunset':5, 'night':7}

    # keyframes = [10, 200, 300, 396, 507, 604, 701,  801, 901, 1001]
    # n_runs = 5

    info = load_data(args.path, tod, keyframes, n_runs)
   
    plot_data(info, tod_num_runs, args.path);

