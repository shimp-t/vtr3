import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import argparse
from PIL import Image, ImageDraw, ImageFont

def load_timstamp(data_dir, run_ind):

    results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, run_ind)
    info_file_path = "{}/info.csv".format(results_dir) 

    with open(info_file_path) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        row_ind = 0

        for row in csv_reader:

            if row_ind == 1:
                timestamp = int(row[0]) 

            row_ind+=1

    return timestamp

def draw_label(images_path, video_path, run_id, i, description_str, dt, image_index):

    image_file = "{}/{}.png".format(images_path, i)
    image_file_new = "{}/{}.png".format(video_path, str(image_index).zfill(6))

    dm_str  = dt.strftime('%d/%m')
    hm_str  = dt.strftime('%H:%M')

    img = Image.open(image_file)
    draw_img = ImageDraw.Draw(img)

    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
    draw_img.multiline_text((5,10), 
                             "{}\n{}\n{}".format(description_str, dm_str, hm_str), 
                             font=fnt, 
                             fill=(0, 255, 0), 
                             stroke_width=2, 
                             stroke_fill="black")
    
    img.save(image_file_new, "png")
    img.close()
    del draw_img


def label_images(timestamp, data_dir, run_id, start, end):  

    results_dir = "{}/graph.index/repeats/{}/results".format(data_dir, run_id)
    images_path = "{}/images".format(results_dir)
    video_path = "{}/graph.index/repeats/videos_parallel/{}".format(data_dir, run_id)

    if not os.path.exists(video_path):
        os.makedirs(video_path)

    if run_id == 0:
        description_str = "Teach"
    else:
        description_str = "Repeat"

    dt = datetime.datetime.fromtimestamp(timestamp / 1e9) 
   
    count = 0
    for i in range(start, end + 1):

        draw_label(images_path, video_path, 
                   run_id, i, 
                   description_str, dt, count)

        count +=1

    draw_label(images_path, video_path, run_id, end+1, description_str, dt, count)   


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str,
                        help='path to results dir (default: None)')

    args = parser.parse_args()


    run_info = {1:[1592,3517], 3:[1484,3529], 7:[1064,2686], 
                8:[731,1950], 11:[1791, 4213], 12:[1867,4343],
                13:[892,2297], 20:[467,1029], 21:[495,1082],
                25:[431,959], 28:[431,932], 29:[533,1189]} 
    
    # image_index = 0
    for run in run_info.keys():

        timestamp = load_timstamp(args.path, run)

        # dt = datetime.datetime.fromtimestamp(timestamp / 1e9) 
        # print("{}-{}".format(run, dt.strftime('%H:%M')))

        # step = 1 if run in [7,21, 22] else 2
   
        image_index = label_images(timestamp, args.path, run, 
                                   run_info[run][0], run_info[run][1]);