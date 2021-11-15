import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import argparse
from pyproj import Proj
import numpy as np
import cv2

import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

class BagFileParser():
    def __init__(self, bag_file):
        try:
            self.conn = sqlite3.connect(bag_file)
        except Exception as e:
            print(e)
            print('could not connect')
            raise Exception('could not connect')

        self.cursor = self.conn.cursor()

        table_names = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        print(len(topics_data))
        
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        self.topic_msg_message = {name_of:get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return messages as list of tuples [(timestamp0, message0), (timestamp1, message1), ...]
    def get_bag_messages(self, topic_name):
        
        topic_id = self.topic_id[topic_name]

        # Get from the db
        # rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchmany(size=10000)
        
        # Deserialise all and timestamp them
        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]   



def load_images(data_dir, results_dir, run_id):

    bag_file = '{}/front-xb3/run_{}/run_{}_0.db3'.format(data_dir, str(run_id).zfill(6), str(run_id).zfill(6))

    try:
        parser = BagFileParser(bag_file)
    except Exception as e:
        print("Could not open rosbag for run {}".format(run_id))

    messages = parser.get_bag_messages("/images") 

    for j in range(len(messages)):

        image_calib_msg = messages[j]

        image_msg = image_calib_msg[1].rig_images.channels[0].cameras[0]

        width, height = image_msg.width, image_msg.height
        image_data = image_msg.data

        image = np.array(image_data, dtype=np.dtype('uint8'))
        image = np.reshape(image,(height,width,3))

        cv2.imwrite("%s/%06d.png" % (results_dir, j), image)

        del image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Assuming following path structure:
    # vtr_folder/front-xb3/run_000xxx/metadata.yaml
    # vtr_folder/front-xb3/run_000xxx/run_000xxx_0.db3
    parser.add_argument('--path', default=None, type=str,
                        help='path to vtr folder (default: None)')
    parser.add_argument('--pathresults', default=None, type=str,
                        help='path to vtr folder (default: None)')
    parser.add_argument('--runid', default=None, type=int,
                         help='number of repeats (default: None)')

    args = parser.parse_args()

    load_images(args.path, args.pathresults, args.runid)    
