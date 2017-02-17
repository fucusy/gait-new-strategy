__author__ = 'fucus'
import os
import time

time_now = time.localtime()

class project:
    base_folder = "/Users/fucus/Documents/irip/gait_recoginition/code/gait-new-strategy/"
    data_path = "/Volumes/Passport/data/gait-new-strategy/"
    debug_data_path = "%s/debug_data/%s/" % (data_path, time.strftime('%y-%m-%d-%H-%M', time_now))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(debug_data_path):
        os.makedirs(debug_data_path)

    debug_info_slower_speed = False

class data:
    video_path = "/Volumes/Passport/data/CASIA_full_gait_data_set/DatasetB/videos/"
