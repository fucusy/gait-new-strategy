import os
import logging
from multiprocessing import Pool
import time,random

level = logging.NOTSET
log_filename = '%s.log' % __file__
logging.basicConfig(level=level,
            format='%(asctime)s [%(levelname)s] %(message)s ',
            filename=log_filename,
            datefmt='[%d/%b/%Y %H:%M:%S]')


import configuration as conf
from data import get_human_seqs, similar_between_img
import tool

def make_data_compute_simi_single(video_id):
    name = video_id
    print 'Run task %s (%s)...' % (name, os.getpid())
    start = time.time()

    match_seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', ]
    views = ['090']
    similar_filename = '%s/single_similar_file_cl_bg_%s.txt' % (conf.project.debug_data_path, video_id)
    similar_file = open(similar_filename, 'w')
    img_path_list1, data1 = get_human_seqs(video_id)
    hid, _, _, _ = tool.extract_info_from_path(video_id)
    logging.info('process video id:%s' % video_id)
    for seq in match_seqs:
        for view in views:
            video_id2 = '%s-%s-%s' % (hid, seq, view)
            img_path_list2, data2 = get_human_seqs(video_id2)
            for k in range(len(img_path_list1)):
                for l in range(len(img_path_list2)):
                    filename1 = img_path_list1[k]
                    filename2 = img_path_list2[l]
                    simi = similar_between_img(data1[k], data2[l])
                    similar_file.write('%s\t%s\t%.03f\n' % (filename1, filename2, simi))
    similar_file.close()
    end = time.time()
    print 'Task %s runs %0.2f seconds.' % (name, (end - start))

def make_data_compute_simi():
    print 'Parent process %s.' % os.getpid()
    hids = ['%03d' % i for i in range(1, 125)]
    views = ['090']
    seqs = ['cl-01', 'cl-02', 'bg-01', 'bg-02', ]

    if not os.path.exists(conf.project.debug_data_path):
        os.makedirs(conf.project.debug_data_path)
    video_ids = []
    for hid in hids:
        for seq in seqs:
            for view in views:
                video_id = '%s-%s-%s' % (hid, seq, view)
                video_ids.append(video_id)

    p = Pool()
    for video_id in video_ids:
        p.apply_async(make_data_compute_simi_single, args=(video_id,))
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    print 'All subprocesses done.'

    print 'cat to one file'

    similar_filename = '%s/similar_file_cl_bg.txt' % conf.project.debug_data_path
    single_filename = '%s/single_similar_file_cl_bg_*' % conf.project.debug_data_path
    cmd = 'cat %s > %s' % (single_filename, similar_filename)
    os.system(cmd)
    print 'done all'

if __name__ == '__main__':
    make_data_compute_simi()