import os
import shutil
import sys

import configuration as conf


class MinPQ():
    def __init__(self, m):
        self.max_size = m
        self.n = 0
        self.pq = ['']

    def size(self):
        return self.n

    def empty(self):
        return self.size() == 0

    def insert(self, item):
        self.pq.append(item)
        self.n += 1
        self.swim(self.n)
        while self.size() > self.max_size:
            self.del_min()

    def exch(self, i, j):
        tmp = self.pq[i]
        self.pq[i] = self.pq[j]
        self.pq[j] = tmp

    def del_min(self):
        item = self.pq[1]
        self.exch(1, self.n)
        del self.pq[self.n]
        self.n -= 1
        self.sink(1)
        return item

    def swim(self, k):
        while k > 1:
            parent = k / 2
            if self.pq[k] < self.pq[parent]:
                self.exch(k, parent)
                k = parent
            else:
                break

    def sink(self, k):
        while k < self.n:
            l_child = k * 2
            r_child = l_child + 1
            min_k = k
            if l_child <= self.n and self.pq[l_child] < self.pq[min_k]:
                min_k = l_child

            if r_child <= self.n and self.pq[r_child] < self.pq[min_k]:
                min_k = r_child

            if min_k == k:
                break
            else:
                self.exch(min_k, k)
                k = min_k

class Transaction():
    def __init__(self, line):
        self.content = line
        split_line = line.split()
        self.img = split_line[0]
        self.compare_img = split_line[1]
        self.simi = float(split_line[2])

    def __cmp__(self, other):
        if self.simi < other.simi:
            return -1
        elif self.simi == other.simi:
            return 0
        else:
            return 1

import logging
level = logging.NOTSET
log_filename = '%s.log' % __file__
logging.basicConfig(level=level,
            format='%(asctime)s [%(levelname)s] %(message)s ',
            #filename=log_filename,
            datefmt='[%d/%b/%Y %H:%M:%S]')

if __name__ == '__main__':
    k = 100
    img_pq = {}
    line_count = -1
    filename = '/Volumes/Passport/data/gait-new-strategy/debug_data/17-02-05-21-01/similar_file_cl_bg.txt'
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    if len(sys.argv) >= 3:
        line_count = int(sys.argv[2])
    count = 0
    all_count = 0
    logging.info('read similar data from %s' % filename)
    for line in open(filename):
        all_count += 1
    logging.info('all_count = %d' % all_count)
    if line_count > 0:
        all_count = line_count
        logging.info('but, select %d line only' % all_count)
    for line in open(filename):
        count += 1
        if count > line_count > 0:
            break
        if count % 100000 == 0:
            logging.info('process %d/%d = %.3f' % (count, all_count, count * 1.0 / all_count))
        content = line.rstrip('\n')
        #logging.info('scan %s' % content)
        t = None
        try:
            t = Transaction(content)
        except:
            print('fail to create transaction with content:%s' % content)
        if t:
            if t.img not in img_pq.keys():
                img_pq[t.img] = MinPQ(k)
            img_pq[t.img].insert(t)
    top_k = []
    for key in img_pq.keys():
        pq = img_pq[key]
        filename = '-'.join(key.split('/')[-5:])
        dir_name = filename
        key_img_dir = '%s/%s' % (conf.project.debug_data_path, dir_name)
        if not os.path.exists(key_img_dir):
            os.makedirs(key_img_dir)
        shutil.copy(key, '%s/%s' % (key_img_dir, filename))

        while not pq.empty():
            top_k.append(pq.del_min())

        while len(top_k) > 0:
            t = top_k.pop()
            filename = '-'.join(t.compare_img.split('/')[-5:])
            filename = '%.3f_%s' % (t.simi, filename)
            shutil.copy(t.compare_img, '%s/%s' % (key_img_dir, filename))
