__author__ = 'fucus'
import os
from topK import MinPQ
from tool import extract_info_from_path

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

if __name__ == '__main__':
    DATA_PATH = '/Volumes/Passport/data/gait-new-strategy/debug_data/17-02-16-22-00'
    k = 7
    works_count = 0
    all_count = 0
    filenames = os.listdir(DATA_PATH)
    #filenames = ['single_match_file_066-bg-01-090.txt']
    for filename in filenames:
        all_count += 1
        seq_pq = MinPQ(k)
        origin_seq = ''
        if not filename.endswith('.txt'):
            print('ignore filename %s' % filename)
            continue
        full_filename = '%s/%s' % (DATA_PATH, filename)
        count = 0
        for line in open(full_filename):
            count += 1

        if count < (124 * 4 - 100):
            print('ignore filename %s, do not have enough compare data' % filename)
            continue
        t = None
        for line in open(full_filename):
            content = line.rstrip('\n')
            try:
                t = Transaction(content)
            except Exception as e:
                print('error %s, fail to create transaction with content:%s' % (e, content))
            if t:
                seq_pq.insert(t)
                origin_seq = t.img

        origin_hid, _, _, _ = extract_info_from_path(origin_seq)

        top_k = []
        while not seq_pq.empty():
            top_k.append(seq_pq.del_min())

        print('for %s' % origin_seq)
        id_count = {}
        while len(top_k) > 0:
            t = top_k.pop()
            print(t.content)
            hid, _, _, _ = extract_info_from_path(t.compare_img)
            if hid not in id_count:
                id_count[hid] = 0
            id_count[hid] += 1
        max_hid = ''
        max_count = 0
        for hid, count in id_count.items():
            if count > max_count:
                max_hid = hid
                max_count = count
        if origin_hid in id_count:
            origin_count = id_count[origin_hid]
        else:
            origin_count = 0
        if max_hid == origin_hid:
            print('works %d/%d' % (max_count, k))
            works_count += 1
        else:
            print('fails %d/%d vs %d/%d of hid:%s' % (origin_count, k, max_count, k, max_hid))
        print('\n\n')
    print('works %d/%d' % (works_count, all_count))
