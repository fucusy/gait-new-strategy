import sys
import logging
import tool
import data


level =logging.NOTSET
log_filename = '%s.log' % __file__
logging.basicConfig(level=level,
            format='%(asctime)s [%(levelname)s] %(message)s ',
            filename=log_filename,
            datefmt='[%d/%b/%Y %H:%M:%S]')


def recognize(video_id):
    logging.info('recognize %s now' % video_id)

    # human_seqs list of image of cropped human
    human_seqs = data.get_human_seqs(video_id)

    # top_seqs list of list of video_id
    top_seqs = data.retrieve_similar(human_seqs, top=3)

    assert len(top_seqs) == len(human_seqs)

    final_hid = data.get_most_id(top_seqs)
    hid, _, _, _ = tool.extract_info_from_path(video_id)

    if final_hid == hid:
        logging.info('correct recognition for %s' % video_id)
    else:
        logging.warn('wrong recognition hid:%s, for %s' % (final_hid, video_id))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
    else:
        video_id = '001-cl-01-090'
    #recognize(video_id)
    data.pre_compute_similarity()
