import argparse
import os
import cv2
import numpy as np
from dv import AedatFile
import time


parser = argparse.ArgumentParser(description='Reading the PKU-DAVIS-SOD dataset.')
parser.add_argument(
    '--input_file',
    default = './raw/test/normal',
    help='The input DVS dataset'
    )
parser.add_argument(
    '--filename',
    default = '001_test_normal.aedat4',
    help='The input file name'
    )


def make_color_histo(events, img=None, width=346, height=260):
    """
    simple display function that shows negative events as blue dots and positive as red one
    on a white background
    args :
        - events structured numpy array: timestamp, x, y, polarity.
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int.
        - height int.
    return:
        - img numpy array, height x width x 3).
    """
    if img is None:
        img = 255 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 255
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        ON_index = np.where(events['polarity'] == 1)

        img[events['y'][ON_index], events['x'][ON_index], :] = [30, 30, 220] * events['polarity'][ON_index][:, None]  # red

        OFF_index = np.where(events['polarity'] == 0)
        img[events['y'][OFF_index], events['x'][OFF_index], :] = [200, 30, 30] * (events['polarity'][OFF_index] + 1)[:,None]  # blue
    return img


def _main(args):
    input_file = os.path.expanduser(args.input_file)
    input_filename = args.filename

    start_time = time.time()

    # create event .npy files
    events_outfile = './asyn/events_npys/{}/{}/{}'.format(input_file.split('/')[-2], input_file.split('/')[-1], input_filename[:-7])
    if not os.path.exists(events_outfile):
        os.makedirs(events_outfile)

    frame_root = './aps_frames/{}/{}/{}'.format(input_file.split('/')[-2], input_file.split('/')[-1], input_filename[:-7])
    davis_root = './asyn/davis_images/{}/{}/{}'.format(input_file.split('/')[-2], input_file.split('/')[-1], input_filename[:-7])
    if not os.path.exists(davis_root):
        os.makedirs(davis_root)
    input_aedat_file = os.path.join(input_file, input_filename)

    frame_timestamps = []
    with AedatFile(input_aedat_file) as f:
        for frame in f['frames']:
            aps_timestamp = frame.timestamp
            frame_timestamps.append(aps_timestamp)

    # get frames and the corresponding event number index
    s_time = time.time()

    with AedatFile(input_aedat_file) as f:
        events = np.hstack([event for event in f['events'].numpy()])
        event_timestamps = [event[0] for event in events]
        start_time = frame_timestamps[0]
        finish_time = frame_timestamps[-1]

        i = 0
        cur_idx = -1
        while True:
            end_time = start_time + 40000
            if end_time > finish_time:
                break

            # get event number index
            event_start_index = np.searchsorted(event_timestamps, start_time)
            event_end_index = np.searchsorted(event_timestamps, end_time)

            corresponding_events = events[event_start_index:event_end_index]
            frame_id = max(np.searchsorted(frame_timestamps, start_time) - 1, 0)
            event_dict = {'event': corresponding_events, 'idx': frame_id}
            events_outfilename = events_outfile + '/{}.npy'.format(i)
            np.save(events_outfilename, event_dict)

            if not frame_id == cur_idx:
                davis_img = cv2.imread(os.path.join(frame_root, '{}.png'.format(frame_id)))
                cur_idx = frame_id
            else:
                davis_img = make_color_histo(corresponding_events)
            save_path = os.path.join(davis_root, '{}.png'.format(i))
            cv2.imwrite(save_path, davis_img)

            print('The {} frame, and event numbers are {}'.format(i, event_end_index-event_start_index))

            start_time += 10000
            i = i + 1

    e_time = time.time()

    print('The {} file has been done, the running time is {}'.format(input_aedat_file, e_time-s_time))


if __name__ == '__main__':
    _main(parser.parse_args())
    