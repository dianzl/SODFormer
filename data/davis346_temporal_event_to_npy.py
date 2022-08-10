import os
import numpy as np
from dv import AedatFile
import time


def write_timestamp_number(list_file, timestamp_indexes, number_indexes):
    for i in range(len(timestamp_indexes)):
        list_file.write(str(i) + " " + str(timestamp_indexes[i]) + " " + str(number_indexes[i]))
        print(str(i) + " " + str(timestamp_indexes[i]) + " " + str(number_indexes[i]))
        list_file.write('\n')


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


def _main():
    raw_root = './raw'
    subsets = os.listdir(raw_root)
    for subset in subsets:
        sub_root = os.path.join(raw_root, subset)
        scenes = os.listdir(sub_root)
        for scene in scenes:
            input_file = os.path.join(sub_root, scene)
            input_filenames = os.listdir(input_file)

            for i in range(len(input_filenames)):
                input_filename = input_filenames[i]
                start_time = time.time()

                # create event .npy files
                events_outfile = './events_npys/{}/{}/{}'.format(input_file.split('/')[-2], input_file.split('/')[-1], input_filename[:-7])
                if not os.path.exists(events_outfile):
                    os.makedirs(events_outfile)

                input_aedat_file = os.path.join(input_file, input_filename)

                # get events
                with AedatFile(input_aedat_file) as f:
                    events = np.hstack([event for event in f['events'].numpy()])
                    event_timestamps = [event[0] for event in events]

                # get frames and the corresponding event number index
                with AedatFile(input_aedat_file) as f:
                    j = 0
                    start_number = 0
                    for frame in f['frames']:
                        aps_timestamp = frame.timestamp

                        # get event number index
                        event_number_index = np.searchsorted(event_timestamps, aps_timestamp) # list B(event_timestamps) insert in list A(timestamp_indexes)
                        corresponding_events = events[start_number:event_number_index]
                        
                        events_outfilename = events_outfile + '/{}.npy'.format(j)
                        np.save(events_outfilename, corresponding_events)

                        print('The {} frame, and event numbers are {}'.format(j, event_number_index-start_number))

                        start_number = event_number_index
                        j = j + 1

                end_time = time.time()

                print('The {} file has been done, the running time is {}'.format(input_aedat_file, end_time-start_time))

if __name__ == '__main__':
    _main()
    