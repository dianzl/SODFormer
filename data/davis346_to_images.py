import os
import cv2
import numpy as np
from dv import AedatFile


def get_event_indexes(event_index_path):
    with open(event_index_path) as f:
        event_indexes = f.readlines()
    event_indexes  = [c.strip() for c in event_indexes]
    return event_indexes


def make_davis_histo(events, img, width=346, height=260):
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

    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        ON_index = np.where(events['polarity'] == 1)

        img[events['y'][ON_index], events['x'][ON_index], :] = [30, 30, 220] * events['polarity'][ON_index][:, None]  # red

        OFF_index = np.where(events['polarity'] == 0)
        img[events['y'][OFF_index], events['x'][OFF_index], :] = [200, 30, 30] * (events['polarity'][OFF_index] + 1)[:,None]  # blue
    return img


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
                # create image files
                aps_frame_outfile = './aps_frames/{}/{}/{}'.format(input_file.split('/')[-2], input_file.split('/')[-1], input_filename[:-7])
                if not os.path.exists(aps_frame_outfile):
                    os.makedirs(aps_frame_outfile)
                input_aedat_file = os.path.join(input_file, input_filename)

                # get frames
                with AedatFile(input_aedat_file) as f:
                    j = 0
                    for frame in f['frames']:
                        aps_image = frame.image
                        aps_image = aps_image.astype(np.uint8)

                        aps_frame_outfilename = aps_frame_outfile + '/{}.png'.format(j)
                        cv2.imwrite(aps_frame_outfilename, aps_image)

                        j = j + 1

                print('The {} frame has been done!'.format(i))


if __name__ == '__main__':
    _main()
    