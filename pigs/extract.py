from skvideo.io import FFmpegReader
from scipy.misc import imsave
import shutil
import os
# import numpy as np

for num in range(1, 31):
    fnm = '/home/alex/Downloads/train/%d.mp4' % num
    print(fnm)
    try:
        shutil.rmtree('/mnt/data/pigs/imgs/{}'.format(num))
        shutil.rmtree('/mnt/data/pigs/val_imgs/{}'.format(num))
    except:
        pass

    os.mkdir('/mnt/data/pigs/imgs/{}'.format(num))
    os.mkdir('/mnt/data/pigs/val_imgs/{}'.format(num))

    vid = FFmpegReader(fnm)
    # vid = cv2.VideoCapture(fnm)

    # success, images = vid.read()
    # print(success)
    # images = [images]
    # images = []

    # split_num = 30
    # frame_num = 2950
    # split_size = frame_num // split_num

    val_split = 2200

    count = 0
    for frame_num, frame in enumerate(vid.nextFrame()):
        # success, next_image = vid.read()
        if frame_num % 100 == 0:
            print(frame_num)
        if frame_num < val_split:
            imsave('/mnt/data/pigs/imgs/{}/{}.png'.format(num, frame_num),
                   frame)
        else:
            imsave('/mnt/data/pigs/val_imgs/{}/{}.png'.format(
                num, frame_num - val_split), frame)

        # if len(images) == split_size:
        # print('Saving data/%d-%d.npy' % (num, count))
        # np.save('/mnt/data/pigs/train/%d-%d.npy' % (num, count),
        #         np.array(images))
        # count += 1
        # images = []

        # images.append(frame)

#       cv2.imwrite("%d/frame%d.jpg" % (num, count), image)     # save frame as JPEG file

# np.save('data/%d-%d.npy' % (num, count), np.array(images))
