import numpy as np

for batch in range(0, 30):
    xs = []
    ys = []
    for i in range(1, 31):
        curr_images = np.load('/mnt/data/pigs/train/{}-0.npy'.format(i))
        print(curr_images.shape)
        xs.append(curr_images)
        ys.append([i] * len(curr_images))

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    print(xs.shape, ys.shape)
    print('Saving {}'.format(batch))
    np.save('/mnt/data/pigs/batches/xs-{}.npz'.format(batch), xs)
    np.save('/mnt/data/pigs/batches/ys-{}.npz'.format(batch), ys)
