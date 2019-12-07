import h5py
import argparse
import numpy as np
from tqdm import trange


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-d', '--dataset-name', type=str,
                        help='name of the hdf5 dataset', default='data')
    parser.add_argument('-l', '--length', type=int,
                        help='length of the hdf5 dataset', default=int(1e6))
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='length of the batch appended to the dataset', default=10)
    args = parser.parse_args()

    # open hdf5 file in write mode
    with h5py.File(args.filename, 'w') as h5f:
        # create random data to be stored
        data = np.random.rand(args.length, 42)

        # create an empty dataset of floats
        # note: maxshape is used to define which dimensions can be resized
        dataset = h5f.create_dataset(name=args.dataset_name,
                                     shape=(0, 42),
                                     maxshape=(None, 42),
                                     dtype='f4')

        # iterate over data, appending to dataset
        for _ in trange(0, args.length, args.batch_size, desc='appending'):
            # compute new shape
            current_length = dataset.shape[0]
            new_length = min(current_length + args.batch_size, args.length)
            new_shape = (new_length,) + dataset.shape[1:]

            # resize dataset
            dataset.resize(new_shape)

            # get batch of data
            batch = data[current_length:new_length, ...]

            # append a batch of data
            dataset[current_length:, ...] = batch

    # open hdf5 file in read mode
    with h5py.File(args.filename, 'r') as h5f:
        # select dataset
        dataset = h5f[args.dataset_name]

        # print shape of dataset
        print('final shape:', dataset.shape)


if __name__ == '__main__':
    main()
