import h5py
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-d', '--dataset-name', type=str,
                        help='name of the hdf5 dataset', default='data')
    parser.add_argument('-s', '--shape', type=int, nargs='+',
                        help='shape of the hdf5 dataset', default=[5, 100])
    args = parser.parse_args()

    # open hdf5 file in write mode
    with h5py.File(args.filename, 'w') as h5f:
        # create random data to be stored
        data = np.random.rand(*args.shape)

        # create a dataset of floats (f4) and store data
        dataset = h5f.create_dataset(name=args.dataset_name,
                                     shape=tuple(args.shape),
                                     dtype='f4',
                                     data=data)

    # open hdf5 file in read mode
    with h5py.File(args.filename, 'r') as h5f:
        # print file content
        print(list(h5f.keys()))

        # select dataset
        dataset = h5f[args.dataset_name]

        # access data
        data = dataset[...]

        # process data
        print(data.mean())


if __name__ == '__main__':
    main()
