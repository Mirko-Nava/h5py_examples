import h5py
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-d', '--dataset-name', type=str,
                        help='name of the hdf5 dataset', default='data')
    parser.add_argument('-s', '--strings', type=str, nargs='+',
                        help='strings to be stored (i.e.: an example of a "string with spaces")',
                        default=['this', 'is', 'an', 'example'])
    args = parser.parse_args()

    # open hdf5 file in write mode
    with h5py.File(args.filename, 'w') as h5f:
        # create string data
        data = np.array([args.strings], dtype=object)

        # create a dataset of strings
        # note: the dtype used here stores variable length strings
        # note: use 'S10' for fixed size strings of length 10
        dataset = h5f.create_dataset(name=args.dataset_name,
                                     shape=data.shape,
                                     dtype=h5py.special_dtype(vlen=str),
                                     data=data)

    # open hdf5 file in read mode
    with h5py.File(args.filename, 'r') as h5f:
        # select dataset
        dataset = h5f[args.dataset_name]

        # access data
        data = dataset[...]

        # print data
        print(data)


if __name__ == '__main__':
    main()
