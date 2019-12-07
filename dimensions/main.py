import h5py
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-d', '--dataset-name', type=str,
                        help='name of the hdf5 dataset', default='data')
    parser.add_argument('-ls', '--labeled-shape', action='append', nargs='+',
                        type=lambda kv: tuple(kv.split('=')), dest='labeled_shape',
                        help='labeled shape (i.e.: height=64 width=80 depth=3)')
    args = parser.parse_args()

    # convert labeled-shape to dict
    labeled_shape = dict(
        args.labeled_shape[0]) if args.labeled_shape is not None else dict()

    # convert values to int
    for k in labeled_shape.keys():
        labeled_shape[k] = int(labeled_shape[k])

    print(labeled_shape)

    # open hdf5 file in write mode
    with h5py.File(args.filename, 'w') as h5f:
        # create random data to be stored
        data = np.random.rand(*labeled_shape.values())

        # create a dataset of floats (f4) and store data
        dataset = h5f.create_dataset(name=args.dataset_name,
                                     shape=tuple(labeled_shape.values()),
                                     dtype='f4',
                                     data=data)

        # assign labels to each dataset dimension
        for label, dim in zip(labeled_shape.keys(), dataset.dims):
            dim.label = label

    # open hdf5 file in read mode
    with h5py.File(args.filename, 'r') as h5f:
        # select dataset
        dataset = h5f[args.dataset_name]

        # print dataset dimension labels
        print('dimension labels:', *[dim.label for dim in dataset.dims])


if __name__ == '__main__':
    main()
