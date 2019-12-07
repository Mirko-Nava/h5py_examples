import h5py
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-d', '--dataset-name', type=str,
                        help='name of the hdf5 dataset', default='data')
    parser.add_argument('-m', '--metadata', action='append', nargs='+',
                        type=lambda kv: tuple(kv.split('=')), dest='metadata',
                        help='sequence of key value pairs (i.e.: a=2 b="a long message")')
    args = parser.parse_args()

    # convert metadata to dict
    metadata = dict(args.metadata[0]) if args.metadata is not None else dict()

    # open hdf5 file in write mode
    with h5py.File(args.filename, 'w') as h5f:
        # note: groups can have metadata too
        h5f.attrs['metadata'] = 'true'

        # create an empty dataset of long ints
        dataset = h5f.create_dataset(name=args.dataset_name, dtype='i8')

        # iterate over metadata
        for k, v in metadata.items():
            dataset.attrs[k] = v

    # open hdf5 file in read mode
    with h5py.File(args.filename, 'r') as h5f:
        # select dataset
        dataset = h5f[args.dataset_name]

        # print dataset metadata
        for k, v in dataset.attrs.items():
            print(k, '=', v)


if __name__ == '__main__':
    main()
