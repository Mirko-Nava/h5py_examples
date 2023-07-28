import cv2
import h5py
import argparse
import numpy as np
from pathlib import Path


def print_h5f_content(h5f, depth=0):
    # if h5f is a dataset
    if isinstance(h5f, h5py.Dataset):
        # print name, shape and dtype
        print('  ' * depth,
              f'{h5f.name}: shape {h5f.shape}, dtype {h5f.dtype}')
    # if h5f is a group
    elif isinstance(h5f, h5py.Group):
        # get list of content
        content = list(h5f.keys())
        # print name, # of elements
        print('  ' * depth, f'{h5f.name}: {len(content)} elements')
        # recurse on content
        for elem in content:
            print_h5f_content(h5f[elem], depth + 1)
    else:
        raise ValueError(
            f'print_h5f_content called with wrong type: {type(h5f)}')


def convert_dataset_to_h5f(h5f, values, keyfn, extractors):
    # get length of dataset
    length = len(values)

    # iterate over principal column values
    for i, principal_value in enumerate(values):
        # get key from current principal value
        key = keyfn(principal_value)

        # iterate over extractors
        for column_name, extractor in extractors.items():
            # get column value using extractor
            column_value = extractor(key)

            # if column does not exists inside h5f
            if column_name not in h5f:
                # get column attibutes
                dtype = column_value.dtype
                shape = column_value.shape

                # create dataset column
                h5f.create_dataset(name=column_name,
                                   shape=(length,) + shape,
                                   maxshape=(None,) + shape,
                                   dtype=dtype)
            
            h5f[column_name][i] = column_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-bp', '--base-path', type=str,
                        help='base path for the dataset', default='.')
    args = parser.parse_args()
    args.base_path = Path(args.base_path)

    # note: add logic here
    values = []
    extractors = {}
    def keyfn(value): return value
    # note: end

    # open hdf5 file in write mode
    with h5py.File(args.filename, 'w') as h5f:
        # converts dataset to hdf5
        convert_dataset_to_h5f(h5f, values, keyfn, extractors)

    # open hdf5 file in read mode
    with h5py.File(args.filename, 'r') as h5f:
        # print file structure
        print_h5f_content(h5f)


if __name__ == '__main__':
    main()
