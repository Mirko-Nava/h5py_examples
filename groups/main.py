import h5py
import argparse
import numpy as np


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-ds', '--dataset-names', type=str, nargs='+',
                        help='names of the hdf5 datasets and groups',
                        default=['dataset1', 'dataset2',
                                 'group1/dataset3', 'group1/dataset4',
                                 'group2/subgroup1/subssubgroup1/dataset5'])
    args = parser.parse_args()

    # open hdf5 file in write mode
    with h5py.File(args.filename, 'w') as h5f:
        # iterate over dataset names
        for dataset_name in args.dataset_names:
            # create an empty dataset of long ints
            dataset = h5f.create_dataset(name=dataset_name,
                                         dtype='i8')

    # open hdf5 file in read mode
    with h5py.File(args.filename, 'r') as h5f:
        # print file structure
        print_h5f_content(h5f)

        # visit explores the tree in order
        h5f.visit(lambda h5f_name: print(h5f_name))


if __name__ == '__main__':
    main()
