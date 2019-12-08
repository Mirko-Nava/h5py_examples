import h5py
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-d', '--dataset-name', type=str,
                        help='name of the hdf5 dataset', default='data')
    parser.add_argument('-ls', '--lengths', type=int, nargs='+',
                        help='lengths of dataset to be stored (i.e.: 2 3 4)',
                        default=[2, 3, 4])
    args = parser.parse_args()

    # create filename for each dataset
    filenames = [f'data_{index}.h5' for index in range(len(args.lengths))]

    # cumulative length used for indexing
    cum_lengths = np.cumsum([0] + args.lengths)

    # create a dataset for each length
    for filename, length in zip(filenames, args.lengths):
        print(filename, 'created')

        with h5py.File(filename, 'w') as h5f:
            # create random data to be stored
            data = np.random.rand(length, 42)

            # create a dataset of floats (f4) and store data
            h5f.create_dataset(name='data',
                               shape=data.shape,
                               dtype='f4',
                               data=data)

    # open hdf5 file in write mode
    # note: without libver='latest' virtual dataset will not work
    with h5py.File(args.filename, 'w', libver='latest') as h5f:
        # create a layout defining type and shape of the virtual datatset
        # note: layout supports resizable datasets
        layout = h5py.VirtualLayout(shape=(sum(args.lengths), 42),
                                    maxshape=(None, 42), dtype='f4')

        # create a virtual source for each dataset
        for i, filename in enumerate(filenames):
            start = cum_lengths[i]
            end = cum_lengths[i + 1]

            # a virtual source can be a h5py.Dataset or a filename, dataset name and shape
            vs = h5py.VirtualSource(filename, name='data',
                                    shape=(args.lengths[i], 42))
            layout[start:end, ...] = vs

        # create the virtual dataset using layout
        h5f.create_virtual_dataset(args.dataset_name, layout, fillvalue=-1.)

    # open hdf5 file in read mode
    with h5py.File(args.filename, 'r') as h5f:
        # select dataset
        dataset = h5f[args.dataset_name]

        print('virtual dataset shape:', dataset.shape)

        # print data
        print('first column:', dataset[:, 0])


if __name__ == '__main__':
    main()
