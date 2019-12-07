import h5py
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-d', '--dataset-name', type=str,
                        help='name of the hdf5 dataset', default='data')
    parser.add_argument('-a', '--arrays', type=str, nargs='+',
                        help='arrays to be stored (i.e.: "[0]" "[1,2,3]" "[10,20,30,40,50]")',
                        default=['[0]', '[1, 2, 3]', '[10, 20, 30, 40, 50]'])
    args = parser.parse_args()

    # open hdf5 file in write mode
    with h5py.File(args.filename, 'w') as h5f:
        # create ragged data
        data = np.array([eval(arr) for arr in args.arrays], dtype=object)

        # create a dataset of ragged floats
        # note: i didn't find a way to store data directly with create_dataset
        dataset = h5f.create_dataset(name=args.dataset_name,
                                     shape=data.shape,
                                     dtype=h5py.special_dtype(vlen=np.float32))

        # store data row by row
        for i in range(len(data)):
            dataset[i] = data[i]

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
