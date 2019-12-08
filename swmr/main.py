import h5py
import argparse
import numpy as np
from multiprocessing import Process, Event


class SwmrReader(Process):
    def __init__(self, event, args, timeout=1.0):
        super(SwmrReader, self).__init__()
        self.event = event
        self.timeout = timeout

        self.h5f = h5py.File(args.filename, 'r', libver='latest', swmr=True)
        self.dataset = self.h5f[args.dataset_name]

    def timeout_wait(self, t=None):
        self.event.wait(t if t is not None else self.timeout)
        self.event.clear()

    def run(self):
        while self.timeout_wait():
            self.dataset.refresh()
            shape = self.dataset.shape
            print('new shape:', shape)

        self.h5f.close()


class SwmrWriter(Process):
    def __init__(self, event, args):
        super(SwmrWriter, self).__init__()
        self.event = event
        self.args = args
        self.data = np.random.rand(self.args.length, 42)

        # open hdf5 file in write mode, with latest lib version
        # note: without libver='latest' SWMR will not work
        self.h5f = h5py.File(args.filename, 'w', libver='latest')

        # create an empty dataset of floats
        self.h5f.create_dataset(name=args.dataset_name,
                                shape=(0, 42),
                                maxshape=(None, 42),
                                dtype='f4')

        self.h5f.flush()

        # activate swmr mode
        # note: new groups and datasets cannot be created in SWMR mode
        self.h5f.swmr_mode = True
        self.dataset = self.h5f[self.args.dataset_name]

    def run(self):
         # iterate over data, appending to dataset
        for _ in range(0, self.args.length, self.args.batch_size):
            # compute new shape
            current_length = self.dataset.shape[0]
            new_length = min(current_length +
                             self.args.batch_size, self.args.length)
            new_shape = (new_length,) + self.dataset.shape[1:]

            # resize dataset
            self.dataset.resize(new_shape)

            # get batch of data
            batch = self.data[current_length:new_length, ...]

            # append a batch of data
            self.dataset[current_length:, ...] = batch

            # flush changes to file
            self.dataset.flush

            # send event to other process
            self.event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the hdf5 file', default='default.h5')
    parser.add_argument('-d', '--dataset-name', type=str,
                        help='name of the hdf5 dataset', default='data')
    parser.add_argument('-l', '--length', type=int,
                        help='length of the hdf5 dataset', default=int(1e6))
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='length of the batch appended to the dataset', default=int(1e5))
    args = parser.parse_args()

    # create event to send signals between processes
    event = Event()

    # create reader and writer processes
    writer = SwmrWriter(event, args)
    reader = SwmrReader(event, args)

    # run processes
    reader.start()
    writer.start()

    # wait for processes to end
    reader.join()
    writer.join()


if __name__ == '__main__':
    main()
