import h5py
import argparse
import numpy as np
from multiprocessing import Process, Event


class SwmrReader(Process):
    def __init__(self, event, args, timeout=1.0):
        super(SwmrReader, self).__init__()
        self.event = event
        self.args = args
        self.timeout = timeout

    def timeout_wait(self, t=None):
        response = self.event.wait(t if t is not None else self.timeout)
        self.event.clear()
        return response

    def run(self):
        # wait for initial signal: dataset has been created
        self.timeout_wait()

        # open hdf5 file in read mode, with latest lib version and swmr=True
        # note: without libver='latest' SWMR will not work
        self.h5f = h5py.File(self.args.filename, 'r',
                             libver='latest', swmr=True)

        # select dataset
        self.dataset = self.h5f[self.args.dataset_name]

        # while signals arrive before timeout
        while self.timeout_wait():
            # refresh dataset
            self.dataset.refresh()

            # get current shape
            shape = self.dataset.shape
            print('new shape:', shape)

        # close hdf5 file
        self.h5f.close()


class SwmrWriter(Process):
    def __init__(self, event, args):
        super(SwmrWriter, self).__init__()
        self.event = event
        self.args = args
        self.data = np.random.rand(self.args.length, 42)

    def run(self):
        # open hdf5 file in write mode, with latest lib version
        # note: without libver='latest' SWMR will not work
        self.h5f = h5py.File(self.args.filename, 'w', libver='latest')

        # create an empty dataset of floats
        self.h5f.create_dataset(name=self.args.dataset_name,
                                shape=(0, 42),
                                maxshape=(None, 42),
                                dtype='f4')

        # flush buffers, saves file to disk
        self.h5f.flush()

        # activate swmr mode
        # note: new groups and datasets cannot be created in SWMR mode
        self.h5f.swmr_mode = True

        # select dataset
        self.dataset = self.h5f[self.args.dataset_name]

        # send event to other process
        self.event.set()

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
            self.dataset.flush()

            # send event to other process
            self.event.set()

        # close hdf5 file
        self.h5f.close()


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
