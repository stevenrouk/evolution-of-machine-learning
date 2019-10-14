import os

import numpy as np

SCRIPT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
QUEUE_FILEPATH = os.path.join(SCRIPT_DIRECTORY, 'queue.txt')


class Status:
    """Status class for ArXivDataPuller class."""

    STOPPED = 0
    STOPPED_ERROR = 1
    RUNNING = 2

class ArXivDataPuller:
    """Pulls ArXiv metadata."""

    def __init__(self):
        self.status = Status.STOPPED
        self.queue = self._get_queue()
    
    def _get_queue(self):
        return np.loadtxt(QUEUE_FILEPATH, dtype=str)
    
    def _save_queue(self):
        np.savetxt(QUEUE_FILEPATH, list(self.queue), fmt='%s')

if __name__ == "__main__":
    a = ArXivDataPuller()
