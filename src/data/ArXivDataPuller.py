import atexit
from collections import deque
import os
import pickle
import re
import time

import bs4
import requests
import numpy as np

SCRIPT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
QUEUE_FILEPATH = os.path.join(SCRIPT_DIRECTORY, 'queue.txt')
BASE_URL = 'http://export.arxiv.org/oai2'


def make_name_file_safe(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()

class Status:
    """Status class for ArXivDataPuller class."""

    STOPPED = 0
    STOPPED_ERROR = -1
    STOPPED_MANUALLY = 1
    RUNNING = 2

class ArXivDataPuller:
    """Pulls ArXiv metadata."""

    def __init__(self):
        self.status = Status.STOPPED
        self.queue = self._get_queue()
        self.last_failed_status_code = None
        self.last_failed_request = None
    
    def _get_queue(self):
        data = np.loadtxt(QUEUE_FILEPATH, dtype=str)
        return deque(np.atleast_1d(data))
    
    def _save_queue(self):
        np.savetxt(QUEUE_FILEPATH, list(self.queue), fmt='%s')
    
    def _run_example(self):
        try:
            while True:
                print(self.queue.popleft())
                time.sleep(1)
        except KeyboardInterrupt:
            self.status = Status.STOPPED_MANUALLY
            self._save_queue()
    
    def run(self):
        try:
            self.status = Status.RUNNING
            while self.status == Status.RUNNING and len(self.queue) > 0:
                # Get a URL from the queue to run
                resumption_token = self.queue.popleft()

                # Build the URL
                url = self.build_resumption_url(resumption_token)

                # Ping the API
                r = requests.get(url)

                # Process the URL
                # Process #1 - if the response is 200 and the file looks good, save
                if r.status_code == 200:
                    output_filename = make_name_file_safe(f'{resumption_token}') + '.xml'
                    output_filepath = os.path.join(DATA_DIRECTORY_RAW, output_filename)
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(r.text)
                    # Log that the pull was successful
                    with open('log.log', 'a') as f:
                        f.write(f'SUCCESS: {resumption_token}\n')
                    print(f'Wrote {resumption_token}')

                # Get the next resumption_token and add it to the queue
                next_token = self.get_resumption_token(r.text)
                self.queue.append(next_token)

                # Process #2 - If the response is 503 (wait), then wait and try again.
                if r.status_code == 503:
                    self.queue.append(resumption_token)
                    self.last_failed_status_code = r.status_code
                    self.last_failed_request = r
                    self._wait_from_wait_text(r.text)
                    print('Continuing')

                # Process #3 - if the response isn't 200 or 503, write an error log with the status_code / url / etc.
                if r.status_code != 200:
                    self.queue.append(resumption_token)
                    print(f'The token {resumption_token} failed with error code {r.status_code}')
                    self.last_failed_status_code = r.status_code
                    self.last_failed_request = r
                    self.status = Status.STOPPED_ERROR
                    # Log that the pull was not successful
                    with open('log.log', 'a') as f:
                        f.write(f'FAILED: {resumption_token} - {r.status_code} - {r.reason} - {r.text}\n')


                # Reset the resumption_token
                resumption_token = None
        except KeyboardInterrupt:
            # If the XML file exists, delete it (just to be safe, in case of corrupted write)
            if resumption_token:
                output_filename = make_name_file_safe(f'{resumption_token}.xml')
                output_filepath = os.path.join(DATA_DIRECTORY_RAW, output_filename)
                if os.path.exists(output_filepath):
                    os.remove(output_filepath)

            # Add resumption token back to the queue
            self.queue.append(resumption_token)

            # Save the queue
            self._save_queue()
            
            # Specify that we stopped the process manually
            self.status = Status.STOPPED_MANUALLY

    @staticmethod
    def build_url(verb=None, metadata_prefix=None, set_spec=None, resumption_token=None):
        url = [BASE_URL, '?']
        if verb:
            url.extend(['verb=', verb, '&'])
        if metadata_prefix:
            url.extend(['metadataPrefix=', metadata_prefix, '&'])
        if set_spec:
            url.extend(['set=', set_spec, '&'])
        if resumption_token:
            url.extend(['resumptionToken=', resumption_token, '&'])

        if url[-1] == '&':
            url.pop()
        if url[-1] == '?':
            url.pop()
        
        return ''.join(url)

    @staticmethod
    def build_first_call_url(set_spec=None):
        """Format: http://export.arxiv.org/oai2?verb=ListRecords&set=stat&metadataPrefix=oai_dc"""
        return ArXivDataPuller.build_url(verb='ListRecords', metadata_prefix='oai_dc', set_spec=set_spec)

    @staticmethod
    def build_resumption_url(resumption_token):
        """Format: http://export.arxiv.org/oai2?verb=ListRecords&resumptionToken=3902257|1001"""
        if resumption_token == 'FIRST':
            return ArXivDataPuller.build_first_call_url()
        else:
            return ArXivDataPuller.build_url(verb='ListRecords', resumption_token=resumption_token)

    @staticmethod
    def get_resumption_token(response_text):
        # resumption token substring
        try:
            resump_text = r.text[r.text.find('resumptionToken'):]
            return re.search(r'>(\d+\|\d+)<', resump_text).groups()[0]
        except:
            return None
    
    def print_status(self):
        print(self.status)
    
    def _wait_from_wait_text(self, text):
        soup = bs4.BeautifulSoup(text, 'html.parser')
        wait_text = soup.h1.text
        seconds = wait_text.split()[2]
        print(f'{wait_text}: waiting {seconds} seconds')
        time.sleep(seconds + 2)


class ArXivDataPullerMonitor:

    def __init__(self):
        self.data_puller = self._load_data_puller()
        self.data_puller_filepath = os.path.join(SCRIPT_DIRECTORY, 'data_puller.pkl')
    
    def _load_data_puller(self):
        # Check to see if a pickled data puller exists. If it does, unpickle it.
        data_puller_filepath = os.path.join(SCRIPT_DIRECTORY, 'data_puller.pkl')
        if os.path.exists(data_puller_filepath):
            with open(data_puller_filepath, 'rb') as f:
                data_puller = pickle.load(f)
        # Otherwise, create a new data puller
        else:
            data_puller = ArXivDataPuller()
        
        return data_puller
    
    def save_data_puller(self):
        with open(self.data_puller_filepath, 'wb') as f:
            pickle.dump(self.data_puller, f)

if __name__ == "__main__":
    a = ArXivDataPullerMonitor()
    atexit.register(a.save_data_puller)
    atexit.register(a.data_puller.print_status)
    #import pdb;pdb.set_trace()

    a.data_puller.run()
