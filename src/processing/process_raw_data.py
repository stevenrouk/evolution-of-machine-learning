from collections import deque
import glob
import os
import re
import time

import bs4
import requests
import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
QUEUE_FILEPATH = os.path.join(SCRIPT_DIRECTORY, 'queue.txt')
BASE_URL = 'http://export.arxiv.org/oai2'

RAW_DATA_FILES = glob.glob(os.path.join(DATA_DIRECTORY_RAW, '*'))

def parse_record(record):
    identifier = record.find('identifier').text
    url = record.find('dc:identifier').text
    title = record.find('dc:title').text
    set_spec = record.find('setSpec').text
    subjects = ','.join([sub.text for sub in record.find_all('dc:subject')])
    authors = ','.join([aut.text for aut in record.find_all('dc:creator')])
    dates = ','.join([d.text for d in record.find_all('dc:date')])
    description = record.find('dc:description').text

    return [identifier, url, title, set_spec, subjects, authors, dates, description]

if __name__ == "__main__":
    for raw_data_filepath in RAW_DATA_FILES:
        print(f'processing {raw_data_filepath}')
        # Check to see if file already exists
        new_filepath = os.path.join(DATA_DIRECTORY_PROCESSED, 'dfs', os.path.split(raw_data_filepath)[1])
        new_filepath = new_filepath.split('.')[0] + '.csv'
        if os.path.exists(new_filepath):
            # don't process
            print(f'already processed {raw_data_filepath}')
            continue
        else:
            print(f'processing {raw_data_filepath}')

        # Read in data and convert to BeautifulSoup format
        with open(raw_data_filepath, 'r', encoding='utf-8') as f:
            raw_xml = f.read()
        
        soup = bs4.BeautifulSoup(raw_xml, 'xml')

        # get records
        records = soup.find_all('record')
        
        # parse records
        parsed_columns = ['identifier', 'url', 'title', 'set_spec', 'subjects', 'authors', 'dates', 'description']
        parsed = []
        for i, rec in enumerate(records):
            #print(i)
            if 'status' in rec.header.attrs and rec.header.attrs['status'] == 'deleted':
                print('record deleted')
                pass
            else:
                parsed.append(parse_record(rec))

        df = pd.DataFrame(parsed, columns=parsed_columns)
        df.to_csv(new_filepath, index=False, encoding='utf-8')
