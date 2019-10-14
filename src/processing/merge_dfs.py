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
DATA_DIRECTORY_PROCESSED_DFS = os.path.join(ROOT_DIRECTORY, 'data', 'processed', 'dfs')
QUEUE_FILEPATH = os.path.join(SCRIPT_DIRECTORY, 'queue.txt')
BASE_URL = 'http://export.arxiv.org/oai2'

RAW_DATA_FILES = glob.glob(os.path.join(DATA_DIRECTORY_RAW, '*'))
PROCESSED_DF_FILES = glob.glob(os.path.join(DATA_DIRECTORY_PROCESSED_DFS, '*'))

if __name__ == "__main__":
    frames = []
    for i, filepath in enumerate(PROCESSED_DF_FILES):
        print(f'{i} - {filepath}')
        frames.append(pd.read_csv(filepath, encoding='utf-8'))
    frames = [pd.read_csv(filepath) for filepath in PROCESSED_DF_FILES]
    df = pd.concat(frames)
    full_df_filepath = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')
    df.to_csv(full_df_filepath, index=False, encoding='utf-8')
