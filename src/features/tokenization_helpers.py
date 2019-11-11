import os

import pandas as pd

SCRIPT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
DATA_DIRECTORY_PROCESSED_DFS = os.path.join(ROOT_DIRECTORY, 'data', 'processed', 'dfs')

FINAL_DF_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')
ML_ONLY_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'machine_learning_only.csv')


if __name__ == "__main__":
    df_ml = pd.read_csv(ML_ONLY_FILEPATH, encoding='utf-8')
