import os

import pandas as pd

SCRIPT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
DATA_DIRECTORY_PROCESSED = os.path.join(ROOT_DIRECTORY, 'data', 'processed')
FINAL_DF_FILEPATH = os.path.join(DATA_DIRECTORY_PROCESSED, 'final.csv')

if __name__ == "__main__":
    df = pd.read_csv(FINAL_DF_FILEPATH, encoding='utf-8')

    # Create subset without physics papers
    print("writing 'no_physics.csv'...")
    df_not_physics = df[~df['set_spec'].str.contains('physics')]
    df_not_physics.to_csv(os.path.join(DATA_DIRECTORY_PROCESSED, 'no_physics.csv'), index=False, encoding='utf-8')
    print("success")

    # Create subset of just papers that have "machine learning" in one of the subjects
    print("writing 'machine_learning_only.csv'...")
    df_ml = df[df['subjects'].str.lower().str.contains('machine learning')]
    df_ml.to_csv(os.path.join(DATA_DIRECTORY_PROCESSED, 'machine_learning_only.csv'), index=False, encoding='utf-8')
    print("success")
