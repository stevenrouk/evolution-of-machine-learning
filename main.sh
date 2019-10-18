##########################################################
### Script demonstrating the flow of the full project. ###
###                                                    ###
### (Eventually want to move these into a make file.)  ###
##########################################################

# Pull data from API (note, this takes a very long time)
/anaconda3/envs/capstone2/bin/python src/data/make_dataset.py

# Process raw XML data into structured CSVs
/anaconda3/envs/capstone2/bin/python src/processing/process_raw_data.py

# Combine CSVs into a single CSV
/anaconda3/envs/capstone2/bin/python src/processing/merge_dfs.py