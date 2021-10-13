import json
import pathlib

# Before using the script the raw dataset from LEAF (femnist_train_test.tgz)
# must be unzipped such that train .json files are in 'datasets/femnist/raw/train'
# and test .json files are in 'datasets/femnist/raw/test'.
# This script does:
# 1. Load the raw .json federated dataset
# 2. Save the federated dataset in .json files, one for each client for train and test set both
# Folder hierarchy:
# tff_experiment
#   |
#   datasets
#       |
#       femnist
#           |
#           preprocessed
#           |   |
#           |   test
#           |   train
#           |   femnist_train_test_preproc.tgz
#           |
#           raw
#               |
#               test
#               train
#               femnist_train_test.tgz
if __name__ == "__main__":
    # Set up paths
    raw_data_path = pathlib.Path('datasets/femnist/raw')
    train_path = raw_data_path/'train'
    test_path = raw_data_path/'test'
    output_path = pathlib.Path('datasets/femnist/preprocessed')
    out_train_path = output_path/'train'
    out_test_path = output_path/'test'
    
    # Train set
    for path in train_path.glob('*.json'):
        # Reading json
        with open(path, 'r') as file:
            f = json.load(file)
            users = list(f['users'])
            for u in users:
                d = f['user_data'][u]
                # Writing json
                with open(out_train_path/'{}.json'.format(u), 'x') as file:
                    json.dump(d, file)
    
    # Test set
    for path in test_path.glob('*.json'):
        # Reading json
        with open(path, 'r') as file:
            f = json.load(file)
            users = list(f['users'])
            for u in users:
                d = f['user_data'][u]
                # Writing json
                with open(out_test_path/'{}.json'.format(u), 'x') as file:
                    json.dump(d, file)
