import json
import pathlib

# Before using the script the raw dataset from LEAF (shakespeare_train_test.tgz)
# must be unzipped such that train .json files are in 'datasets/shakespeare/raw/train'
# and test .json files are in 'datasets/shakespeare/raw/test'.
# This script does:
# 1. Load the raw .json federated dataset
# 2. Save the federated dataset in .json files, one for each client for train and test set both
# Folder hierarchy:
# tff_experiment
#   |
#   datasets
#       |
#       shakespeare
#           |
#           preprocessed
#           |   |
#           |   test
#           |   train
#           |   shakespeare_train_test_preproc.tgz
#           |
#           raw
#               |
#               test
#               train
#               shakespeare_train_test.tgz
if __name__ == "__main__":
    # TODO
    # Set up paths
    raw_data_path = pathlib.Path('datasets/shakespeare/raw')
    train_path = raw_data_path/'train'
    test_path = raw_data_path/'test'
    output_path = pathlib.Path('datasets/shakespeare/preprocessed')
    out_train_path = output_path/'train'
    out_test_path = output_path/'test'
    
    # Train set
    for path in train_path.glob('*.json'):
        # Reading json
        with open(path, 'r') as file:
            f = json.load(file)
            users = list(f['users'])
            num_samples = list(f['num_samples'])
            hierarchies = list(f['hierarchies'])
            user_data = f['user_data']
            for i, u in enumerate(users):
                data = f['user_data'][u]
                data['hierarchies'] = hierarchies[i]
                data['num_samples'] = num_samples[i]
                # Writing json
                with open(out_train_path/'{}.json'.format(u), 'x') as file:
                    json.dump(data, file)
    
    # Test set
    for path in test_path.glob('*.json'):
        # Reading json
        with open(path, 'r') as file:
            f = json.load(file)
            users = list(f['users'])
            num_samples = list(f['num_samples'])
            hierarchies = list(f['hierarchies'])
            user_data = f['user_data']
            for u in users:
                data = f['user_data'][u]
                data['hierarchies'] = hierarchies[i]
                data['num_samples'] = num_samples[i]
                # Writing json
                with open(out_test_path/'{}.json'.format(u), 'x') as file:
                    json.dump(data, file)
