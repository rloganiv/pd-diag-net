"""Download and process the Parkinsons Speech Dataset"""

import argparse
import urllib

#---Constants---

# Location of the dataset
FILE_URL = '''
https://archive.ics.uci.edu/ml/machine-learning-databases/00301/Parkinson_Multiple_Sound_Recording_Data.rar
'''

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_file', type=str, 
    help='file to write data to',
    default='Parkinson_Multiple_Sound_Recording_Data.rar')
parser.add_argument('-d', '--data_dir', type=str,
    help='directory to download dataset to',
    default='data/')
args = parser.parse_args()





if __name__ == '__main__':
    urllib.urlretrieve(FILE_URL, args.data_dir + args.output_file)

