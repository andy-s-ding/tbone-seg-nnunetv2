import os
import pickle
import sys


def main(argv):
    pickle_path = argv[0]
    with open(pickle_path, "rb") as f:
        pickle_data = pickle.load(f)
        print(pickle_data)
      
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    # usage: python create_datasplit.py ../nii_images/