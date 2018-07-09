import h5py
import argparse

parser = argparse.ArgumentParser(description='Test file for reading h5 texts')
parser.add_argument('--target_text', '-t', help='target text to be read')

def main():
    global args
    args = parser.parse_args()
    filename = args.target_image
    texts = h5py.File(filename, 'r')

    # List all groups
    print("Keys: %s" % texts.keys())
    a_group_key = list(texts.keys())[0]

    # Get the data
    data = list(texts[a_group_key])
    print(len(data))

if __name__ == '__main__':
    main()
