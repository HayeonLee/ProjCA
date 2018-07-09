import argparse
import torch
from torch.utils.serialization import load_lua

parser = argparse.ArgumentParser(description='Test file for reading t7 image using PyTorch')
parser.add_argument('--target_image', '-t', help='target image to be read')

def main():
    global args
    args = parser.parse_args()
    filename = args.target_image
    image = load_lua(filename)

    print (type(image))
    print (image.size())
    print (image)
if __name__ == '__main__':
    main()
