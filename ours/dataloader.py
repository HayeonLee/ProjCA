#refer code: https://github.com/fartashf/vsepp/blob/master/data.py
import argparse
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms as T

from PIL import Image
import matplotlib
matplotlib.use('agg')
from pycocotools.coco import COCO
from os.path import join as pth
import random
import json
import pickle
import nltk
from preprocess import Vocabulary
from utils import get_vocab

'''
1. MS-COCO retrieval task
    - Datasets are two type: normal, rVal
        [rVal] train: 83k(113k), val: 5k (2014)
        [normal] train: 113k, val: 5k (2017)
    - Dataloader should load (image, corresponding 5 captions)
2. MS-COCO visual grounding of phrases
3. our DB retrieval task
4. our DB Visual grounding of phrases

# train data json and valid data json have different structure since 5 captions
'''

class MSCOCO(data.Dataset):

        def __init__(self, config, transform, mode):
            self.main_dir = config.main_dir
            self.data_name = config.data_name
            self.mode = mode
            self.max_token_len = config.max_token_len
            self.split = self.get_split_type(config)
            self.data = self.get_data_list()
            self.vocab = get_vocab(self.main_dir, self.data_name)
            self.transform = transform
            ''' debugging '''
            self.coco_split = config.coco_split
            tmp = pth(self.main_dir, self.data_name, 'annotations','captions_{}{}.json'.format(self.mode, '2014' if self.coco_split is 'rval' else '2017'))
            print('caption path: {}'.format(tmp))
            ''''''
            self.caps = self.get_coco()

        def __getitem__(self, index):
            info = self.data[index]
            img = Image.open(pth(self.main_dir, self.data_name, info['filepath'], info['filename'])).convert('RGB')
            if info['filepath'] in ['train2017', 'train2014']:
                caps = self.caps['train']
            else:
                caps = self.caps['val']
            anns = caps.loadAnns(info['sentid'])

            img_id = info['imgid']
            # random pick one of 5 text captions
            idx = random.randint(0, len(anns)-1) if self.mode is 'train' else 0
            cap = anns[idx]['caption']
            vec = self.encode_token(cap)
            # tokens = nltk.tokenize.word_tokenize(str(cap).lower().decode('utf-8'))
            # cap = []
            # # cap.append(self.vocab('<start>'))
            # cap.extend([self.vocab(token) for token in tokens])
            # cap.append(self.vocab('<end>'))
            return self.transform(img), vec, img_id

        def __len__(self):
            return len(self.data)

        def encode_token(self, caption):
            vec = torch.zeros(self.max_token_len).long()
            tokens = nltk.tokenize.word_tokenize(str(caption).lower().decode('utf-8'))
            for i, token in enumerate(tokens):
                vec[i] = self.vocab(token)
            return vec
        #
        def get_split_type(self, config):
            if self.data_name == 'coco':
                split = config.coco_split
            elif self.data_name == 'ours':
                split = config.ours_split
            return split

        def get_data_list(self):
            anndir = pth(self.main_dir, self.data_name, 'annotations')
            if self.split == 'rval': #use restval data as train data
                ''' debugging print '''
                tmp = pth(anndir, 'rval_{}.json'.format(self.mode))
                print('datalist path: {}'.format(tmp))
                ''''''
                datalist = json.load(open(pth(anndir, 'rval_{}.json'.format(self.mode))))
            elif self.split == '2017':
                datalist = json.load(open(pth(anndir, '{}2017.json'.format(self.mode))))
                ''' debugging print '''
                tmp = pth(anndir, '{}2017.json'.format(self.mode))
                print('datalist path: {}'.format(tmp))
                ''''''
            return datalist

        def get_coco(self):
            year = '2014' if self.coco_split is 'rval' else '2017'
            caps = {}
            for mode in ['train', 'val']:
                caps[mode] = COCO(pth(self.main_dir, self.data_name, 'annotations','captions_{}{}.json'.format(mode, year)))
            # caps = COCO(pth(self.main_dir, self.data_name, 'annotations','captions_{}{}.json'.format('train', year)))
            # caps_val = COCO(pth(self.main_dir, self.data_name, 'annotations','captions_{}{}.json'.format('val', year)))
            return caps


def get_transform(mode, crop_size, image_size, flip=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if mode == 'train':
        t_list.append(T.RandomResizedCrop(crop_size))
        if flip:
                t_list.append(T.RandomHorizontalFlip())
    elif mode in ['val', 'test']:
        t_list.append(T.Resize(image_size))
        t_list.append(T.CenterCrop(crop_size))
    t_list.append(T.ToTensor())
    t_list.append(normalizer)
    transform = T.Compose(t_list)
    return transform

def get_loader(config):
    """Build and return a data loader."""
    print('mode:{} [train|val]'.format(config.mode))
    # print('Dataset size: {}'.format(len(dataset)))

    if config.mode == 'train':
        train_dataset = MSCOCO(config, get_transform(config.mode, config.crop_size, config.img_size, config.flip), config.mode)
        trainloader = data.DataLoader(dataset=train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=config.num_workers)
    else :
        trainloader = None

    valid_dataset = MSCOCO(config, get_transform('val', config.crop_size, config.img_size), 'val')
    validloader = data.DataLoader(dataset=valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    return trainloader, validloader

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    # COCO Dataset
    parser.add_argument('--split_mode', type=str, default='2014', help='MS COCO split type: [normal:2014|rval:2017]')
    parser.add_argument('--coco_dir', type=str, default='/DATA/cvpr19/coco', help='coco data directory.')
    parser.add_argument('--mode', type=str, default='val', help='[train|val]')
    parser.add_argument('--batch_size',type=int, default=20, help='number of sequences to train on in parallel')
    parser.add_argument('--num_workers',type=int, default=4, help='num workers')
    parser.add_argument('--image_size',type=int, default=256, help='image_size')
    parser.add_argument('--crop_size',type=int, default=224, help='crop_size')

    config = parser.parse_args()

    loader = get_loader(config)
    print('Batch size: {}'.format(config.batch_size))
    print('The number of batches: {}'.format(len(loader)))

    data_iter = iter(loader)
    img, tokens, cap = next(data_iter)
    print('Size of img: {}'.format(img.size()))
    import numpy as np
    print('Size of txt: {}'.format(np.shape(np.array(cap))))
    print(cap)
    for ann in tokens:
        print(ann)

            # self.coco = COCO(pth(self.coco_dir,'annotations','instances_{}{}.json'.format(self.mode, self.year)))
            # self.caps = COCO(pth(self.coco_dir,'annotations','captions_{}{}.json'.format(self.mode, self.year)))
            # self.img_ids = self.coco.getImgIds()
            # self.img_dir = pth(self.coco_dir,'{}{}'.format(self.mode, self.year))
            # with open(pth(self.coco_dir, 'annotations', 'vocabs.json'), 'r') as fd:
            #     vocab_file = json.load(fd)
            #     self.token_to_index = vocab_file['tokens']
            #     self.max_token_len = vocab_file['max_len']

            # tokens = self._tokenize_cap(cap)
            # tokens = self._encode_token(tokens)

            # img_info = self.coco.loadImgs(self.img_ids[index])[0]
            # img = Image.open(pth(self.img_dir, img_info['file_name']))
            # img = Image.open(pth(self.img_dir, img_info['file_name'])).convert('RGB')
            # ann_ids = self.caps.getAnnIds(imgIds=img_info['id'])
            # cap = self._unicodeToAscii(anns[idx]['caption'])

            # elif self.split == 'train_only' and self.mode == 'train':
            #     datalist = json.load(open(pth(anndir, 'rval_train_only.json')))
            # elif self.split == 'train_only' and self.mode == 'val':
            #     datalist = json.load(open(pth(anndir, 'rval_val.json')))

        # def _encode_token(self, tokens):
        #     vec = torch.zeros(self.max_token_len).long()
        #     for i, token in enumerate(tokens):
        #         index = self.token_to_index[token]
        #         vec[i] = index
        #     return vec

        # def _tokenize_cap(self, cap):
        #     cap = cap.lower().strip('\n')
        #     return cap.split(' ')
        #

    # transform = []
    # transform.append(T.Resize(config.image_size))
    # transform.append(T.CenterCrop(config.crop_size))
    # transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
    # transform = T.Compose(transform)
