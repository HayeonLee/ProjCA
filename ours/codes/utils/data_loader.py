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
import numpy as np
import json

'''
1. MS-COCO retrieval task
    - Datasets are two type: normal, rVal
        [normal] train: 83k, val: 40k (2014)
        [rVal] train: 113k, val: 5k (2017)
    - Dataloader should load (image, corresponding 5 captions)
2. MS-COCO visual grounding of phrases
3. our DB retrieval task
4. our DB Visual grounding of phrases
'''
''' issue '''
# speed

class MSCOCO(data.Dataset):

        def __init__(self, config, transform):
            self.coco_dir = config.coco_dir
            self.mode = config.mode
            self.coco = COCO(pth(self.coco_dir,'annotations','instances_{}{}.json'.format(self.mode, config.split_mode)))
            self.caps = COCO(pth(self.coco_dir,'annotations','captions_{}{}.json'.format(self.mode, config.split_mode)))
            self.img_ids = self.coco.getImgIds()
            self.img_dir = pth(self.coco_dir,'{}{}'.format(self.mode, config.split_mode))
            self.transform = transform
            with open(pth(self.coco_dir, 'annotations', 'vocabs.json'), 'r') as fd:
                vocab_file = json.load(fd)
                self.token_to_index = vocab_file['tokens']
                self.max_token_len = vocab_file['max_len']

        def __getitem__(self, index):
            img_info = self.coco.loadImgs(self.img_ids[index])[0]
            img = Image.open(pth(self.img_dir, img_info['file_name']))
            # img = Image.open(pth(self.img_dir, img_info['file_name'])).convert('RGB')

            ann_ids = self.caps.getAnnIds(imgIds=img_info['id'])
            anns = self.caps.loadAnns(ann_ids)
            idx = random.randint(0, len(anns)-1) # random pick one of 5 text captions
            # cap = self._unicodeToAscii(anns[idx]['caption'])
            cap = anns[idx]['caption']
            tokens = self._tokenize_cap(cap)
            tokens = self._encode_token(tokens)

            return self.transform(img), tokens, cap

        def __len__(self):
            return len(self.img_ids)

        def _encode_token(self, tokens):
            vec = torch.zeros(self.max_token_len).long()
            for i, token in enumerate(tokens):
                index = self.token_to_index[token]
                vec[i] = index
            return vec

        def _tokenize_cap(self, cap):
            cap = cap.lower().strip('\n')
            return cap.split(' ')


def get_loader(config):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(config.image_size))
    transform.append(T.CenterCrop(config.crop_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
    transform = T.Compose(transform)

    dataset = MSCOCO(config, transform)
    print('mode:{} [train|val]'.format(config.mode))
    print('Dataset size: {}'.format(len(dataset)))

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=(config.mode=='train'),
                                  num_workers=config.num_workers)
    return data_loader

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


