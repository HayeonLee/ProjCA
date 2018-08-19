#refer code: https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
import json
from collections import Counter
import itertools
import matplotlib
matplotlib.use('agg')
from pycocotools.coco import COCO
from os.path import join as pth
import unicodedata
import string
import os

''' issue '''
# unicode to ascii?
# remove [",.,?,<,>,(,),{,},,,]?
# what is normalization?

def tokenize_corpus(anns, anns_val):
    """ Tokenize and normalize annotations from a given mscoco json. """
    caps = [ann['caption'].lower().strip('\n').replace('.','').replace(',','') for ann in anns]
    caps_val = [ann_val['caption'].lower().strip('\n').replace('.','').replace(',','') for ann_val in anns_val]
    caps = caps + caps_val
    tokens = [ cap.split(' ') for cap in caps ]
    return tokens

def extract_vocab(tokenized_corpus):
    vocabs = []
    max_tokens = 0
    for sentence in tokenized_corpus:
        if len(sentence) > max_tokens:
            max_tokens = len(sentence)
        for token in sentence:
            if (token not in vocabs) and (token is not ''):
                vocabs.append(token)
    return vocabs, max_tokens

def main():
    coco_dir = '/DATA/cvpr19/coco'
    caps = COCO(pth(coco_dir,'annotations','captions_train2014.json'))
    caps_val = COCO(pth(coco_dir,'annotations','captions_val2014.json'))

    ann_ids = caps.getAnnIds()
    ann_ids_val = caps_val.getAnnIds()

    anns = caps.loadAnns(ann_ids)
    anns_val = caps_val.loadAnns(ann_ids_val)

    tokenized_corpus = tokenize_corpus(anns, anns_val)
    vocabs, max_len = extract_vocab(tokenized_corpus)

    word2idx = {w: idx for (idx, w) in enumerate(vocabs)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabs)}
    vocabs_size = len(vocabs)

    vocab = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocabs_size': vocabs_size,
        'max_tokens_len': max_len
    }

    with open(os.path.join(coco_dir,'annotations','vocab.json'), 'w') as fd:
        json.dump(vocab, fd)

if __name__ == '__main__':
    main()
