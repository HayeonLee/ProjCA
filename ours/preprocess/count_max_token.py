import os
import json
import matplotlib
matplotlib.use('agg')
from pycocotools.coco import COCO
import nltk

jf = json.load(open('/DATA/cvpr19/coco/annotations/captions_train2017.json'))
jf_val = json.load(open('/DATA/cvpr19/coco/annotations/captions_val2017.json'))

cnt = 0
for i in range(len(jf['annotations'])):
    tokens = nltk.tokenize.word_tokenize(str(jf['annotations'][i]['caption']).lower().decode('utf-8'))
    cnt = max(cnt, len(tokens))

for i in range(len(jf_val['annotations'])):
    tokens = nltk.tokenize.word_tokenize(str(jf_val['annotations'][i]['caption']).lower().decode('utf-8'))
    cnt = max(cnt, len(tokens))

print(cnt)