import os
import json
import matplotlib
matplotlib.use('agg')
from pycocotools.coco import COCO
import nltk

def struct_val(jf, i):
    tmp = []
    for j in range(5):
        data = {'sentid':[jf[i]['sentids'][j]],
                'imgid':jf[i]['cocoid'],
                'filepath':jf[i]['filepath'],
                'filename':jf[i]['filename'],}
        tmp.append(data)
    return tmp

def struct_train(jf, i):
    tmp = []
    data = {'sentid':jf[i]['sentids'],
            'imgid':jf[i]['cocoid'],
            'filepath':jf[i]['filepath'],
            'filename':jf[i]['filename'],}
    tmp.append(data)
    return tmp
# save data list centered sentids
main_path = '/DATA/cvpr19/coco/annotations'
# 1. make rval data list
# 2. make rval(without restval) data list
train = []
trainrestval = []
valid = []
test = []

jf = json.load(open(os.path.join(main_path, 'dataset.json'), 'r'))['images']
for i in range(len(jf)):
    if jf[i]['split'] in ['train']:
        train += struct_train(jf, i)
        trainrestval += struct_train(jf, i)
    elif jf[i]['split'] in ['restval']:
        trainrestval += struct_train(jf, i)
    elif jf[i]['split'] in ['val']:
        valid += struct_val(jf, i)
    elif jf[i]['split'] in ['test']:
        test += struct_val(jf, i)
    if i % 3000 == 0:
        print('{}th data are processed'.format(i))

with open(os.path.join(main_path, 'rval_train.json'), 'w') as outfile:
    json.dump(trainrestval, outfile)
with open(os.path.join(main_path, 'rval_val.json'), 'w') as outfile:
    json.dump(valid, outfile)
with open(os.path.join(main_path, 'rval_test.json'), 'w') as outfile:
    json.dump(test, outfile)
with open(os.path.join(main_path, 'rval_train_only.json'), 'w') as outfile:
    json.dump(test, outfile)
print('1,2 are finished')

# 3. make 2017 data list
train = []
valid = []
jf = json.load(open(os.path.join(main_path, 'captions_train2017.json')))['images']
caps = COCO(os.path.join(main_path, 'captions_train2017.json'))

for i in range(len(jf)):
    ann_ids = caps.getAnnIds(imgIds=jf[i]['id'])
    data = {'sentid':ann_ids,
        'imgid':jf[i]['id'],
        'filepath':'train2017',
        'filename':jf[i]['file_name'],}
    train.append(data)
    if i % 3000 == 0:
        print('train2017-{}th data are processed'.format(i))

jf = json.load(open(os.path.join(main_path, 'captions_val2017.json')))['images']
caps = COCO(os.path.join(main_path, 'captions_val2017.json'))

for i in range(len(jf)):
    ann_ids = caps.getAnnIds(imgIds=jf[i]['id'])
    for j in range(5):
        data = {'sentid':[ann_ids[j]],
            'imgid':jf[i]['id'],
            'filepath':'val2017',
            'filename':jf[i]['file_name'],}
        valid.append(data)
    if i % 3000 == 0:
        print('val2017-{}th data are processed'.format(i))

with open(os.path.join(main_path, 'train2017.json'), 'w') as outfile:
    json.dump(train, outfile)
with open(os.path.join(main_path, 'val2017.json'), 'w') as outfile:
    json.dump(valid, outfile)
print('3 is finished')








