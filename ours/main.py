import argparse
import os
import torch
from torch.backends import cudnn
from solver import Solver
from dataloader import get_loader
from utils import str2bool
from preprocess import Vocabulary


def main(config):
    cudnn.benchmark = False

    save_path = os.path.join(config.main_dir, 'beenburger', config.model_name)
    dirs = ['checkpoint', 'result']
    for d in dirs:
        if not os.path.exists(os.path.join(save_path, d)):
            os.makedirs(os.path.join(save_path, d))

    trainloader, valloader = get_loader(config)

    solver = Solver(trainloader, valloader, config)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'val':
        solver.valid_retrieval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories.
    parser.add_argument('--main_dir', type=str, default='/DATA/cvpr19', help='path of main directory')
    parser.add_argument('--model_name', type=str, default='coco1',help='model name')

    # Dataset.
    parser.add_argument('--data_name', type=str, default='coco', help='[coco|ours]')
    parser.add_argument('--coco_split', type=str, default='rval', help='MS COCO split type: [rval|2017], rval use rest of val2014 as train data')
    parser.add_argument('--ours_split', type=str, default='val', help='our data split type: [train|val]')
    parser.add_argument('--flip', type=str2bool, default='True')

    # Model configurations.
    parser.add_argument('--max_token_len', type=int, default=57, help='coco:57')
    parser.add_argument('--D', type=int, default='2048', help='dimension of image feature from ResNet')
    parser.add_argument('--D_prime', type=int, default='2400', help='dimension of adaptation + pooling')
    parser.add_argument('--d', type=int, default='2400', help='dimension of projection')
    parser.add_argument('--K', type=int, default='620', help='dimension of word2vec embedding')
    parser.add_argument('--rnn_num', type=int, default='4', help='the number of stack of rnn')
    parser.add_argument('--margin', type=float, default='0.2', help='margin for hard negative loss')
    parser.add_argument('--pt_path', type=str, default='/DATA/cvpr19/w2v', help='path of the pretrained vocabs("Skip-thought vectors"[NIPS2015])')

    # Training configurations.
    parser.add_argument('--mode', type=str, default='train', help='[train|val]')
    parser.add_argument('--batch_size',type=int, default=160, help='larger batch size show good convergence & performance')
    parser.add_argument('--num_workers',type=int, default=4, help='num workers')
    parser.add_argument('--img_size',type=int, default=256, help='image_size')
    parser.add_argument('--crop_size',type=int, default=224, help='crop_size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--init_ep', type=int, default=8, help='initial training for pi of txt encoder, theta2 of img encoder ')
    parser.add_argument('--max_ep', type=int, default=100, help='max epochs')

    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=100, help='how many steps/minibatches between printing out the loss')
    parser.add_argument('--draw_step', type=int, default=1, help='draw step')
    parser.add_argument('--acc_step', type=int, default=1, help='step to caculate accuracy of validation data')
    parser.add_argument('--save_step', type=int, default=100, help='step to save model')
    parser.add_argument('--use_visdom', type=str2bool, default='False')
    parser.add_argument('--init_from', type=int, default=0)


    config = parser.parse_args()
    print(config)
    main(config)

