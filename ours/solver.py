import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visdom import Visdom

from img_enc import ImageEncoder
from txt_enc import TextEncoder
from utils import print_network, create_vis, update_vis, get_vocab
from evaluation import i2t, t2i

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 7
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size

class Solver(object):
    def __init__(self, trainloader, valloader, config):
        """Initialize configurations."""
        # Data loader.
        self.trainloader = trainloader
        self.validloader = valloader

        # Directories.
        self.main_dir = config.main_dir
        self.model_name = config.model_name

        # Dataset.
        self.data_name = config.data_name

        # Model configurations.
        self.vocab = get_vocab(self.main_dir, self.data_name)
        self.D = config.D
        self.D_prime = config.D_prime
        self.d = config.d
        self.K = config.K
        self.rnn_num = config.rnn_num
        self.margin = config.margin
        self.pt_path = config.pt_path

        # Training configurations.
        self.mode = config.mode
        self.batch_size = config.batch_size
        self.img_size = config.img_size
        self.crop_size = config.crop_size
        self.lr = config.lr # 0.001
        self.lr_decay = config.lr_decay # 0.98
        self.init_ep = config.init_ep
        self.max_ep = config.max_ep

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_step = config.log_step
        self.draw_step = config.draw_step
        self.acc_step = config.acc_step
        self.save_step = config.save_step
        self.use_visdom = config.use_visdom
        self.init_from = config.init_from
        self.best_r = np.array([0, 0, 0])
        self.best_ri = np.array([0, 0, 0])

        # Build model.
        self.build_model()

        if self.use_visdom:
            self.viz = Visdom()
            self.loss_plot = create_vis(self.viz, self.model_name, 'loss', self.max_ep, 5)
            self.acc_plot_i2t = create_vis(self.viz, self.model_name, 'accuracy', self.max_ep, 100)
            self.acc_plot_t2i = create_vis(self.viz, self.model_name, 'accuracy', self.max_ep, 100)

    def build_model(self):

        print('     Create model...')
        self.img_enc = ImageEncoder(input_size=self.crop_size, D=self.D, D_prime=self.D_prime)
        self.txt_enc = TextEncoder(pretrained_path=self.pt_path, vocab=self.vocab, K=self.K, d=self.d, num_stack=self.rnn_num)

        print('     Create optimizer...')
        params_list = list(self.img_enc.parameters()) + list(self.txt_enc.parameters())
        self.optimizer = optim.Adam(params_list, lr=self.lr)

        # print_network(self.img_enc, 'image encoder')
        print_network(self.txt_enc, 'text encoder')

        # GPU mode.
        print('     Change to GPU mode...')
        self.img_enc.to(self.device)
        self.txt_enc.to(self.device)

    def update_lr(self):
        self.lr = self.lr * self.lr_decay #decay it
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        print('Decayed learning rate by a factor {} to {}'.format(self.lr_decay, self.lr))

    # def hard_negative_loss_old(self, y, z):
    #     #max(loss(y, z, z')) = max(max{0, margin - <y,z> + <y, z'>})
    #     batch_size = y.size(0)
    #     total_hn = 0
    #     cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #     for n in range(batch_size):
    #         pos = cos(y[n], z[n])
    #         hard_negtive = 0
    #         for n_prime in range(batch_size):
    #             if n_prime is not n:
    #                 neg = cos(y[n], z[n_prime])
    #                 loss = max(0, self.margin - pos + neg)
    #                 hard_negtive = max(loss, hard_negtive)
    #         total_hn += hard_negtive
    #     total_hn /= batch_size
    #     return total_hn

    def hard_negative_loss(self, y, z):
        # same thing of VSE++
        # max(loss(y, z, z')) = max(max{0, margin - <y,z> + <y, z'>})
        batch_size = y.size(0)
        total_hn = 0
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        for n in range(batch_size):
            yn = y[n].repeat(batch_size, 1) #yn size: (160, 2400)
            loss = cos_sim(yn, z) #z size: (160, 2400), loss size: (160,)
            pos = loss[n]
            loss = self.margin - pos + loss

            hns, idx = torch.topk(loss, 2)

            if idx[0] == n:
                total_hn += max(0, hns[1])
            else:
                total_hn += max(0, hns[0])
        total_hn /= batch_size
        return total_hn

    def train_init(self, start_time):
        print("     Initial training for pi of txt encoder, theta2 of img encoder...")
        for param in self.img_enc.parameters():
            param.requires_grad = False
        self.img_enc.proj.weight.requires_grad = True
        self.img_enc.proj.bias.requires_grad = True

        data_iter = iter(self.trainloader)
        iters = len(data_iter)
        mean_loss = 0

        for i in range(self.init_ep):
            for j in range(iters):
                try:
                    img, tokens, _ = next(data_iter)
                except:
                    data_iter = iter(self.trainloader)
                    img, tokens, _ = next(data_iter)

                tokens = tokens.to(self.device)
                img = img.to(self.device)

                embed_txt = self.txt_enc(tokens)
                embed_img = self.img_enc(img)

                ## Compute loss.
                loss = self.hard_negative_loss(embed_img, embed_txt) + self.hard_negative_loss(embed_txt, embed_img)

                ## Reset the gradient buffers.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mean_loss += loss
                ## Print out training information.
                if (j+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch=[{}/{}], Iter=[{}/{}], Loss=[{:4.5f}], lr=[{:4.5f}]".format(et, int(i+1), self.init_ep, j, iters, mean_loss/self.log_step, self.lr)
                    print(log)
                    mean_loss = 0
            self.update_lr()


    def train(self):

        start_time = time.time()
        self.train_init(start_time)

        print('Start training for all models...')
        for param in self.img_enc.parameters():
            param.requires_grad = True

        data_iter = iter(self.trainloader)
        iters = len(data_iter)
        mean_loss = 0

        for i in range(self.init_ep, self.max_ep):
            for j in range(iters):
                try:
                    img, tokens, _ = next(data_iter)
                except:
                    data_iter = iter(self.trainloader)
                    img, tokens, _ = next(data_iter)

                tokens = tokens.to(self.device)
                img = img.to(self.device)

                embed_txt = self.txt_enc(tokens)
                embed_img = self.img_enc(img)

                ## Compute loss.
                loss = self.hard_negative_loss(embed_img, embed_txt) + self.hard_negative_loss(embed_txt, embed_img)

                ## Reset the gradient buffers.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mean_loss += loss

                ## Print out training information.
                if (j+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch=[{}/{}], Iter=[{}/{}], Loss=[{:4.5f}], lr=[{:4.5f}]".format(et, int(i+1), self.max_ep, j, iters, mean_loss/self.log_step, self.lr)
                    print(log)
                    mean_loss = 0

            if (i+1) % self.acc_step == 0:
                r, ri, val_loss = self.valid_retrieval()
                self.best_r = np.maximum(r, self.best_r)
                self.best_ri = np.maximum(ri, self.best_ri)
                print('[i2t Best] R@1:{:.1f}, R@5:{:.1f}, R@10:{:.1f}'.format(self.best_r[0], self.best_r[1], self.best_r[2]))
                print('[t2i Best] R@1:{:.1f}, R@5:{:.1f}, R@10:{:.1f}'.format(self.best_ri[0], self.best_ri[1], self.best_ri[2]))
                self.txt_enc.train()
                self.img_enc.train()

            if self.use_visdom and (i+1) % self.draw_step == 0:
                update_vis(self.viz, mean_loss, i+1, self.loss_plot, 'train loss')
                update_vis(self.viz, val_loss, i+1, self.loss_plot, 'valid loss')
                for m in range(2):
                    update_vis(self.viz, r[m], i+1, self.acc_plot_i2t, m)
                    update_vis(self.viz, ri[m], i+1, self.acc_plot_t2i, m)

            if (i+1) % self.save_step == 0:
                self.save_model(i, self.main_dir, self.model_name, self.txt_enc, self.img_enc)

    def valid_retrieval(self):
        '''The task is performed 5 times on 1000-image subsets of the test set and the results are averaged.'''
        '''Our best results are obtained with a different strategy:
        Images are resized to 400x400 irrespective of their size and aspect ratio'''
        mean_loss = 0
        embed_txts = []
        embed_imgs = []

        if self.init_from or (self.mode == 'val'):
            self.txt_enc, self.img_enc = self.restore_model(self.init_from, self.main_dir, self.model_name)

        self.txt_enc.eval()
        self.img_enc.eval()

        data_iter = iter(self.validloader)
        iters = len(data_iter)
        mean_loss = 0

        for j in range(iters):
            img, tokens, _ = next(data_iter)

            tokens = tokens.to(self.device)
            img = img.to(self.device)

            embed_txt = self.txt_enc(tokens)
            embed_img = self.img_enc(img)#.type(torch.cuda.DoubleTensor)

            ## Compute loss.
            loss = self.hard_negative_loss(embed_img, embed_txt) + self.hard_negative_loss(embed_txt, embed_img)

            mean_loss += loss
            embed_txts.extend(embed_txt)
            embed_imgs.extend(embed_img)

        mean_loss /= iters
        r = np.zeros(4)
        ri = np.zeros(4)

        for i in range(5):
            r += i2t(embed_imgs[1000*i:1000(i+1)], embed_txts[1000*i:1000(i+1)])
            ri += t2i(embed_imgs[1000*i:1000(i+1)], embed_txts[1000*i:1000(i+1)])

        r /= 5
        ri /= 5

        print("Image to text: %.1f, %.1f, %.1f, %.1f" % (r[0],r[1],r[2],r[3]))
        print("Text to image: %.1f, %.1f, %.1f, %.1f" % (ri[0],ri[1],ri[2],ri[3]))

        return r, ri, mean_loss

            # ## Reset the gradient buffers.
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()


        # with torch.no_grad(): #hayeon edit
        #     data_iter = iter(self.data_loader)
        #     #self.build_model()
        #     self.restore_model()
        #     print('Start test...')
        #     acc = 0.0
        #     total = 0.0
        #     for i in range(len(data_iter)):
        #         txt, img = next(data_iter) #hayeon edit
        #         labels = torch.arange(img.size(0), dtype=torch.long)

        #         txt = txt.to(self.device)
        #         img = img.to(self.device)

        #         fea_txt = self.txt_enc(txt).type(torch.cuda.DoubleTensor)
        #         fea_img = self.img_enc(img).type(torch.cuda.DoubleTensor)
        #         num_class = fea_txt.size(0)
        #         pair_size = fea_img.size(0)
        #         total += num_class
        #         score = torch.zeros(pair_size, num_class)
        #         for i in range(pair_size):
        #             for j in range(num_class):
        #                 score[i,j] = torch.dot(fea_img[i], fea_txt[j])
        #             label_score = score[i, labels[i]]

        #         max_socre, max_ix = torch.max(score, dim=1)
        #         acc += sum(max_ix == labels).float()
        #     print("acc: {:.4f}".format(acc.item()/total)) #hayeon edit
            #loss
                # self.viz.line(Y=torch.Tensor([train_acc]),
                #               X=torch.Tensor([i+1]),
                #               win=self.acc_plot,
                #               name='train',
                #               update='append',)
                # self.viz.line(Y=torch.Tensor([valid_acc]),
                #               X=torch.Tensor([i+1]),
                #               win=self.acc_plot,
                #               name='test',
                #               update='append',)
                # self.viz.line(Y=torch.Tensor([mean_loss]),
                #               X=torch.Tensor([i+1]),
                #               win=self.loss_plot,
                #               name='train loss',
                #               update='append',)
                # self.viz.line(Y=torch.Tensor([valid_loss]),
                #               X=torch.Tensor([i+1]),
                #               win=self.loss_plot,
                #               name='valid loss',
                #               update='append',)

            # self.acc_plot = self.viz.line(Y=torch.Tensor([0.]),
            #                               X=torch.Tensor([0.]),
            #                               opts = dict(title = 'Test accuracy for ' + self.model_name,
            #                                           xlabel= 'epoch',
            #                                           xtickmin=0,
            #                                           xtickmax=self.max_epoch,
            #                                           ylabel='Accuracy',
            #                                           ytickmin=0,
            #                                           ytickmax=100,),)

    # def cal_scores(self, fea_txt, fea_imgs):
    #     scores = []
    #     for i in range(len(fea_imgs)):
    #         scores.append([torch.dot(fea_txt, fea_imgs[i]), i]) # [score, index]
    #     scores = sorted(scores, reverse=True)
    #     # return index corresponding to large scores in the descending order
    #     # index = torch.Tensor(scores)[:10, 1].type(torch.int)
    #     index = np.array(scores, dtype=np.int)[:10, 1]
    #     index = [index[i] for i in range(10)]
    #     scores = torch.Tensor(scores)[:10, 0]
    #     return scores, index

    # def cal_errors(self, scores, true_class, mAP_k=[1, 5, 10]):
    #     acc = torch.zeros(3)
    #     if true_class in scores[:mAP_k[0]]:
    #         acc[0] = 1
    #     if true_class in scores[:mAP_k[1]]:
    #         acc[1] = 1
    #     if true_class in scores[:mAP_k[2]]:
    #         acc[2] = 1
    #     return acc

    # def make_grid(self, img_path, img_dirs, i, index):
    #     fname = os.listdir(os.path.join(img_path, img_dirs[i].strip()))[0]
    #     img = np.array(Image.open(os.path.join(img_path, img_dirs[i].strip(), fname)))
    #     img_size = np.shape(img)[:2][::-1]
    #     for i in index[:5]:
    #         fname = os.listdir(os.path.join(img_path, img_dirs[i].strip()))[0]
    #         img2 = Image.open(os.path.join(img_path, img_dirs[i].strip(), fname))
    #         img2 = np.array(img2.resize(img_size))
    #         img = np.concatenate((img, img2), axis=1)
    #     return img

    # def imshow(self, img_grid, ith):
    #     plt.clf()
    #     fig, ax = plt.subplots(5, 1)

    #     for i, row_img in enumerate(img_grid[ith:ith+5]):
    #         ax[i].imshow(row_img)
    #         ax[i].axis('off')
    #         ax[i].set_title('GT, top1, top2, top3, top4, top5')
    #     plt.savefig('/home/cvpr19/scottreed/cvpr2016/results/{}.png'.format(ith))
    #     plt.close()
    #     print('save figures')

    # def img_retrieval(self):
    #     '''
    #     1. Make image feature list and text feature list
    #     2. For each text feature, rank image features
    #        by computing scores between the text feature and image features
    #     3. Calculate MAP at top k errors while changing k values
    #     4. Make image grids with top 5 errors
    #        (text descriptions;
    #         1 col: original images, 2~6 cols: retrievaled images)
    #     caution. fix to the first image and text for each class
    #     '''
    #     with torch.no_grad():
    #         data_iter = iter(self.data_loader)
    #         self.restore_model()
    #         print('Start image retrieval...')
    #         fea_txts = []
    #         fea_imgs = []
    #         mAP = torch.zeros(3)
    #         mAP_k = [1, 5, 10]
    #         f = open('/home/cvpr19/scottreed/DATA/CUB/valclasses.txt')
    #         cls_name_list = f.readlines()
    #         img_path = '/home/cvpr19/scottreed/DATA/CUB_200_2011/images'
    #         # print(cls_name_list)
    #         img_grid = []
    #         for i in range(len(data_iter)):
    #             txt, img = next(data_iter)
    #             txt = txt.to(self.device)
    #             img = img.to(self.device)

    #             fea_txt = self.txt_enc(txt).type(torch.cuda.DoubleTensor)
    #             fea_img = self.img_enc(img).type(torch.cuda.DoubleTensor)
    #             #1. Make image feature list and text feature list
    #             fea_txts += [fea_txt[i] for i in range(fea_txt.size(0))]
    #             fea_imgs += [fea_img[i] for i in range(fea_img.size(0))]
    #         true_classes = range(len(fea_txts))
    #         for i in range(len(fea_txts)):
    #             _, index = self.cal_scores(fea_txts[i], fea_imgs)
    #             acc = self.cal_errors(index, true_classes[i])
    #             mAP += acc
    #             img_grid.append(self.make_grid(img_path, cls_name_list, i, index))
    #         for i in range(10):
    #             self.imshow(img_grid, i)
    #         mAP = mAP / len(fea_txts)
    #         for i in range(mAP.size(0)):
    #             print('mAP@{}:{:.2f}'.format(mAP_k[i], mAP[i]))