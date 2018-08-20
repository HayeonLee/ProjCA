import os
import torch
import pickle
from preprocess import Vocabulary

def str2bool(v):
    return v.lower() in ('true')

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))

def get_vocab(main_dir, data_name):
    vocab = pickle.load(open(os.path.join(main_dir, data_name, 'annotations', '{}_vocab.pkl'.format(data_name)), 'rb'))
    return vocab

def create_vis(viz, model, ylabel, xtickmax, ytickmax):
    plot = viz.line(Y=torch.Tensor([0.]),
                    X=torch.Tensor([0.]),
                    opts = dict(title = '{} for {}'.format(ylabel, model),
                                xlabel='epoch',
                                xtickmin=0,
                                xtickmax=xtickmax,
                                ylabel=ylabel,
                                ytickmin=0,
                                ytickmax=ytickmax,
                                ),)
    return plot

def update_vis(viz, y_val, x_val, plot, name):
    updated_viz = viz.line(Y=torch.Tensor([y_val]),
                              X=torch.Tensor([x_val]),
                              win=plot,
                              name=name,
                              update='append',)
    return updated_viz

def restore_model(init_from, load_pth, model_name):
    main_path = os.path.join(save_path, model_name, 'checkpoints')
    txt_enc_path = os.path.join(main_path, '{0:d}-txt_enc.ckpt'.format(i+1))
    img_enc_path = os.path.join(main_path, '{0:d}-img_enc.ckpt'.format(i+1))
    txt_enc.load_state_dict(torch.load(txt_enc_path, map_location=lambda storage, loc: storage))
    img_enc.load_state_dict(torch.load(enc_img_path, map_location=lambda storage, loc: storage))
    print('Loading the trained models from the checkpoint: {}'.format(main_path))
    return txt_enc, img_enc

def save_model(i, save_path, model_name, txt_enc, img_enc):
    main_path = os.path.join(save_path, model_name, 'checkpoints')
    txt_enc_path = os.path.join(main_path, '{0:d}-txt_enc.ckpt'.format(i+1))
    img_enc_path = os.path.join(main_path, '{0:d}-img_enc.ckpt'.format(i+1))
    torch.save(txt_enc.state_dict(), txt_enc_path)
    torch.save(img_enc.state_dict(), img_enc_path)
    print('Saved model checkpoints into {}...'.format(main_path))



