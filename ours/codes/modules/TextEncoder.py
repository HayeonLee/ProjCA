# refer code: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
# refer code2: https://github.com/Cadene/skip-thoughts.torch.git
import torch
import torch.nn as nn
import os
import numpy as np
from sru import SRU

class TextEncoder(nn.Module):
    def __init__(self, w2v_path, vocabs, num_embed, K=620, d=2400, num_stack=4):
        super(self).__init__()
        self.embedding = self.create_emb_layer(self.load_dicts(w2v_path),
                                               self.load_emb_params(w2v_path),
                                               vocabs,
                                               num_embed,
                                               K)
        self.input_size = K
        self.hidden_size = d
        self.num_layers = num_stack
        self.sru = SRU(self.input_size, self.hidden_size,
                       num_layers = self.num_layers,          # number of stacking RNN layers
                       rnn_dropout = 0.25,      # variational dropout applied on linear transformation
                       use_tanh = 1,            # use tanh?
                       use_relu = 0,            # use ReLU?
                       use_selu = 0,            # use SeLU?
                       bidirectional = False,   # bidirectional RNN ?
                       weight_norm = False,     # apply weight normalization on parameters
                       layer_norm = False,      # apply layer normalization on the output of each layer
                       highway_bias = 0         # initial bias of highway gate (<= 0)
                       )

    def forward(self, inp):
        hidden = self.init_hidden(inp.size(0))
        output = self.sru(self.embedding(inp), hidden)
        output = output / torch.norm(output, 2, dim=2)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def load_dicts(self, w2v_path):
        path = os.path.join(w2v_path, 'dictionary.txt')
        with open(path_dict, 'r') as f:
            dicts = {w: idx for (idx, w) in enumerate(f)} # .strip()?
        return dicts

    def load_emb_params(self, w2v_path):
        path = os.path.join(w2v_path, 'utable.npy')
        params = np.load(path, encoding='latin1') # to load from python2
        return torch.from_numpy(params)

    def create_emb_layer(self, dicts, params, vocabs, num_embed, K, non_train=False):
        #word2vec: "Skip-thought vectors" [NIPS2015]
        #https://github.com/ryankiros/skip-thoughts
        weights_matrix = torch.zeros((num_embed, K)) # first dim = zeros -> +1
        unknown_params = params[dicts['UNK']]
        unknown = 0

        for word in vocabs.keys():
            try:
                weights_matrix[word2idx[word]] = params[dicts[word]]
            except KeyError:
                weights_matrix[word2idx[word]] = unknown_params
                unknown += 1

        emb_layer = nn.Embedding(num_embed, K)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_train:
            emb_layer.weight.requires_grad = False
        if unknown > 0:
            print('Warning: {}/{} words are not in dictionary, thus set UNK'
                  .format(unknown, len(vocabs.keys())))
        # if self.save:
        #     torch.save(self.embedding, path)
        return emb_layer

#  class TextEncoder(nn.Module):
#      def __init__(self, num_embed, K, hidden_size):
#          super(TextEncoder, self).__init__()
#          self.embed = nn.Embedding(num_embed=num_embed, K=K)
#          self.bilstm = nn.LSTM(input_size=K, hidden_size=hidden_size,
#                                  batch_first=False,
#                                  bidirectional=True,
#                                  dropout=0.5)

#          self.dropout = nn.Dropout(p=0.5)

#      def forward(self, x):
#          embed_output = self.embed(x)
#          bilstm_output, _ = self.bilstm(self.dropout(embed_output))
#          return bilstm_output

#      def load_pretrained(self, dictionary):
#          print("Loading pretrained weights...")
#          # Load pretrained vectors for embedding layer
#          glove = vocab.GloVe(name='6B', dim=self.embed.K)

#          # Build weight matrix here
#          pretrained_weight = self.embed.weight.data


# class rDAN(nn.Module):
#     def __init__(self, num_embed, K, hidden_size, answer_size, k=2):
#         super(rDAN, self).__init__()
#         # Build Text Encoder
#         self.textencoder = TextEncoder(num_embed=num_embed,
#                                        K=K,
#                                         hidden_size=hidden_size)

# def _make_emb_state_dict(self, dictionary, parameters):
#     weight = torch.zeros(len(self.vocab)+1, 620) # first dim = zeros -> +1
#     unknown_params = parameters[dictionary['UNK']]
#     nb_unknown = 0
#     for id_weight, word in enumerate(self.vocab):
#         if word in dictionary:
#             id_params = dictionary[word]
#             params = parameters[id_params]
#         else:
#             #print('Warning: word `{}` not in dictionary'.format(word))
#             params = unknown_params
#             nb_unknown += 1
#         weight[id_weight+1] = torch.from_numpy(params)
#     state_dict = OrderedDict({'weight':weight})
#     if nb_unknown > 0:
#         print('Warning: {}/{} words are not in dictionary, thus set UNK'
#               .format(nb_unknown, len(dictionary)))
#     return state_dict

# def load_embedding(self):
#     self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 1,
#                                   embedding_dim=620,
#                                   padding_idx=0, # -> first_dim = zeros (?)
#                                   sparse=False)

#     dictionary = self._load_dictionary()
#     parameters = self._load_emb_params()
#     state_dict = self._make_emb_state_dict(dictionary, parameters)
#     self.embedding.load_state_dict(state_dict)
#     if self.save:
#         torch.save(self.embedding, path)
#     return self.embedding