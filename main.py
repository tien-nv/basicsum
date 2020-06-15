#!/usr/bin/env python3

import json
import model
import utils
import argparse, random, logging, numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from time import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir', type=str, default='checkpoints/')
parser.add_argument('-embed_dim', type=int, default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-pos_dim', type=int, default=50)
parser.add_argument('-pos_num', type=int, default=100)
parser.add_argument('-seg_num', type=int, default=10)
parser.add_argument('-kernel_num', type=int, default=100)
parser.add_argument('-kernel_sizes', type=str, default='3,4,5')
parser.add_argument('-model', type=str, default='RNN_RNN')
parser.add_argument('-hidden_size', type=int, default=200)
# train
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-train_dir', type=str, default='../data/train_cnn_dailymail.json')
parser.add_argument('-val_dir', type=str, default='../data/val_cnn_dailymail.json')
parser.add_argument('-embedding', type=str, default='input/embedding.npz')
parser.add_argument('-word2id', type=str, default='input/word2id.json')
parser.add_argument('-report_every', type=int, default=1000)
parser.add_argument('-seq_trunc', type=int, default=50)
parser.add_argument('-max_norm', type=float, default=1.0)
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1_epoch_2_id_7999.pt')
# device
parser.add_argument('-device', type=int, default=0)
# option

args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)


def eval(net, vocab, data_iter, criterion):
    net.eval()
    if use_gpu:
        net.cuda()
    print("start compute validation");
    total_loss = 0
    batch_num = 0
    time_val = 0
    for batch in tqdm(data_iter):
        t1_val = time()
        features, targets, _, doc_lens = vocab.make_features(batch)
        features, targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_lens)
        loss = criterion(probs, targets)
        total_loss += loss.data.item()
        batch_num += 1
        t2_val = time()
        time_val += t2_val-t1_val
    loss = total_loss / batch_num
    print("done!")
    net.train()
    return loss


def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')

    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id,encoding='utf-8') as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir,encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir,encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    # net = getattr(model, args.model)(args, embed)
    if use_gpu:
        checkpoint = torch.load(args.load_dir, map_location='cuda:0')
    else:
        checkpoint = torch.load(args.load_dir, map_location='cpu')
    if not use_gpu:
        checkpoint['args'].device = None
    # print(checkpoint['args'].model)
    net = getattr(model, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    # print(net)
    if use_gpu:
        net.cuda()
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False)
    # loss function
    criterion = nn.BCELoss()
    # model info
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))

    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()

    t1 = time()
    for epoch in range(4, args.epochs):
        print("Epoch: ",epoch)
        for i, batch in enumerate(tqdm(train_iter)):
            features, targets, _, doc_lens = vocab.make_features(batch)
            features, targets = Variable(features), Variable(targets.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features, doc_lens)
            loss = criterion(probs, targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optimizer.step()
            # if i%200 == 0:
            #     print('Batch ID:%d Loss:%f' % (i, loss.data.item()))
            if i % args.report_every == 0:
                cur_loss = eval(net, vocab, val_iter, criterion)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    net.save(epoch,i)
                logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                             % (epoch, min_loss, cur_loss))
    t2 = time()
    logging.info('Total Cost:%f h' % ((t2 - t1) / 3600))

if __name__ == '__main__':
    train()

# python3 run.py -batch_size 64 -load_dir checkpoints/RNN_RNN_seed_1_YOUR.pt
# python3 main_v2.py -batch_size 64 -hyp outputs/hyp/cnn_dailymail/RNN_RNN_7F/SRSF_RNN_RNN_V2_2019_10_18_19_16_49_2_seed_1_0_6_topk_m -test -test_dir data/test/test_cnn_dailymail.json -load_dir checkpoints/checkpoints_RNN_RNN_8F/SRSF_RNN_RNN_V2_2019_10_18_19_16_49_2_seed_1.pt
# python3 main_v3.py -batch_size 64 -hyp outputs/hyp/cnn_dailymail/RNN_RNN_9F/SRSF_RNN_RNN_V4_2019_11_13_11_20_45_2_seed_1.pt_0_6_topk_m -test -test_dir data/test/test_cnn_dailymail.json -load_dir checkpoints/SRSF_RNN_RNN_V4_2019_11_13_11_20_45_2_seed_1.pt
# python3 run.py -batch_size 32 -hyp outputs/hyp/cnn_dailymail/RNN_RNN_2F/SRS2F_RNN_RNN_2019_10_15_17_29_02_2_seed_1_0_60_top_ktt -test -test_dir data/test/test_cnn_dailymail.json -load_dir checkpoints/checkpoints_2f/SRS2F_RNN_RNN_2019_10_15_17_29_02_2_seed_1.pt
