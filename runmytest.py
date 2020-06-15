#cần phải có file __init__.py để thông báo đó là 1 package.
# from preprocess import word2vec as w2v
# w2v.word2vec('c')
# from preprocess import dataset,process
import model
import utils

import json
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm
# from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-auto_save_dir',type=str,default='checkpoint/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='Bertsum_biGRU')
parser.add_argument('-hidden_size',type=int,default=200) #hoặc 3072 nếu concatnate 4 layer cuối
parser.add_argument('-embedding', type=str, default='input/embedding.npz')
parser.add_argument('-word2id', type=str, default='input/word2id.json')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=32)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1_epoch_1_id_2000.pt')
parser.add_argument('-test_dir',type=str,default='../data/test_cnn_dailymail.json')
parser.add_argument('-ref',type=str,default='outputs/ref/') #references
parser.add_argument('-hyp',type=str,default='outputs/hyp/') #hypothe
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=4) #top các câu xác suất cao nhất
# device
parser.add_argument('-device',type=int,default=None)
#eval
parser.add_argument('-eval',type=bool,default=True)
#option
parser.add_argument('-test',type=bool,default=True)
#
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


def mytest():
    print("====running====")
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]
    test_dataset = utils.Dataset(examples)

    test_iter = DataLoader(dataset=test_dataset,
                           batch_size=args.batch_size,
                           shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir, map_location='cuda:0')
    else:
        checkpoint = torch.load(args.load_dir, map_location='cpu')
    # print(checkpoint)
    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    print(checkpoint['args'].model)
    net = getattr(model, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    print(net)
    if use_gpu:
        net.cuda()
    net.eval()

    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(test_iter):
        features, _, summaries, doc_lens = vocab.make_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id, doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            # topk = min(args.topk, doc_len)
            # topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            # topk_indices.sort()
            # doc = batch['doc'][doc_id].split('\n')[:doc_len]
            # hyp = [doc[index] for index in topk_indices]

            # =====================
            # topk = min(args.topk, doc_len)
            prob_n = prob.cpu().data.numpy()
            topk_indices = np.where(prob_n > 0.6)
            # print(topk_indices)
            if len(topk_indices[0]) > args.topk:
                topk_index = topk_indices[0][:args.topk]
                topk_index = sorted(topk_index)
            else:
                topk_index = topk_indices[0]
                topk_index = sorted(topk_index)
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            hyp = [doc[index] for index in topk_index]
            # ref = summaries[doc_id]
            # with open(os.path.join(args.ref, str(file_id) + '.txt'), 'w') as f:
            #     f.write(ref)
            
            with open(os.path.join(args.hyp, str(file_id) + '.txt'), 'w') as f:
                f.write('. '.join(hyp))
                #f.write('. \n'.join(hyp))
                #f.write('.')
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))

if __name__=='__main__':
    if args.test:
        mytest()