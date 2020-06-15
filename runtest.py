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
# option
parser.add_argument('-test',action='store_false')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
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

def test():
    logging.info('Loading vocab,test dataset.Wait a second,please')

    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id,encoding='utf-8') as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    # build model
    
    with open(args.test_dir,encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    test_dataset = utils.Dataset(examples)

    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(model,checkpoint['args'].model)(checkpoint['args'],embed)
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    print('running test!')
    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    count = 0
    for batch in tqdm(test_iter):
        count+=1
        features, targets,summaries, doc_lens = vocab.make_features(batch)
        t1 = time()
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_lens)
        # probs=probs.to('cpu')
        # y_pred = np.where(probs>=0.6,1,0)
        # y_true = targets
        # accuracy += accuracy_score(y_true,y_pred)
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            # print(prob)
            
            # label = labels[start:stop]
            # prob_n = prob.cpu().data.numpy()
            
            
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].to('cpu')
            topk_indices.sort()
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            ref = summaries[doc_id]
            with open(os.path.join(args.ref,str(file_id)+'.txt'), 'w',encoding='utf-8') as f:
                f.write(ref)
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w',encoding='utf-8') as f:
                f.write('.\n'.join(hyp))
            start = stop
            file_id = file_id + 1
        t2 = time()
        time_cost += t2 - t1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))
import nltk
import os
import rouge
# import nltk
from shutil import copyfile
import statistics as sta
import sys
sys.setrecursionlimit(15000)
# nltk.download('punkt')
def prepare_results(metric,p, r, f):
    # return '\t{}:\t{}: {:5.2f}'.format(metric, 'F1', 100.0 * f)
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'Percision', 100.0 * p, 'Recall', 100.0 * r, 'F1-score', 100.0 * f)


def my_eval():
    hyp = 'outputs/hyp'
    # ref = 'summary_to_evaluate'
    raw_ref = 'outputs/ref'    
    FJoin = os.path.join
    files_hyp = [FJoin(hyp, f) for f in os.listdir(hyp)]
    files_raw_ref = [FJoin(raw_ref, f) for f in os.listdir(raw_ref)]
    
    f_hyp = []
    f_raw_ref = []
    for file in files_hyp:
        f = open(file,encoding='utf-8')
        f_hyp.append(f.read())
        f.close()
    for file in files_raw_ref:
        f = open(file,encoding='utf-8')
        f_raw_ref.append(f.read())
        f.close()
    print("compute 75bytes: ")
    mrouge_75 = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],max_n=2,
                             limit_length=True,length_limit=75,
                             length_limit_type='bytes',apply_avg=True)
    scores = mrouge_75.get_scores(f_hyp,f_raw_ref)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
          print(prepare_results(metric,results['p'], results['r'], results['f']))
    print("compute 275bytes: ")
    mrouge_275 = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],max_n=2,
                             limit_length=True,length_limit=275,
                             length_limit_type='bytes',apply_avg=True)
    scores = mrouge_275.get_scores(f_hyp,f_raw_ref)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
          print(prepare_results(metric,results['p'], results['r'], results['f']))
    print("compute full lenght: ")
    mrouge_full = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],max_n=2,
                             limit_length=False,
                             length_limit_type='bytes',apply_avg=True)
    scores = mrouge_full.get_scores(f_hyp,f_raw_ref)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
          print(prepare_results(metric,results['p'], results['r'], results['f']))
if __name__=='__main__':
    test()
    my_eval()