import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import genotypes
import pdb
import torch
from sklearn import metrics
from functions import seq2num,element,combination,hilbert_curve,plot_hb_dna,read_file,plot_row1,snake_curve
from torch.autograd import Variable
from model import NetworkCIFAR as Network
from architect import Architect
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
use_plot = True
use_save = True
if use_save:
    import pickle
    from datetime import datetime
parser = argparse.ArgumentParser("TEST SPLICE SITE CLASSIFICATION")
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--data', type=str, help='data_species')
parser.add_argument('--type', type=str, help='typ')
parser.add_argument('--test_file', type=str, help='testfile')
parser.add_argument('--val_size', type=int, default=500, help='validation batch size')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


CIFAR_CLASSES = 2




criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
	

def batchify1(data, seq_len, bsz, args):
   nbatch = data.size(0) // (seq_len * bsz)
   data = data.narrow(0, 0, nbatch * bsz * seq_len)
   data = data.view(bsz * nbatch, seq_len).t().contiguous()
   print(data.size())
   data = data.cuda()
   return data

import data 
import torch.utils.data as data_utils

data_species= args.data 


typ=args.type
test_fil=str(args.test_file)
file_test_seq=test_fil
file_test_label=test_fil.split("seq",1)[0] +'label.txt'
lab_dic = {
    "1": np.array([0, 1]),
    "0": np.array([1, 0])
}
sub_length = 1
test_Seq = [line.rstrip('\n')[:-1] for line in open(file_test_seq)]
test_Raw_lab = [line.rstrip('\n') for line in open(file_test_label)]
test_LABEL = seq2num(test_Raw_lab, lab_dic)
test_n = len(test_Raw_lab)
test_a = test_Seq[1]
test_elements = element(test_a)
test_mapping_dic = combination(test_elements, sub_length)
if len(test_Seq[0])==140:
    H = np.array(range(142)).reshape(142,1)
else:
    H = np.array(range(602)).reshape(602,1)
test_d_1,test_d_2 = H.shape
test_DATA_ = -1. * np.ones((test_n, test_d_1, test_d_2, 4 ** sub_length))
for i in range(test_n):
        test_DATA_[i, :, :]= plot_hb_dna(seq=test_Seq[i],H_curve=H,sub_length=sub_length, map_dic=test_mapping_dic)
test_IMG = test_DATA_
test_IMG=np.transpose(test_IMG,[0, 2, 1, 3])
test_LABELS= test_LABEL
test_d1,test_d2,test_d3,test_d4 = test_IMG.shape
test_LABELS=torch.from_numpy(test_LABELS)
test_IMG=torch.from_numpy(test_IMG)
test_data = data_utils.TensorDataset(test_IMG,test_LABELS)
num_test_sep=len(test_data)
indices_test=list(range(num_test_sep))
split_test=int(np.floor(num_test_sep))
test_idx_sep=np.random.choice(indices_test, size=split_test, replace=False)
test_sampler_sep = SubsetRandomSampler(test_idx_sep)
test_queue = torch.utils.data.DataLoader(test_data,batch_size=args.val_size, sampler=test_sampler_sep)

print('*********************************************************************') 
print('*********************************************************************') 
print('*********************************************************************') 
print('Number of Unseen %s test samples: %d '%(data_species,test_n)) 
print('*********************************************************************') 
print('*********************************************************************') 
print('*********************************************************************') 


def testinfer(test_queue, model, criterion):
  auc = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  y_true, y_pred, auc_pred = [], [], []
  for step, (input, target) in enumerate(test_queue):
    input=input.float()
    input=input.cuda()
    target=target.cuda()
    
    logits,_= model(input)
    pred = torch.max(logits.data, dim=1)[1].cpu().numpy().tolist()
    bb=torch.max(target, 1)[1]
    cc= bb.cpu().data.numpy()
    fpr, tpr, thresholds = metrics.roc_curve(cc,pred)
    aucpred= metrics.auc(fpr, tpr)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    auc.update(aucpred.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('Test %03d %f %f', step, top1.avg, top5.avg)
    target=torch.max(target, 1)[1]
    y_pred.extend(pred)
    y_true.extend(target.data)
  return top1.avg,y_pred,y_true,auc.avg

modl='/media/appadmin/MY_BOOK/data/darts_revision/models_new/model_all_'+ typ +'_'+ str(len(test_Seq[0])+1) + '_v2.pt' 

model = torch.load(modl)
parallel_model = model.cuda()

test_acc,y_pred,y_true,auc_p = testinfer(test_queue, model, criterion)
logging.info('=' * 89)
logging.info('Prediction on TEST '+ data_species + ' ' + typ + 'data using ' + ' ' + typ + 'model'  )
logging.info(' test_acc %f', test_acc)
test_f1 = metrics.f1_score(torch.stack(y_true).tolist(), y_pred, average='macro')
logging.info(' F1 Score %f', test_f1)
logging.info(' Precision, Recall and F1-Score...')
logging.info(metrics.classification_report(torch.stack(y_true).tolist(), y_pred))
logging.info(' Confusion Matrix...')
cm = metrics.confusion_matrix(torch.stack(y_true).tolist(), y_pred)
logging.info(cm)
logging.info(' roc_auc_score...')
logging.info(auc_p)
logging.info('=' * 89)

