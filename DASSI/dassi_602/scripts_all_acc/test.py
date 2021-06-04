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
parser.add_argument('--model', type=str, help='model_species')
parser.add_argument('--val_size', type=int, default=100, help='validation batch size')
parser.add_argument('--learning_rate', type=float, default=0.0025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='CELEGANS_ACC', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.9, help='portion of validation data')
parser.add_argument('--arch', type=str, default='celegans_acc', help='which architecture to use')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


CIFAR_CLASSES = 2



np.random.seed(args.seed)
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
logging.info('gpu device = %d' % args.gpu)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()


def common_data(list1, list2): 
	result = 0

	for x in list1: 

		for y in list2: 
	
			if x == y: 
				result = result+1
				
	return result 
	
a = [1, 2, 3, 4, 5] 
b = [5, 6, 7, 8, 9] 
print(common_data(a, b)) 

a = [1, 2, 3, 4, 5] 
b = [6, 7, 8, 9] 
print(common_data(a, b)) 


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

model_species=args.model

typ=args.type

file_test_seq='/media/appadmin/MY_BOOK/data/splicing/splice2deep_data/proc_' + data_species +'/test_' + data_species + '_' + typ +'_seq.txt'
file_test_label='/media/appadmin/MY_BOOK/data/splicing/splice2deep_data/proc_' + data_species +'/test_' + data_species + '_' + typ +'_label.txt'
lab_dic = {
    "1": np.array([0, 1]),
    "0": np.array([1, 0])
}
sub_length = 1
H = np.array(range(602)).reshape(602,1)










test_Seq = [line.rstrip('\n')[:-1] for line in open(file_test_seq)]
test_Raw_lab = [line.rstrip('\n') for line in open(file_test_label)]
test_LABEL = seq2num(test_Raw_lab, lab_dic)
test_n = len(test_Raw_lab)
test_a = test_Seq[1]
test_elements = element(test_a)
test_mapping_dic = combination(test_elements, sub_length)
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
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('Test %03d %f %f', step, top1.avg, top5.avg)
    target=torch.max(target, 1)[1]
    y_pred.extend(pred)
    y_true.extend(target.data)
  return top1.avg,y_pred,y_true,aucpred






modl='/media/appadmin/MY_BOOK/data/darts_revision/models_new/model_'+ model_species + '_'+ typ + '.pt' 

model = torch.load(modl)
parallel_model = model.cuda()

test_acc,y_pred,y_true,auc_pred = testinfer(test_queue, model, criterion)
logging.info('=' * 89)
logging.info('Prediction on TEST '+ data_species + ' ' + typ + 'data using ' + model_species + ' ' + typ + 'model'  )
logging.info(' test_acc %f', test_acc)
test_f1 = metrics.f1_score(torch.stack(y_true).tolist(), y_pred, average='macro')
logging.info(' F1 Score %f', test_f1)
logging.info(' Precision, Recall and F1-Score...')
logging.info(metrics.classification_report(torch.stack(y_true).tolist(), y_pred))
logging.info(' Confusion Matrix...')
cm = metrics.confusion_matrix(torch.stack(y_true).tolist(), y_pred)
logging.info(cm)
logging.info(' roc_auc_score...')
logging.info(auc_pred)
logging.info('=' * 89)

