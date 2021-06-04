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
import pdb
import torch
from sklearn import metrics
from functions import seq2num,element,combination,hilbert_curve,plot_hb_dna,read_file,plot_row1,snake_curve
from model_search import Network
from architect import Architect
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
use_plot = False
use_save = False
if use_save:
    import pickle
    from datetime import datetime
parser = argparse.ArgumentParser("ALL DON DATA SPLICE SITE CLASSIFICATION")
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--val_size', type=int, default=100, help='validation batch size')
parser.add_argument('--learning_rate', type=float, default=0.0025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='ALL_DON', help='experiment name')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.8, help='portion of validation data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 2



np.random.seed(args.seed)
gpus = [int(i) for i in args.gpu.split(',')]
if len(gpus)==1:
    torch.cuda.set_device(int(args.gpu))

cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
logging.info('gpu device = %s' % args.gpu)
logging.info("args = %s", args)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)

if len(gpus)>1:
    model = nn.DataParallel(model)

model = model.cuda()

arch_params = list(map(id, model.module.arch_parameters()))
weight_params = filter(lambda p: id(p) not in arch_params,model.parameters())


logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

optimizer = torch.optim.SGD(
    weight_params,
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)

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

file_name1 ="/raid/project/darts_splice/data_splice/proc_all/all_don_seq.txt"
file_name2 ="/raid/project/darts_splice/data_splice/proc_all/all_don_label.txt"
Seq = [line.rstrip('\n')[:-1] for line in open(file_name1)]
Raw_lab = [line.rstrip('\n') for line in open(file_name2)]
n = len(Raw_lab)
lab_dic = {
    "1": np.array([0, 1]),
    "0": np.array([1, 0])
}
LABEL = seq2num(Raw_lab, lab_dic)
sub_length = 1
a = Seq[1]
elements = element(a)
mapping_dic = combination(elements, sub_length)
H = np.array(range(602)).reshape(602,1)
d_1,d_2 = H.shape
DATA_ = -1. * np.ones((n, d_1, d_2, 4 ** sub_length))
for i in range(n):
    DATA_[i, :, :]= plot_hb_dna(seq=Seq[i],H_curve=H,sub_length=sub_length, map_dic=mapping_dic)



IMG = DATA_
IMG=np.transpose(IMG,[0, 2, 1, 3])


LABELS= LABEL
d1,d2,d3,d4 = IMG.shape
LABELS=torch.from_numpy(LABELS)
IMG=torch.from_numpy(IMG)
train_data = data_utils.TensorDataset(IMG,LABELS)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))
train_idx = np.random.choice(indices, size=split, replace=False)
int_idx = list(set(indices) - set(train_idx))
train_sampler = SubsetRandomSampler(train_idx)
train_queue = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size, sampler=train_sampler)

num_valid=len(int_idx)
indices_valid=list(int_idx)
valid_split = int(np.floor(args.valid_portion* num_valid))
valid_idx = np.random.choice(indices_valid, size=valid_split, replace=False)
test_idx=list(set(indices_valid) - set(valid_idx))
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)
valid_queue = torch.utils.data.DataLoader(train_data,batch_size=args.val_size, sampler=valid_sampler)
test_queue = torch.utils.data.DataLoader(train_data,batch_size=args.val_size, sampler=test_sampler)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, float(args.epochs), eta_min=args.learning_rate_min)

architect = Architect(model, args)


def append_new_line(file_name, text_to_append):
   """Append given text as a new line at the end of file"""
   with open(file_name, "a+") as file_object:
      file_object.seek(0)
      data = file_object.read(100)
      if len(data) > 0:
         file_object.write("\n")
      file_object.write(text_to_append)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
      model.train()
      n = input.size(0)
      input=input.float()
      input=input.cuda()
      target=target.cuda()
      input_search, target_search = next(iter(valid_queue))
      input_search=input_search.float()
      input_search=input_search.cuda()
      target_search=target_search.cuda()
      model.train()
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
      optimizer.zero_grad()
      logits = model(input)
      loss = criterion(logits, torch.max(target, 1)[1])
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()
      scheduler.step() 
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
      if step % args.report_freq == 0:
        logging.info('all_don_search train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input=input.float()
    input=input.cuda()
    target=target.cuda()
    logits = model(input)
    loss = criterion(logits,torch.max(target, 1)[1])

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('all_don_search valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def testinfer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  auc = utils.AvgrageMeter()
  model.eval()
  y_true, y_pred, auc_pred = [], [], []
  for step, (input, target) in enumerate(valid_queue):
    input=input.float()
    input=input.cuda()
    target=target.cuda()
    logits = model(input)
    loss = criterion(logits,torch.max(target, 1)[1])
    aa=logits.cpu().data.numpy()
    aucpred = np.divide(np.exp(aa),  np.sum(np.exp(aa), axis=1).reshape((aa.shape[0],1)))[:, 0]
    pred = torch.max(logits.data, dim=1)[1].cpu().numpy().tolist()
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    auc.update(aucpred.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('all_don_search valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    target=torch.max(target, 1)[1]
    auc_pred.extend(aucpred)
    y_pred.extend(pred)
    y_true.extend(target.data)
  return top1.avg, objs.avg,y_pred,y_true,auc.avg



bb=0
best_acc= 0
result = {}
train_loss_ = []
valid_loss_ = []
train_don_ = []
valid_don_ = [] 
for epoch in range(1,args.epochs+1):
  lr = scheduler.get_last_lr()[0]
  logging.info('epoch %d lr %e', epoch, lr)
  genotype = model.module.genotype()
  logging.info('*****************************************************************')
  logging.info('*****************************************************************')
  logging.info('*****************************************************************')
  logging.info('| ALL_DON_SEARCH-ARCH{}= {}'.format(epoch,genotype))
  logging.info('*****************************************************************')
  logging.info('*****************************************************************')
  logging.info('*****************************************************************')
  print(F.softmax(model.module.alphas_normal, dim=-1))
  print(F.softmax(model.module.alphas_reduce, dim=-1))
  train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
  logging.info('-' * 89)
  logging.info('all_don_search train_acc %f', train_acc)
  logging.info('all_don_search train_loss %f', train_obj)
  logging.info('-' * 89)
  train_loss_.append(train_obj)
  train_don_.append(train_acc)
  valid_acc, valid_obj = infer(valid_queue, model, criterion)
  logging.info('-' * 89)
  logging.info('all_don_search valid_acc %f', valid_acc)
  logging.info('all_don_search valid_loss %f', valid_obj)
  logging.info('-' * 89)
  valid_don_.append(valid_acc)
  valid_loss_.append(valid_obj)
  is_best = False
  if valid_acc > best_acc:
    bb=bb+1  
    best_acc = valid_acc
    logging.info('*' * 89)
    logging.info('WOWWW !!!..ALL_DON_SEARCH MODEL JUST GOT BETTER AND IM AT %f', best_acc)
    logging.info('*' * 89)
    is_best = True
    geno= 'all_acc' + str(bb) + '=' + str(genotype)
    append_new_line(os.path.join(args.save, 'genotypes_all_acc.txt'),geno)

  utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc': best_acc,
      'optimizer' : optimizer.state_dict(),
      },model,optimizer,is_best, args.save)

result['train loss'] = train_loss_
result['valid loss'] = valid_loss_
result['train acc'] = train_don_
result['valid acc'] = valid_don_
if use_plot:
   import PlotFigure as PF
   PF.PlotFigure(result, use_save)
if use_save:
   filename = 'ALL_DON_SEARCH_DARTS_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
   result['filename'] = filename
   fp = open(filename, 'wb')
   pickle.dump(result, fp)
   print('File %s is saved.' % filename)
model = torch.load(os.path.join(args.save, 'model_best.pt'))
parallel_model = model.cuda()

test_acc, test_obj,y_pred,y_true,auc_pred = testinfer(test_queue, model, criterion)
logging.info('=' * 89)
logging.info('| End of training |')
logging.info('all_don_search test_acc %f', test_acc)
logging.info('all_don_search test_loss %f', test_obj)
test_f1 = metrics.f1_score(torch.stack(y_true).tolist(),y_pred, average='macro')
logging.info('all_don_search F1 Score %f', test_f1)
logging.info('all_don_search Precision, Recall and F1-Score...')
logging.info(metrics.classification_report(torch.stack(y_true).tolist(), y_pred))
logging.info('all_don_search Confusion Matrix...')
cm = metrics.confusion_matrix(torch.stack(y_true).tolist(), y_pred)
logging.info(cm)


