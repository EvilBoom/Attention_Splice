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
import pandas as pd
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
parser = argparse.ArgumentParser("ALL_ACC_EVAL SPLICE SITE CLASSIFICATION")
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--val_size', type=int, default=100, help='validation batch size')
parser.add_argument('--learning_rate', type=float, default=0.0025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')
parser.add_argument('--epochs', type=int, default=70, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='All_ACC', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.8, help='portion of validation data')
parser.add_argument('--arch', type=str, default='all_acc', help='which architecture to use')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()
args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
SPLICE_CLASSES = 2
np.random.seed(args.seed)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
genotype = eval("genotypes.%s" % args.arch)
model = Network(args.init_channels, SPLICE_CLASSES, args.layers, args.auxiliary, genotype)
model = model.cuda()
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
def batchify1(data, seq_len, bsz, args):
   nbatch = data.size(0) // (seq_len * bsz)
   data = data.narrow(0, 0, nbatch * bsz * seq_len)
   data = data.view(bsz * nbatch, seq_len).t().contiguous()
   print(data.size())
   data = data.cuda()
   return data
import data 
import torch.utils.data as data_utils
file_name1 ="/media/appadmin/MY_BOOK/data/splicing/asp_data_subset/proc_all/all_acc_seq.txt"
file_name2 ="/media/appadmin/MY_BOOK/data/splicing/asp_data_subset/proc_all/all_acc_label.txt"
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
H = np.array(range(142)).reshape(142,1)
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
for species in ['hs','athaliana','celegans','dmelanogaster']:
    file_test_seq='/media/appadmin/MY_BOOK/data/splicing/asp_data_subset/proc_' + species +'/test_' + species + '_acc_seq.txt'
    file_test_label='/media/appadmin/MY_BOOK/data/splicing/asp_data_subset/proc_' + species +'/test_' + species + '_acc_label.txt'
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
    test_sp_queue='test_'+species+'_queue'
    vars()[test_sp_queue]=torch.utils.data.DataLoader(test_data,batch_size=args.val_size, sampler=test_sampler_sep)
def train(train_queue, valid_queue, model,criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()
  for step, (input, target) in enumerate(train_queue):
      n = input.size(0)
      input=input.float()
      input=input.cuda()
      target=target.cuda()
      optimizer.zero_grad()
      logits,logits_aux = model(input)
      loss = criterion(logits, torch.max(target, 1)[1])
      if args.auxiliary:
         loss_aux = criterion(logits_aux,torch.max(target, 1)[1])
         loss += args.auxiliary_weight*loss_aux     
      loss.backward()
      nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
      optimizer.step()
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
      if step % args.report_freq == 0:
        logging.info('all_acc_eval train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
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
    logits,_ = model(input)
    loss = criterion(logits,torch.max(target, 1)[1])
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    if step % args.report_freq == 0:
      logging.info('all_acc_eval valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  return top1.avg, objs.avg
def testinfer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  auc = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  y_true, y_pred, auc_pred = [], [], []
  for step, (input, target) in enumerate(valid_queue):
    input=input.float()
    input=input.cuda()
    target=target.cuda()
    logits,_ = model(input)
    loss = criterion(logits,torch.max(target, 1)[1])
    pred = torch.max(logits.data, dim=1)[1].cpu().numpy().tolist()
    bb=torch.max(target, 1)[1]
    cc= bb.cpu().data.numpy()
    fpr, tpr, thresholds = metrics.roc_curve(cc,pred)
    aucpred= metrics.auc(fpr, tpr)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    auc.update(aucpred.item(), n)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    if step % args.report_freq == 0:
      logging.info('all_acc_eval valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    target=torch.max(target, 1)[1]
    y_pred.extend(pred)
    y_true.extend(target.data)
  return top1.avg, objs.avg,y_pred,y_true,auc.avg
best_acc= 0
result = {}
train_loss_ = []
valid_loss_ = []
train_acc_ = []
valid_acc_ = [] 
for epoch in range(1,args.epochs+1):
  lr = scheduler.get_last_lr()[0]
  logging.info('epoch %d lr %e', epoch, lr)
  logging.info('*****************************************************************')
  logging.info('*****************************************************************')
  logging.info('*****************************************************************')
  logging.info('| ALL_ACC ARCH= {}'.format(genotype))
  logging.info('*****************************************************************')
  logging.info('*****************************************************************')
  logging.info('*****************************************************************')
  model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
  train_acc, train_obj = train(train_queue, valid_queue, model,criterion, optimizer)
  logging.info('-' * 89)
  logging.info('all_acc_eval train_acc %f', train_acc)
  logging.info('all_acc_eval train_loss %f', train_obj)
  logging.info('-' * 89)
  train_loss_.append(train_obj)
  train_acc_.append(train_acc)
  valid_acc, valid_obj = infer(valid_queue, model, criterion)
  logging.info('-' * 89)
  logging.info('all_acc_eval valid_acc %f', valid_acc)
  logging.info('all_acc_eval valid_loss %f', valid_obj)
  logging.info('-' * 89)
  valid_acc_.append(valid_acc)
  valid_loss_.append(valid_obj)
  is_best = False
  if valid_acc > best_acc:
    best_acc = valid_acc
    logging.info('*' * 89)
    logging.info('WOWWW !!!..ALL_ACC MODEL EVAL JUST GOT BETTER AND IM AT %f', best_acc)
    logging.info('*' * 89)
    is_best = True
    test_all_acc, test_all_obj,y_all_pred,y_all_true,all_auc_pred = testinfer(test_queue, model, criterion)
    test_hs_acc, test_hs_obj,y_hs_pred,y_hs_true,hs_auc_pred = testinfer(test_hs_queue, model, criterion)
    test_athaliana_acc, test_athaliana_obj,y_athaliana_pred,y_athaliana_true,athaliana_auc_pred = testinfer(test_athaliana_queue, model, criterion)
    test_celegans_acc, test_celegans_obj,y_celegans_pred,y_celegans_true,celegans_auc_pred = testinfer(test_celegans_queue, model, criterion)
    test_dmelanogaster_acc, test_dmelanogaster_obj,y_dmelanogaster_pred,y_dmelanogaster_true,dmelanogaster_auc_pred = testinfer(test_dmelanogaster_queue, model, criterion)
    logging.info('=' * 89) 
    logging.info('| Intermediate test results at epoch : %d |',epoch)
    logging.info('Intermediate all_acc test_acc %f', test_all_acc)
    logging.info('Intermediate hs_acc test_acc %f', test_hs_acc)
    logging.info('Intermediate athaliana_acc test_acc %f', test_athaliana_acc)
    logging.info('Intermediate celegans_acc test_acc %f', test_celegans_acc)
    logging.info('Intermediate dmelanogaster_acc test_acc %f', test_dmelanogaster_acc)
    logging.info('=' * 89)
    logging.info('=' * 89)
    logging.info('Intermediate all_acc test_auc %f', all_auc_pred*100)
    logging.info('Intermediate hs_acc test_auc %f', hs_auc_pred*100)
    logging.info('Intermediate athaliana_acc test_auc %f', athaliana_auc_pred*100)
    logging.info('Intermediate celegans_acc test_auc %f', celegans_auc_pred*100)
    logging.info('Intermediate dmelanogaster_acc test_auc %f', dmelanogaster_auc_pred*100)
    logging.info('=' * 89)
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer' : optimizer.state_dict(),
        },model,optimizer,is_best, args.save)
result['train loss'] = train_loss_
result['valid loss'] = valid_loss_
result['train acc'] = train_acc_
result['valid acc'] = valid_acc_
if use_plot:
   import PlotFigure as PF
   PF.PlotFigure(result,args.save,use_save)
if use_save:
   filename = 'ALL_ACC_EVAL_DARTS_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
   result['filename'] = filename
   fp = open(os.path.join(args.save,filename), 'wb')
   pickle.dump(result, fp)
   print('File %s is saved.' % filename)
model = torch.load(os.path.join(args.save, 'model_best.pt'))
parallel_model = model.cuda()
test_acc, test_obj,y_pred,y_true,auc_pred = testinfer(test_queue, model, criterion)
logging.info('=' * 89)
logging.info('| End of training |')
logging.info('all_acc_eval test_acc %f', test_acc)
logging.info('all_acc_eval test_loss %f', test_obj)
test_f1 = metrics.f1_score(torch.stack(y_true).tolist(), y_pred, average='macro')
logging.info('all_acc_eval F1 Score %f', test_f1)
logging.info('all_acc_eval Precision, Recall and F1-Score...')
logging.info(metrics.classification_report(torch.stack(y_true).tolist(), y_pred))
logging.info('all_acc_eval Confusion Matrix...')
cm = metrics.confusion_matrix(torch.stack(y_true).tolist(), y_pred)
logging.info(cm)
logging.info('all_acc_eval roc_auc_score...')
logging.info(auc_pred)
logging.info('=' * 89)
