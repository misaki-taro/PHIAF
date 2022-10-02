# -*- coding: utf-8 -*-
import os
import pandas as pd
import warnings
import numpy as np    
import random 
import math
import argparse
import pickle

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from CNNModel import *
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter 
from utils import *

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(1)


######################################
######## INPUT PARAMETERS ############
######################################
parser = argparse.ArgumentParser(description='MyPHIAF')
parser.add_argument('--neg-each-epoch', default=False, type=bool, help='Neg sample for each epoch')
parser.add_argument('--out-channels', default=32, type=int, help='Out channels of Model')
parser.add_argument('--batchsize', default=8, type=int, help='Batch size of training')
parser.add_argument('--subdir', default='sub', type=str, help='Sub directory of results')

inputs = parser.parse_args()

# Print configurations
print('######################################################')
print('######################################################')
print('neg_each_epoch:                  {0}'.format(inputs.neg_each_epoch))
print('out_channels:                    {0}'.format(inputs.out_channels))
print('batchsize:                       {0}'.format(inputs.batchsize))
print('subdir:                          {0}'.format(inputs.batchsize))

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ImageDataset(Dataset):
    def __init__(self, dna, pro, y):
        self.dna = dna
        self.pro = pro
        self.y = y
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.dna[idx], self.pro[idx], self.y[idx]

def reshapes(X_en_tra,X_pr_tra,X_en_val,X_pr_val):
    sq=int(math.sqrt(X_en_tra.shape[1]))
    if pow(sq,2)==X_en_tra.shape[1]:
        X_en_tra2=X_en_tra.reshape((-1,sq,sq))
        X_pr_tra2=X_pr_tra.reshape((-1,sq,sq))
        X_en_val2=X_en_val.reshape((-1,sq,sq))
        X_pr_val2=X_pr_val.reshape((-1,sq,sq))
    else:
        X_en_tra2=np.concatenate((X_en_tra,np.zeros((X_en_tra.shape[0],int(pow(sq+1,2)-X_en_tra.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        X_pr_tra2=np.concatenate((X_pr_tra,np.zeros((X_pr_tra.shape[0],int(pow(sq+1,2)-X_pr_tra.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        X_en_val2=np.concatenate((X_en_val,np.zeros((X_en_val.shape[0],int(pow(sq+1,2)-X_en_val.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        X_pr_val2=np.concatenate((X_pr_val,np.zeros((X_pr_val.shape[0],int(pow(sq+1,2)-X_pr_val.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
    return X_en_tra2, X_pr_tra2, X_en_val2, X_pr_val2

def obtainfeatures(data,file_path1,file_path2,strs):
    phage_features=[]
    host_features=[]
    labels=[]
    for i in data:
        phage_features.append(np.loadtxt(file_path1+i[0]+strs).tolist())
        host_features.append(np.loadtxt(file_path2+i[1].split('.')[0]+strs).tolist())
        labels.append(i[-1])
    return np.array(phage_features), np.array(host_features), np.array(labels)

def obtain_neg(X_tra,X_val):    
    X_tra_pos=[mm for mm in X_tra if mm[2]==1]
    X_neg=[]
    training_neg=[]
    phage=list(set([mm[0]for mm in X_tra_pos]))
    host=list(set([mm[1]for mm in X_tra_pos]))
    for p in phage:
        for h in host:
            if str(p)+','+str(h) in X_neg:
                continue
            else:
                training_neg.append([p,h,0])
    return random.sample(training_neg,len(X_tra_pos))

def format_print(print_dict):
    comment = '[Epoch {0:>3d}] '.format(print_dict['epoch'])
    for k, v in print_dict.items():
        if(k == 'epoch'):
            continue
        comment += '{0}: {1:.2f},'.format(k,v)
    print(comment)

# Convert sample to image-like data
def sample2image(X):
    # Obtain features
    X_phage_dna, X_host_dna, y_ =  obtainfeatures(X,'../data/phage_dna_norm_features/','../data/host_dna_norm_features/','.txt')
    X_phage_pro, X_host_pro, _ = obtainfeatures(X,'../data/phage_protein_normfeatures/','../data/host_protein_normfeatures/','.txt')
    
    # Reshapes
    X_phage_dna2, X_host_dna2, _, _ = reshapes(X_phage_dna, X_host_dna, X_phage_dna, X_host_dna)
    X_phage_pro2, X_host_pro2, _, _ = reshapes(X_phage_pro, X_host_pro, X_phage_pro, X_host_pro)
    
    X_dna_ = np.array([X_phage_dna2,X_host_dna2]).transpose(1,0,2,3)
    X_pro_ = np.array([X_phage_pro2,X_host_pro2]).transpose(1,0,2,3)
    
    return X_dna_, X_pro_, y_

result_all=[]
pred_all=[]
test_y_all=[]

data1=pd.read_csv('../data/data_pos_neg.txt',header=None,sep=',')
data1=data1[data1[2]==1]
allinter=[str(data1.loc[i,0])+','+str(data1.loc[i,1]) for i in data1.index]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
training=pd.read_csv('../data/data_pos_neg.txt',header=None,sep=',').values.tolist()
training_tar = [tt[2] for tt in training]

fold_result = []

for train_index, test_index in skf.split(training, training_tar): 
    ###obtain data
    X_tra=[training[ii] for ii in train_index if training[ii][2] == 1]
    X_val=[training[ii] for ii in test_index]
    neg_select=obtain_neg(X_tra,X_val)  ##add extra negative samples
    X_phage_tra_dna,X_host_tra_dna,y_tra=obtainfeatures(X_tra+neg_select,'../data/phage_dna_norm_features/','../data/host_dna_norm_features/','.txt')
    X_phage_val_dna,X_host_val_dna,y_val=obtainfeatures(X_val,'../data/phage_dna_norm_features/','../data/host_dna_norm_features/','.txt')
    X_phage_tra_pro,X_host_tra_pro,_=obtainfeatures(X_tra+neg_select,'../data/phage_protein_normfeatures/','../data/host_protein_normfeatures/','.txt')
    X_phage_val_pro,X_host_val_pro,_=obtainfeatures(X_val,'../data/phage_protein_normfeatures/','../data/host_protein_normfeatures/','.txt')
    
    # Reshapes to image -> [bs, c, h, w]
    X_phage_tra_dna3,X_host_tra_dna3,X_phage_val_dna3,X_host_val_dna3=reshapes(X_phage_tra_dna,X_host_tra_dna,X_phage_val_dna,X_host_val_dna)
    X_phage_tra_pro3,X_host_tra_pro3,X_phage_val_pro3,X_host_val_pro3=reshapes(X_phage_tra_pro,X_host_tra_pro,X_phage_val_pro,X_host_val_pro)
    X_dna = np.array([X_phage_tra_dna3,X_host_tra_dna3]).transpose(1,0,2,3)
    X_pro = np.array([X_phage_tra_pro3,X_host_tra_pro3]).transpose(1,0,2,3)

    X_dna_val = np.array([X_phage_val_dna3,X_host_val_dna3]).transpose(1,0,2,3)
    X_pro_val = np.array([X_phage_val_pro3,X_host_val_pro3]).transpose(1,0,2,3)
    
    if(inputs.neg_each_epoch is False):
        # Train
        train_dataset = ImageDataset(torch.tensor(X_dna, dtype=torch.float32), torch.tensor(X_pro, dtype=torch.float32), torch.tensor(y_tra, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=inputs.batchsize, shuffle=True)
    
    epochs = 300
    in_channels = 2
    out_channels = 32
    kernel_size = 3
    dna_pool_size = 3
    pro_pool_size = 2
    
    
    model = PHIAFModel(in_channels, inputs.out_channels, kernel_size, dna_pool_size, pro_pool_size).to(device)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    

    X_dna_val = torch.tensor(X_dna_val, dtype=torch.float, device=device)
    X_pro_val = torch.tensor(X_pro_val, dtype=torch.float, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float, device=device)
    bm = BinaryMetric(y_val)

    for epoch in range(1, 1+epochs):
        # Train
        model.train()
        
        if(inputs.neg_each_epoch is True):
            # neg sample for each epoch
            neg_select = obtain_neg(X_tra, X_val)
            X_dna, X_pro, y_tra = sample2image(X_tra+neg_select)
            train_dataset = ImageDataset(torch.tensor(X_dna, dtype=torch.float32), torch.tensor(X_pro, dtype=torch.float32), torch.tensor(y_tra, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=inputs.batchsize, shuffle=True)
        
        train_loss = AverageMeter()
        val_loss = 0.0
        

        for X_dna, X_pro, y in train_loader:
            X_dna, X_pro, y = X_dna.to(device), X_pro.to(device), y.to(device)
            y_hat = model(X_dna, X_pro)
            optimizer.zero_grad()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            
            train_loss.update(l.cpu().item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_hat = model(X_dna_val, X_pro_val)
            val_loss = loss(y_hat, y_val).cpu().item()
            auc = bm.auroc(y_hat)
            aupr = bm.auprc(y_hat)
            acc = bm.accuracy(y_hat)
            f1 = bm.f1_score(y_hat)
            recall = bm.recall(y_hat)
        
        # Tensorboard
        with SummaryWriter('./PHIAF_out') as writer:
            writer.add_scalars(
                'Loss',
                {
                    'train': train_loss.avg,
                    'val': val_loss
                },
                epoch)
            writer.add_scalar('auc', auc, epoch)
            writer.add_scalar('aupr', aupr, epoch)
            writer.add_scalar('acc', acc, epoch)
            writer.add_scalar('f1', f1, epoch)
            writer.add_scalar('recall', recall, epoch)
        
        if(epoch % 10 == 0):
            print_dict = {
                'epoch': epoch,
                'auc': auc,
                'aupr': aupr,
                'acc': acc,
                'f1': f1,
                'recall': recall
            }
            format_print(print_dict)
            if(epoch == epochs):
                fold_result.append(print_dict)

cals = {}

for ii in range(len(fold_result)):
    print('= = = = Fold {} = = = ='.format(ii+1))
    for k, v in fold_result[ii].items():
        if(k not in cals):
            cals[k] = AverageMeter()
        print('{0}: {1:.2f}'.format(k, v))
        cals[k].update(v)

for k, v in cals.items():
    print('{0}: {1}'.format(k, v.avg))

###################################
########## SAVE RESULTS ###########
###################################
save_root_dir = '../tune_results/'
sub_dir = inputs.subdir
os.makedirs(save_root_dir+sub_dir, exist_ok=True)

file_name = ''
for k, v in inputs.__dict__.items():
    file_name += '{0}-{1}__'.format(k, v)

with open('{0}{1}/{2}.pkl'.format(save_root_dir, sub_dir, file_name), 'wb') as f:
    pickle.dump(fold_result, f)
    print('Saved!')
