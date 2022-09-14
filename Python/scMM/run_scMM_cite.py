import argparse
import time
from joblib import Parallel, delayed
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
import gc
import numpy as np
import torch
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.utils.data import Subset, DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix
import os

os.chdir('/Users/imbi-mac-102/Desktop/MultimodalDataIntegration/scMM_neurips_subsamples/')

parser = argparse.ArgumentParser(description='scMM Hyperparameters')
parser.add_argument('--experiment', type=str, default='test', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='rna_protein', metavar='M',
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default='m_elbo_naive_warmup', metavar='O',
                    help='objective to use (default: elbo)')
parser.add_argument('--llik_scaling', type=float, default=1.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='L',
                    help='learning rate (default: 1e-3)')                   
parser.add_argument('--latent_dim', type=int, default=10, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num_hidden_layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--r_hidden_dim', type=int, default=100, 
                    help='number of hidden units in enc/dec for gene')
parser.add_argument('--p_hidden_dim', type=int, default=20, 
                    help='number of hidden units in enc/dec for protein/peak')
parser.add_argument('--pre_trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn_prior', action='store_true', default=False,
                    help='learn model prior parameters')
parser.add_argument('--analytics', action='store_true', default=True,
                    help='disable plotting analytics')
parser.add_argument('--print_freq', type=int, default=0, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset_path', type=str, default="")
parser.add_argument('--r_dim', type=int, default=1)
parser.add_argument('--p_dim', type=int, default=1)
parser.add_argument('--deterministic_warmup', type=int, default=50, metavar='W',
                    help='deterministic warmup')
#
"""
to set up arguments: from the methods part of the paper:

Model architecture and optimization
    Optimization was performed with an Adam optimizer with AMSGrad (Reddi et al., 2019). 
    Hyperparameter optimization was performed by Optuna (Akiba et al., 2019). 
    For CITE-seq data, three hidden layers with 200 hidden units were used for both modalities. 
    For SHARE-seq data, three hidden layers with 500 units for transcriptome and 100 hidden units for chromatin accessibility were used. 
    Learning rates were set to 2 × 10−3 and 1 × 10−4 for CITE-seq and SHARE-seq data, respectively. 
    Minibatch sizes of 128 and 64 were used for CITE-seq and SHARE-seq data, respectively. 
    We used a deterministic warm-up learning scheme for 25 and 50 epochs, 
        with maximum of 50 and 100 epochs for CITE-seq and SHARE-seq data, respectively (Sønderby et al., 2016). 
    After deterministic warm-up, early stopping with a tolerance of 10 epochs was applied. 
    We observed that minor changes in hyperparameters did not significantly affect the analyzed results.

Data preproccessing
    For transcriptome count data, 5000 most variable genes were first selected by applying the Seurat FindMostVariable function to log-normalized counts. 
    Raw counts were used for model input. For chromatin accessibility data, the top 25% peaks were selected for input using Seurat's FindTopFeatures function. 
    No preprocessing and feature selection were performed on surface protein count data.
"""
# args
args = parser.parse_args([])
args.experiment='rna_protein'
args.model='rna_protein'
args.obj='m_elbo_naive_warmup'
args.batch_size=200#128
args.epochs=50 # 50
args.deterministic_warmup=25 # 25
args.lr=2e-3
args.latent_dim=10
args.num_hidden_layers=3
args.r_hidden_dim=200
args.p_hidden_dim=200
args.learn_prior
args.seed = 1234
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data, EarlyStopping, Constants, log_mean_exp, is_multidata, kl_divergence
from datasets import RNA_Dataset, ATAC_Dataset

def run(i,j):
    rep=i
    print('%s cells...'%j)
    # set up run path
    runId = '/%s_cells_rep_%s'%(j, rep)
    args.seed = i*j
    if j >= 5000:
        args.lr = 2e-4
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    experiment_dir = Path('experiments/' + args.experiment + '/csv')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = str('experiments/' + args.experiment)
    args.dataset_path = '/Users/imbi-mac-102/Desktop/MultimodalDataIntegration/data/neurips-data_cite_subsamples_for_scMM/%s_cells_rep_%s'%(j, rep)
    if args.model == 'rna_atac':
        modal = 'ATAC-seq'
    elif args.model == 'rna_protein':
        modal = 'CITE-seq'
    rna_path = args.dataset_path + '/RNA-seq'
    modal_path = args.dataset_path + '/{}'.format(modal)
    r_dataset = RNA_Dataset(rna_path, transpose=True)
    args.r_dim = r_dataset.data.shape[1]
    modal_dataset = ATAC_Dataset(modal_path, transpose=True) if args.model == 'rna_atac' else RNA_Dataset(modal_path, transpose=True)
    args.p_dim = modal_dataset.data.shape[1]
    # set timers 
    cputime_begin = time.process_time()
    clocktime_begin = time.time()
    # shuffle train set
    num_cell = r_dataset.data.shape[0]
    t_size = np.round(num_cell*1.0).astype('int')
    t_id = np.random.choice(a=num_cell, size=t_size, replace=False) 
    torch.save(t_id, runPath + '/t_id_%s_cells_rep_%s.rar'%(j,rep))
    train_dataset = [Subset(r_dataset, t_id), Subset(modal_dataset, t_id)] # length 400 
    t_id = pd.DataFrame(t_id)
    t_id.to_csv('{}/t_id_{}_cells_rep_{}.csv'.format(runPath, j, rep))
    # load model
    modelC = getattr(models, 'VAE_{}'.format(args.model))
    model = modelC(args).to(device)
    torch.save(args,runPath+'/args_%s_cells_rep_%s.rar'%(j,rep))
    # Dataloader
    train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device)
    # preparation for training
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.lr, amsgrad=True)
    objective = getattr(objectives,args.obj) 
    s_objective = getattr(objectives,args.obj)
    def train(epoch, agg, W):
        model.train()
        b_loss = 0
        for i, dataT in enumerate(train_loader):
            beta = (epoch - 1) / W  if epoch <= W else 1
            if dataT[0].size()[0] == 1:
                continue
            data = [d.to(device) for d in dataT] #multimodal
            optimizer.zero_grad()
            loss = -objective(model, data, beta)
            loss.backward()
            optimizer.step()
            b_loss += loss.item()
            if args.print_freq > 0 and i % args.print_freq == 0:
                print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
        agg['train_loss'].append(b_loss / len(train_loader.dataset))
        print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))
        return b_loss
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, verbose=True) 
        W = args.deterministic_warmup
        start_early_stop = W
        for epoch in range(1, args.epochs + 1):
            b_loss = train(epoch, agg, W)
            if torch.isnan(torch.tensor([b_loss])):
                break
    if args.analytics:
        def get_latent(dataloader, train_test, runPath):
            model.eval()
            with torch.no_grad():
                if args.model == 'rna_atac':
                    modal = ['rna', 'atac'] 
                elif args.model == 'rna_protein':
                    modal = ['rna', 'protein']
                    pred = []
                for i, dataT in enumerate(dataloader):
                    data = [d.to(device) for d in dataT]
                    lats = model.latents(data, sampling=False)
                    if i == 0:
                        pred = lats
                    else:
                        for m,lat in enumerate(lats):
                            pred[m] = torch.cat([pred[m], lat], dim=0) 
                for m,lat in enumerate(pred):
                    lat = lat.cpu().detach().numpy()
                    lat = pd.DataFrame(lat)
                    lat.to_csv('{}/lat_{}_{}_{}_cells_rep_{}.csv'.format(runPath, train_test, modal[m], j, rep))
            mean_lats = sum(pred)/len(pred)
            mean_lats = mean_lats.cpu().detach().numpy()
            mean_lats = pd.DataFrame(mean_lats)
            mean_lats.to_csv('{}/csv/latent_cite_subsample_{}_cells_rep_{}.csv'.format(runPath,j, rep))
    train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)
    get_latent(train_loader, 'train', runPath)
    cputime_end = time.process_time()
    clocktime_end = time.time()
    cputime_elapsed = cputime_end - cputime_begin
    clocktime_elapsed = clocktime_end - clocktime_begin
    # save timings 
    with open(runPath+'/timings_cite_%s_cells_rep_%s.txt'%(j,i), 'w') as f: 
        f.write('clocktime: %s \n'%(clocktime_elapsed))
        f.write('cputime: %s \n'%(cputime_elapsed))
    f.close()
    del model, train_loader, optimizer
    gc.collect()

n_cells = [500, 1000, 2500, 5000, 10000]

for i in range(10):
    print(i)
    for j in n_cells:
        rep=i
        experiment_dir = Path('experiments/' + args.experiment + '/csv')
        experiment_dir.mkdir(parents=True, exist_ok=True)
        runPath = str('experiments/' + args.experiment)
        #if os.path.isfile('{}/csv/latent_cite_subsample_{}_cells_rep_{}.csv'.format(runPath,j, rep)):
        #    continue
        try:
            run(i,j)
        except ValueError: 
            with open(runPath+'/log.txt', 'a') as f: 
                f.write('error encountered for %s cells in rep %s, skipping dataset \n \n'%(i,j))
            f.close()