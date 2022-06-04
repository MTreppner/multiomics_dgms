# activate virtual environment 

import os
import numpy as np
import pandas as pd
import time

os.chdir('/scratch/global/maren/multiomics-data/scMVP/')

from dataset import LoadData,GeneExpressionDataset, CellMeasurement
from models import VAE_Attention, Multi_VAE_Attention, VAE_Peak_SelfAttention
from inference import UnsupervisedTrainer 
# in multi_inference.py: error because of deprecated version of scikit-learn. 
# To get rid of the issue, change line "from sklearn.utils.linear_assignment_ import linear_assignment" in original code to the following: 
# from scipy.optimize import linear_sum_assignment as linear_assignment
# see here: https://stackoverflow.com/questions/62390517/no-module-named-sklearn-utils-linear-assignment
from inference import MultiPosterior, MultiTrainer
from scMVP_anndata_dataloader import LoadFromAnnData
import torch

import scanpy as sc
import anndata

import scipy.io as sp_io
from scipy.sparse import csr_matrix, issparse

torch.set_num_threads(40) # do not use all CPU threads

data_path_rna='/scratch/global/treppner/multi_omics_data/neurips_data/subsampled'
data_path_atac='/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks'

n_cells = [500, 1000, 2500, 5000, 10000]

import random

for i in range(10):
    print('rep %s...'%i)
    for j in n_cells:
        print('%s cells...'%j)
        # 1) set seeds 
        torch.manual_seed(i*j)
        random.seed(i*j)
        np.random.seed(i*j)
        # 2) set paths 
        output_path=os.getcwd() + '/output'
        if os.path.isfile('{}/csv/latent_subsample_{}_cells_rep_{}.csv'.format(output_path, j, i)):
            continue
        # 3) load subsampled data
        print('loading data...')
        load_rna_filename = data_path_rna + '/adata_gex_subsample_%s_cells_rep_%s.h5ad'%(j,i)
        load_atac_filename = data_path_atac + '/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad'%(j,i)
        rnadata = anndata.read_h5ad(load_rna_filename)
        atacdata = anndata.read_h5ad(load_atac_filename)
        # 4) set timers 
        cputime_begin = time.process_time()
        clocktime_begin = time.time()
        # 5) prepare data 
        dataset = LoadFromAnnData(rnadata=rnadata, atacdata=atacdata, atac_threshold=0.001, cell_threshold=1)
        # 6) set training hyperparameters
        n_epochs = 30
        if j < 5000:
            lr = 5e-4
        else:
            lr = 5e-4#3
        use_batches = False
        use_cuda = False # False if using CPU
        n_centroids = 10
        n_alfa = 1.0
        # 7) define model
        multi_vae = Multi_VAE_Attention(dataset.nb_genes, len(dataset.atac_names), n_batch=0, n_latent=10, n_centroids=n_centroids, n_alfa = n_alfa, mode="mm-vae") # should provide ATAC num, alfa, mode and loss type
        trainer = MultiTrainer(
            multi_vae,
            dataset,
            train_size=1.0,
            use_cuda=use_cuda,
            frequency=5,
        )
        # 8) define model
        trainer.train(n_epochs=n_epochs, lr=lr)
        # 9) extract and save latent representations 
        full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)),type_class=MultiPosterior)
        latent, latent_rna, latent_atac, cluster_gamma, cluster_index, batch_indices, labels = full.sequential().get_latent() 
        latent_df = pd.DataFrame(latent)
        # 10) re-order cells so that they match the barcodes 
        reordered_inds = np.array([np.argwhere(el==np.array(dataset.barcodes))[0,0] for el in rnadata.obs_names])
        latent_df = latent_df.loc[reordered_inds,:]
        assert((dataset.barcodes[reordered_inds] == rnadata.obs_names).sum() == j)
        latent_df.index = rnadata.obs_names
        latent_rna_df = pd.DataFrame(latent_rna)
        latent_atac_df = pd.DataFrame(latent_atac)
        latent_rna_df.index = rnadata.obs_names
        latent_atac_df.index = rnadata.obs_names
        # 11) get timings
        cputime_end = time.process_time()
        clocktime_end = time.time()
        cputime_elapsed = cputime_end - cputime_begin
        clocktime_elapsed = clocktime_end - clocktime_begin
        # 12) save latents
        latent_df.to_csv('{}/csv/latent_subsample_{}_cells_rep_{}.csv'.format(output_path, j, i))
        latent_rna_df.to_csv('{}/rna/latent_rna_{}_cells_rep_{}.csv'.format(output_path, j, i))
        latent_atac_df.to_csv('{}/atac/latent_atac_{}_cells_rep_{}.csv'.format(output_path, j, i))
        # 13) save timings 
        with open('{}/timings/timings_{}_cells_rep_{}.txt'.format(output_path, j, i), 'w') as f: 
            f.write('clocktime: %s \n'%(clocktime_elapsed))
            f.write('cputime: %s \n'%(cputime_elapsed))
        f.close()
#