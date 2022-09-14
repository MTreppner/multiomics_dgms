# activate virtual environment 
# source /Users/imbi-mac-102/Desktop/MultimodalDataIntegration/virtualenvs/scbean-env/bin/activate

import os
import torch 
import time 
import random 
import scanpy as sc
import scbean.model.davae as davae
import scbean.tools.utils as tl
import matplotlib
import anndata
import numpy as np
import ttools as tool
import umap
import pandas as pd
matplotlib.use('TkAgg')

torch.set_num_threads(40) # do not use all CPU threads

SERVER = False # toggle 
if SERVER:
    data_path_rna='/scratch/global/treppner/multi_omics_data/neurips_data/subsampled'
    data_path_atac='/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks'
else:
    #data_path_rna='/Users/multiomics-data/neurips_data/subsampled'
    #data_path_atac='/Users/scratch/multiomics-data/neurips_data_atac_top_peaks'
    data_path_rna='/Users/imbi-mac-102/Desktop/MultimodalDataIntegration/data/subsampled'
    data_path_atac='/Users/imbi-mac-102/Desktop/MultimodalDataIntegration/data/neurips_data_atac_top_peaks'
#
n_cells = [500, 1000, 2500, 5000, 10000]
# run on subsamples
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
        adata_rna_raw = anndata.read_h5ad(load_rna_filename)
        adata_atac_raw = anndata.read_h5ad(load_atac_filename)
        # create atac object from gene activity 
        adata_atac_genes = anndata.AnnData(adata_atac_raw.obsm["gene_activity"])
        adata_atac_genes.var.index = adata_atac_raw.uns["gene_activity_var_names"]
        adata_atac_genes.obs = adata_atac_raw.obs
        adata_atac_genes.obsm = adata_atac_raw.obsm
        # 4) set timers 
        cputime_begin = time.process_time()
        clocktime_begin = time.time()
        # 5) DAVAE preprocessing 
        adata1 = adata_rna_raw
        adata2 = adata_atac_genes
        adata_all = tl.davae_preprocessing([adata1, adata2], n_top_genes=2000, hvg=False, lognorm=False)
        adata_all.obsm["batch_label"] = np.array(["rna"]*j + ["atac"]*j)
        # sc.pp.scale(adata_all)
        # print(adata_all)
        # 6) train model
        adata_integrate = davae.fit_integration(
            adata_all,
            # mode='DAVAE',
            batch_num=2,
            batch_size=100, # 128 bad for 2500 cells --> 2*2500 % 128 = 8
            split_by='batch_label',
            domain_lambda=6.0,
            epochs=100,
            sparse=True,
            hidden_layers=[128, 64, 32, 10]
        )
        # 7) get latent 
        latent = adata_integrate.obsm["X_davae"]
        latent_df = pd.DataFrame(latent)
        latent_df.index = adata_all.obs.index
        # 8) get timings
        cputime_end = time.process_time()
        clocktime_end = time.time()
        cputime_elapsed = cputime_end - cputime_begin
        clocktime_elapsed = clocktime_end - clocktime_begin
        # 9) save latents
        latent_df.to_csv('{}/csv/latent_subsample_{}_cells_rep_{}.csv'.format(output_path, j, i))
        # 10) save timings 
        with open('{}/timings/timings_{}_cells_rep_{}.txt'.format(output_path, j, i), 'w') as f: 
            f.write('clocktime: %s \n'%(clocktime_elapsed))
            f.write('cputime: %s \n'%(cputime_elapsed))
        f.close()
#

# viz 
#sc.pp.neighbors(adata_integrate, use_rep='X_davae')
#sc.tl.umap(adata_integrate)
#sc.pl.umap(adata_integrate, color=['batch_label'], s=3, cmap='tab20c')
