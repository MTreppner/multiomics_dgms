# activate virtual environment 
# source /Users/imbi-mac-102/Desktop/MultimodalDataIntegration/virtualenvs/portal-env/bin/activate

import os
import numpy as np
import pandas as pd
import torch
import time 
import random 
import anndata
import portal

os.getcwd()

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

for i in range(10):
    print('rep %s...'%i)
    for j in n_cells:
        print('%s cells...'%j)
        # 1) set seeds 
        torch.manual_seed(i*j)
        random.seed(i*j)
        np.random.seed(i*j)
        # 2) create a folder for saving results        
        output_path=os.getcwd() + '/output'
        result_path = output_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if os.path.isfile('{}/csv/latent_subsample_{}_cells_rep_{}.csv'.format(output_path, j, i)):
            continue
        # 3) load subsampled data
        print('loading data...')
        load_rna_filename = data_path_rna + '/adata_gex_subsample_%s_cells_rep_%s.h5ad'%(j,i)
        load_atac_filename = data_path_atac + '/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad'%(j,i)
        # load from h5ad files
        adata_rna_raw = anndata.read_h5ad(load_rna_filename)
        adata_atac_raw = anndata.read_h5ad(load_atac_filename)
        # preprocess RNA
        adata_rna = anndata.AnnData(adata_rna_raw.X.todense())
        adata_rna.obs.index = adata_rna_raw.obs.index
        adata_rna.obs["cell_type"] = adata_rna_raw.obs["cell_type"]
        adata_rna.obs["data_type"] = "rna"
        adata_rna.var.index = adata_rna_raw.var.index
        # preprocess ATAC
        adata_atac = anndata.AnnData(adata_atac_raw.obsm["gene_activity"].todense())
        adata_atac.obs.index = adata_atac_raw.obs.index
        adata_atac.obs["cell_type"] = adata_atac_raw.obs["cell_type"]
        adata_atac.obs["data_type"] = "atac"
        adata_atac.var.index = adata_atac_raw.uns["gene_activity_var_names"]
        # preprocess meta data
        meta_rna = adata_rna.obs
        meta_atac = adata_atac.obs
        meta = pd.concat([meta_rna, meta_atac], axis=0)
        # 4) initialise portal model
        cputime_begin = time.process_time()
        clocktime_begin = time.time()
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        if j/10 < 500:
            batch_size=int(j/10)
        else:
            batch_size=500
        model = portal.model.Model(batch_size=batch_size, training_steps=2000, lambdacos=10.0)
        model.preprocess(adata_rna, adata_atac) # perform preprocess and PCA
        # 5) train model 
        model.train()
        # 6) get integrated latent representation of cells
        model.eval()
        latent = model.latent
        latent_df = pd.DataFrame(latent)
        # save timings
        cputime_end = time.process_time()
        clocktime_end = time.time()
        cputime_elapsed = cputime_end - cputime_begin
        clocktime_elapsed = clocktime_end - clocktime_begin
        # get barcodes to dataframe
        adata_total = adata_rna.concatenate(adata_atac, index_unique=None)
        latent_df.index = adata_total.obs.index
        #portal.utils.plot_UMAP(model.latent, meta, colors=["data_type", "cell_type"], save=False, result_path=result_path)
        # save latents
        latent_df.to_csv('{}/csv/latent_subsample_{}_cells_rep_{}.csv'.format(output_path, j, i))
        # save timings 
        with open('{}/timings/timings_{}_cells_rep_{}.txt'.format(output_path, j, i), 'w') as f: 
            f.write('clocktime: %s \n'%(clocktime_elapsed))
            f.write('cputime: %s \n'%(cputime_elapsed))
        f.close()
#



# get parameters (do everything until and including model.train())

model.E_A
model.E_B
model.G_A
model.G_B
model.D_A
model.D_B
params_G = list(model.E_A.parameters()) + list(model.E_B.parameters()) + list(model.G_A.parameters()) + list(model.G_B.parameters())
params_D = list(model.D_A.parameters()) + list(model.D_B.parameters())

total_params_G = sum(param.numel() for param in params_G)
total_params_D = sum(param.numel() for param in params_D)
total_params = total_params_D + total_params_G

trainable_params_G = sum(p.numel() for p in params_G if p.requires_grad)
trainable_params_D = sum(p.numel() for p in params_D if p.requires_grad)
trainable_params = trainable_params_D + trainable_params_G

# total_params = sum(param.numel() for param in model.parameters()) and 
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
