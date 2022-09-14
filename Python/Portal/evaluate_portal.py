# activate virtual environment 
# source /Users/imbi-mac-102/Desktop/MultimodalDataIntegration/virtualenvs/portal-env/bin/activate

import os
import numpy as np
import pandas as pd
import torch
import random 
import anndata
import scanpy as sc

def preprocess(adata_A_input,
            adata_B_input, 
            hvg_num=4000,
            ):
    '''
    Performing preprocessing for a pair of datasets.
    '''
    adata_A = adata_A_input.copy()
    adata_B = adata_B_input.copy()

    print("Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata_A, flavor='seurat_v3', n_top_genes=hvg_num, check_values=False)
    sc.pp.highly_variable_genes(adata_B, flavor='seurat_v3', n_top_genes=hvg_num, check_values=False)
    hvg_A = adata_A.var[adata_A.var.highly_variable == True].sort_values(by="highly_variable_rank").index
    hvg_B = adata_B.var[adata_B.var.highly_variable == True].sort_values(by="highly_variable_rank").index
    hvg_total = hvg_A & hvg_B
    if len(hvg_total) < 100:
        raise ValueError("The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(hvg_total))

    print("Normalizing and scaling...")
    sc.pp.normalize_total(adata_A, target_sum=1e4)
    sc.pp.log1p(adata_A)
    adata_A = adata_A[:, hvg_total]
    sc.pp.scale(adata_A, max_value=10)

    sc.pp.normalize_total(adata_B, target_sum=1e4)
    sc.pp.log1p(adata_B)
    adata_B = adata_B[:, hvg_total]
    sc.pp.scale(adata_B, max_value=10)

    adata_total = adata_A.concatenate(adata_B, index_unique=None)
    return adata_total


os.getcwd()
os.chdir("/Users/imbi-mac-102/Desktop/MultimodalDataIntegration/portal/")

data_path_rna='/Users/imbi-mac-102/Desktop/MultimodalDataIntegration/data/subsampled'
data_path_atac='/Users/imbi-mac-102/Desktop/MultimodalDataIntegration/data/neurips_data_atac_top_peaks'

n_cells = [500, 1000, 2500, 5000, 10000]

i=0
j=1000

torch.manual_seed(i*j)
random.seed(i*j)
np.random.seed(i*j)
output_path=os.getcwd() + '/output'
result_path = output_path
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
# combine datasets 
adata_comb = preprocess(adata_rna, adata_atac)
embedding = pd.read_csv('{}/csv/latent_subsample_{}_cells_rep_{}.csv'.format(output_path, j, i))
embedding.index = embedding.iloc[:,0]
embedding = embedding.iloc[:,1:]
(adata_comb.obs.index == embedding.index).sum() # sould be 2000 
# add metadata 
cell_type = pd.concat([adata_rna_raw.obs["cell_type"], adata_atac_raw.obs["cell_type"]], axis=0)
batch = pd.concat([adata_rna_raw.obs["batch"], adata_atac_raw.obs["batch"]], axis=0)
pseudotime = pd.concat([adata_rna_raw.obs["pseudotime_order_GEX"], adata_atac_raw.obs["pseudotime_order_ATAC"]], axis=0)
adata_comb.obs["cell_type"] = cell_type
adata_comb.obs["batch"] = batch
adata_comb.obs["pseudotime"] = pseudotime
adata_comb.obs = adata_comb.obs.reset_index()
adata_comb.obs.index = [str(i) for i in adata_comb.obs.index]
combined_dat = adata_comb.copy()
# create combined adata object 
combined_dat_int = combined_dat.copy()
combined_dat_int.obsm['X_emb'] = embedding.to_numpy()
sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
sc.tl.leiden(combined_dat_int)
label_key = 'cell_type'
batch_key = 'batch'

import scib
eval = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, 
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        pcr_=False,
        isolated_labels_f1_=False,
        nmi_=True,
        ari_=True,
        graph_conn_=True,
        embed = 'X_emb'
)
trajectory_score = scib.metrics.trajectory_conservation(
            combined_dat,
            combined_dat_int,
            label_key=label_key,
            batch_key = batch_key,
            pseudotime_key = 'pseudotime'
)