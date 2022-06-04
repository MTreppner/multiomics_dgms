import os
import numpy as np
#from scipy.io import mmread, mmwrite
import pandas as pd
import anndata

#-------------------------------------------------------------------------------------------
# RNA + ATAC-seq
#-------------------------------------------------------------------------------------------

path_to_results = "/scratch/global/maren/multiomics-data/scMM/experiments/rna_atac/"
path_to_anndata = "/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/"

for i in range(10):
    print('rep %s'%(i))
    for j in [500, 1000, 2500, 5000, 10000]:
        print('%s cells'%(j))
        print('reading data...')
        anndata_rna_filename =  path_to_anndata + 'adata_gex_subsample_%s_cells_rep_%s.h5ad'%(j,i)
        rnadata = anndata.read_h5ad(anndata_rna_filename)
        latent_from_csv = pd.read_csv(path_to_results+'/csv/latent_subsample_{}_cells_rep_{}.csv'.format(j, i), sep=',', index_col=0).to_numpy() # shape: n_cells x latent_dim
        train_inds = pd.read_csv(path_to_results+'/t_id_{}_cells_rep_{}.csv'.format(j,i), sep=',', index_col=0).to_numpy()
        print('processing data...')
        # re-order the rows of the shuffled indices 
        latent_dim = latent_from_csv.shape[1]
        latent = np.zeros((j,latent_dim), dtype=float)
        latent[train_inds.flatten(),:]=latent_from_csv
        # extract barcodes 
        latent_new_df = pd.DataFrame(data=latent, index=rnadata.obs.index)
        print('saving data...')
        latent_new_df.to_csv(path_to_results + 'csv_barcodes/latent_subsample_{}_cells_rep_{}.csv'.format(j, i), sep=',', index=True)
#

#-------------------------------------------------------------------------------------------
# CITE-seq
#-------------------------------------------------------------------------------------------

path_to_results = "/scratch/global/maren/multiomics-data/scMM/experiments/rna_protein/"
path_to_anndata = "/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/"

for i in range(10):
    print('rep %s'%(i))
    for j in [500, 1000, 2500, 5000, 10000]:
        print('%s cells'%(j))
        print('reading data...')
        anndata_rna_filename =  path_to_anndata + 'adata_cite_gex_subsample_%s_cells_rep_%s.h5ad'%(j,i)
        rnadata = anndata.read_h5ad(anndata_rna_filename)
        try:
            latent_from_csv = pd.read_csv(path_to_results+'/csv/latent_cite_subsample_{}_cells_rep_{}.csv'.format(j, i), sep=',', index_col=0).to_numpy() # shape: n_cells x latent_dim
            train_inds = pd.read_csv(path_to_results+'/t_id_{}_cells_rep_{}.csv'.format(j,i), sep=',', index_col=0).to_numpy()
            print('processing data...')
            # re-order the rows of the shuffled indices 
            latent_dim = latent_from_csv.shape[1]
            latent = np.zeros((j,latent_dim), dtype=float)
            latent[train_inds.flatten(),:]=latent_from_csv
            # extract barcodes 
            latent_new_df = pd.DataFrame(data=latent, index=rnadata.obs.index)
            print('saving data...')
            latent_new_df.to_csv(path_to_results + 'csv_barcodes/latent_cite_subsample_{}_cells_rep_{}.csv'.format(j, i), sep=',', index=True)
        except:
            continue
#