#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:54:44 2022

@author: admin
"""

# import sys


# Use development version of SCALEX at https://github.com/ericli0419/SCALEX
import scalex
from scalex import SCALEX
import os

###########################################

import scanpy as sc
import anndata as ad
import glob
import pandas as pd
import numpy as np
# import multiprocessing as mp
# import scipy as sp

from timeit import default_timer as timer

def run_scalex(substr):
    print(substr)
    
    adata_gex = ad.read_h5ad("adata_cite_gex_" + substr)
    adata_protein = ad.read_h5ad("adata_cite_protein_" + substr)

    # RNA adata preprocess
    sc.pp.normalize_per_cell(adata_gex, counts_per_cell_after=1e4)
    sc.pp.log1p(adata_gex)
    
    # Protein adata preprocess
    sc.pp.normalize_per_cell(adata_protein, counts_per_cell_after=1e4)
    sc.pp.log1p(adata_protein)
    
    # Concatenate RNA and protein adata
    norm_full_data = ad.AnnData(
        X=np.concatenate(
            [
                # concatenate dense matrices as np.concatenate can not be used on sparse matrices
                adata_gex.X.todense(),
                adata_protein.X.todense(),
            ],
            axis=1,
        )
    )

    col_names = np.concatenate([adata_gex.var_names, adata_protein.var_names])
    norm_full_data.var_names = col_names
    norm_full_data.var_names_make_unique()
    
    norm_full_data.obs_names = adata_protein.obs_names
    norm_full_data.write('scalex_CITE_' + substr.replace(".h5ad", "") + '.h5ad')
    
    
    
    start = timer()
        
    # SCALEX Integration
    adata_scalex=SCALEX('scalex_CITE_' + substr.replace(".h5ad", "") + '.h5ad',
                        batch_name='batch',
                        min_features=1,
                        min_cells=1, 
                        outdir='scalex_CITE_' + substr.replace(".h5ad", "") + '_log_output/', 
                        show=False, 
                        # gpu=9,
                        batch_size=64, 
                        # max_iteration=100, Default: 30000 , processing time decreases if decreasing this value
                        ignore_umap = False,
                        verbose = True,
                        processed=True)
    
    latent = adata_scalex.obsm['latent']
    
    end = timer()
    
    
    
    df = pd.DataFrame(latent, index=adata_scalex.obsm.dim_names)
    
    df.to_csv('scalex_latent_cite_' + substr.replace(".h5ad", "") + '.csv')  
    
    extime = end - start
    f = open("scalex_executionTimes_cite.txt", "a")
    f.write(substr.replace(".h5ad", "") + ": " +  str(extime) + "\n")
    f.close()


if __name__ == '__main__':
    #N= os.cpu_count() - 1
    # N= 5

    os.chdir('subsampled/')
            
    h5ad_files = glob.glob('*_cite_*.h5ad')
    
    
    h5ad_files_substr = list(set([item.split("_", 3)[3] for item in h5ad_files]))
    
    for substr in h5ad_files_substr:
        run_scalex(substr)

    # with mp.Pool(processes = N) as p:
    #    results = p.map(run_scalex, h5ad_files_substr)




