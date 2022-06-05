# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
sys.path.append("/Users/admin/cobolt")


import cobolt
from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import os


###########################################

import anndata as ad
import glob
import pandas as pd
import numpy as np
import multiprocessing as mp

from timeit import default_timer as timer


def run_cobolt(substr):
    print(substr)
    
    adata_gex = ad.read_h5ad("adata_cite_gex_" + substr)
    adata_protein = ad.read_h5ad("adata_cite_protein_" + substr)
    
    start = timer()
    cite_gex = SingleData(feature_name="GEX", 
                            dataset_name="CITE", 
                            feature=adata_gex.var.index, 
                            count=adata_gex.X, 
                            barcode=adata_gex.obs.index)
 
    cite_protein = SingleData(feature_name="Protein", 
                            dataset_name="CITE", 
                            feature= adata_protein.var.index,         #adata_atac.var.index, 
                            count= adata_protein.X,              #adata_atac.X, 
                            barcode=adata_protein.obs.index)
    
    multi_dt_neurips = MultiomicDataset.from_singledata(
        cite_gex, cite_protein)
    print(multi_dt_neurips)
    
    model = Cobolt(dataset=multi_dt_neurips, lr=0.005, n_latent=10)
    model.train() # default of num_epochs is 100
    
    model.calc_all_latent()
    
    
    latent = model.get_all_latent()
    
    end = timer()
    
    df = pd.DataFrame(latent[0], index=latent[1])
    
    df.to_csv('latent_cite_' + substr.replace(".h5ad", "") + '.csv')  
    
    extime = end - start
    f = open("executionTimes_cite.txt", "a")
    f.write(substr.replace(".h5ad", "") + ": " +  str(extime) + "\n")
    f.close()


if __name__ == '__main__':
    #N= os.cpu_count() - 1
    N= 5

    os.chdir('../subsampled/')
            
    h5ad_files = glob.glob('*_cite_*.h5ad')
    
    
    h5ad_files_substr = list(set([item.split("_", 3)[3] for item in h5ad_files]))

    with mp.Pool(processes = N) as p:
        results = p.map(run_cobolt, h5ad_files_substr)

