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
from time import process_time

import random

def run_cobolt(substr):
    print(substr)
    
    adata_gex = ad.read_h5ad("adata_cite_gex_" + substr)
    adata_protein = ad.read_h5ad("adata_cite_protein_" + substr)
    
    randint = random.randint(0,100000000)

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
    
    del adata_gex, adata_protein
    
    start = timer()
    t0 = process_time() 
    
    random.seed(randint)
    
    model = Cobolt(dataset=multi_dt_neurips, lr=0.005, n_latent=10)
    model.train() # default of num_epochs is 100  
    model.calc_all_latent()
    latent = model.get_all_latent()
    
    df = pd.DataFrame(latent[0], index=latent[1])
    
    df.to_csv('latent_cite_' + substr.replace(".h5ad", "") + '.csv')  
    
    t1 = process_time()
    end = timer()
    
    extime = end - start    
    f = open("cobolt_cite_clockTimes.txt", "a")
    f.write(substr.replace(".h5ad", "") + ": " +  str(extime) + "\n")
    f.close()
    
    f = open("cobolt_cite_CPUTimes.txt", "a")
    f.write(substr.replace(".h5ad", "") + ": " +  str(t1 - t0) + "\n")
    f.close()
    
    f_seed = open("cobolt_cite_seeds.txt", "a")
    f_seed.write(substr.replace(".h5ad", "") + ": " + str(randint) + "\n")
    f_seed.close()
    
    del model


if __name__ == '__main__':
    #N= os.cpu_count() - 1
    N= 5
    
    os.chdir('../')
    # os.chdir('subsampled/')
            
    h5ad_files = glob.glob('*_cite_*.h5ad')
    
    h5ad_files_substr = list(set([item.split("_", 3)[3] for item in h5ad_files]))
        
    h5ad_files_substr.sort()


    with mp.Pool(processes = N) as p:
        results = p.map(run_cobolt, h5ad_files_substr)

