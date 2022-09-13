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
    
    #adata_atac = ad.read_h5ad("adata_atac_" + substr)
    adata_gex = ad.read_h5ad("adata_gex_" + substr)
    adata_atac_topPeaks = ad.read_h5ad("adata_atac_subsample_top_peaks" + substr.split("subsample", 1)[1])
    
    randint = random.randint(0,100000000)
    
    start = timer()
    multiome_gex = SingleData(feature_name="GEX", 
                            dataset_name="Multiome", 
                            feature=adata_gex.var.index, 
                            count=adata_gex.X, 
                            barcode=adata_gex.obs.index)
    
    
    # len(adata_atac_topPeaks.var.index[adata_atac_topPeaks.var.highly_variable == True])

    highly_variable_index = np.where(adata_atac_topPeaks.var.highly_variable == True)[0]   
    count_topPeaks = adata_atac_topPeaks.X[:,highly_variable_index]
 
    multiome_atac = SingleData(feature_name="ATAC", 
                            dataset_name="Multiome", 
                            feature= adata_atac_topPeaks.var.index[highly_variable_index],         #adata_atac.var.index, 
                            count= count_topPeaks,              #adata_atac.X, 
                            barcode=adata_atac_topPeaks.obs.index)
    
    multi_dt_neurips = MultiomicDataset.from_singledata(
        multiome_gex, multiome_atac)
    print(multi_dt_neurips)
    
    del adata_gex, adata_atac_topPeaks
    
    start = timer()
    t0 = process_time() 
    
    random.seed(randint)
    
    model = Cobolt(dataset=multi_dt_neurips, lr=0.005, n_latent=10)
    model.train() # default of num_epochs is 100
    model.calc_all_latent()
    latent = model.get_all_latent()
    
    df = pd.DataFrame(latent[0], index=latent[1])
    
    df.to_csv('latent_' + substr.replace(".h5ad", "") + '.csv')  
    
    t1 = process_time()
    end = timer()
    
    extime = end - start    
    f = open("cobolt_multiome_clockTimes.txt", "a")
    f.write(substr.replace(".h5ad", "") + ": " +  str(extime) + "\n")
    f.close()
    
    f = open("cobolt_multiome_CPUTimes.txt", "a")
    f.write(substr.replace(".h5ad", "") + ": " +  str(t1 - t0) + "\n")
    f.close()
    
    f_seed = open("cobolt_multiome_seeds.txt", "a")
    f_seed.write(substr.replace(".h5ad", "") + ": " + str(randint) + "\n")
    f_seed.close()
    
    del model

if __name__ == '__main__':
    #N= os.cpu_count() - 1
    N= 5

    os.chdir('../')
    # os.chdir('subsampled/')
            
    # # h5ad_files = glob.glob('./*.h5ad')
    h5ad_files = glob.glob('*.h5ad')
    h5ad_files = [item for item in h5ad_files if '_cite_' not in item]
    
    h5ad_files_substr = [item.split("_", 2)[2] for item in h5ad_files]
    h5ad_files_substr = list(set([item.replace("_top_peaks", "") for item in h5ad_files_substr]))
    
    h5ad_files_substr.sort()
    
    with mp.Pool(processes = N) as p:
        results = p.map(run_cobolt, h5ad_files_substr)


