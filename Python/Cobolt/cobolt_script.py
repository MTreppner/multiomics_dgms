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
    
    #adata_atac = ad.read_h5ad("adata_atac_" + substr)
    adata_gex = ad.read_h5ad("adata_gex_" + substr)
    adata_atac_topPeaks = ad.read_h5ad("adata_atac_subsample_top_peaks" + substr.split("subsample", 1)[1])
    
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
    
    model = Cobolt(dataset=multi_dt_neurips, lr=0.005, n_latent=10)
    model.train() # default of num_epochs is 100
    
    model.calc_all_latent()
    
    
    latent = model.get_all_latent()
    
    end = timer()
    
    df = pd.DataFrame(latent[0], index=latent[1])
    
    df.to_csv('latent_' + substr.replace(".h5ad", "") + '.csv')  
    
    extime = end - start
    f = open("executionTimes.txt", "a")
    f.write(substr.replace(".h5ad", "") + ": " +  str(extime) + "\n")
    f.close()


if __name__ == '__main__':
    #N= os.cpu_count() - 1
    N= 10

    os.chdir('../subsampled/')
            
    # # h5ad_files = glob.glob('./*.h5ad')
    h5ad_files = glob.glob('*.h5ad')
    
    
    h5ad_files_substr = [item.split("_", 2)[2] for item in h5ad_files]
    h5ad_files_substr = list(set([item.replace("_top_peaks", "") for item in h5ad_files_substr]))
    
    
            # already_done = list("subsample_2500_cells_rep_3.h5ad",  "subsample_2500_cells_rep_8.h5ad","subsample_2500_cells_rep_9.h5ad",
            # "subsample_1000_cells_rep_8.h5ad", "subsample_1000_cells_rep_3.h5ad", "subsample_1000_cells_rep_1.h5ad",
            # "subsample_1000_cells_rep_0.h5ad", "subsample_1000_cells_rep_6.h5ad", "subsample_1000_cells_rep_9.h5ad",
            # "subsample_500_cells_rep_7.h5ad",  "subsample_500_cells_rep_4.h5ad",  "subsample_500_cells_rep_1.h5ad",
            # "subsample_500_cells_rep_3.h5ad",  "subsample_500_cells_rep_8.h5ad",  "subsample_500_cells_rep_6.h5ad",
            # "subsample_500_cells_rep_0.h5ad")
            # h5ad_files_substr = list(set(h5ad_files_substr) - set(already_done))
            
            # h5ad_files_substr = ["subsample_10000_cells_rep_1.h5ad", "subsample_10000_cells_rep_6.h5ad",
            #              "subsample_10000_cells_rep_7.h5ad"]

    with mp.Pool(processes = N) as p:
        results = p.map(run_cobolt, h5ad_files_substr)




    

# First round resulted in the following error message to teh time when the 10000 cell files were still running:

# Traceback (most recent call last):
#   File "/usr/lib/python3.7/multiprocessing/pool.py", line 121, in worker
#     result = (True, func(*args, **kwds))
#   File "/usr/lib/python3.7/multiprocessing/pool.py", line 44, in mapstar
#     return list(map(*args))
#   File "cobolt_script.py", line 62, in run_cobolt
#     model.train() # default of num_epochs is 100
#   File "/dsk/scratch/brombach/multiomics/cobolt/cobolt/model/cobolt.py", line 139, in train
#     raise ValueError("DIVERGED. Try a smaller learning rate.")
# ValueError: DIVERGED. Try a smaller learning rate.
# """

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "cobolt_script.py", line 96, in <module>
#     results = p.map(run_cobolt, h5ad_files_substr)
#   File "/usr/lib/python3.7/multiprocessing/pool.py", line 268, in map
#     return self._map_async(func, iterable, mapstar, chunksize).get()
#   File "/usr/lib/python3.7/multiprocessing/pool.py", line 657, in get
#     raise self._value
# ValueError: DIVERGED. Try a smaller learning rate.
    
