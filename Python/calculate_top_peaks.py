import anndata
import numpy as np
from scipy.io import mmread, mmwrite
import scanpy as sc
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd

#----------------------------------------------------------------------------------------------------------------
# Calculate top peaks in ATAC data and save
#----------------------------------------------------------------------------------------------------------------

"""
Procedure adapted the FindTopFeatures function from Signac (v.1.7.0):

    FindTopFeatures.default <- function(
        object,
        assay = NULL,
        min.cutoff = "q5",
        verbose = TRUE,
        ...) 
    {
        featurecounts <- rowSums(x = object)
        e.dist <- ecdf(x = featurecounts)
        hvf.info <- data.frame(
            row.names = names(x = featurecounts),
            count = featurecounts,
            percentile = e.dist(featurecounts)
        )
        hvf.info <- hvf.info[order(hvf.info$count, decreasing = TRUE), ]
        return(hvf.info)
    }
    
from Github repo : https://github.com/timoast/signac/blob/63d61bed2d6f43c7fd483b77eef0c84aa70952c9/R/preprocessing.R#L224
"""

percent_cutoff = 0.75

load_path = '/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/'
save_path = '/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/'

n_cells = [500, 1000, 2500, 5000, 10000]

for i in range(10):
    print('rep %s...'%i)
    for j in n_cells:
        print('%s cells...'%j)
        # 1) load subsampled data
        print('loading data...')
        atacdata = anndata.read_h5ad(load_path + 'adata_atac_subsample_%s_cells_rep_%s.h5ad'%(j,i))
        # 2) extract highly variable peaks according to set percent_cutoff
        print('calcuating top peaks...')
        featurecounts = atacdata.layers['counts'].sum(axis=0) # featurecounts <- rowSums(x = object) # returns vector of length features 
        # turn into array to calculate ECDF and percentiles
        edist = ECDF(np.squeeze(np.asarray(featurecounts))) # e.dist <- ecdf(x = featurecounts)
        percentiles = np.squeeze(np.asarray(edist(featurecounts)))
        # make dataframe with feature names
        atac_features = atacdata.var["feature_types"].keys()
        df = pd.DataFrame(percentiles, columns=['percentile'], index =atac_features)
        # keep top 25% 
        df['hvp'] = df['percentile'] >= percent_cutoff
        keep_features=df.index[df['hvp']]
        # sort according to percentiles 
        df_sorted = df.sort_values(by=['percentile'], ascending=False)
        # 3) save as additional columns in .var to atacdata object
        print('save new adata object...')
        atacdata.var['highly_variable']=df['hvp']
        atacdata.var['highly_variable_percentile']=df['percentile']
        atacdata.write_h5ad(save_path + 'adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad'%(j,i))
#