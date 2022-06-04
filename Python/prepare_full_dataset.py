import anndata as ad
import scipy as sp
import numpy as np
import pandas as pd
import scanpy as sc
from statsmodels.distributions.empirical_distribution import ECDF

#----------------------------------------------------------------------------------------------------------------
# Multiome data preparation
#----------------------------------------------------------------------------------------------------------------

load_path = "/scratch/global/treppner/multiomics-data/neurips_data/"
save_path = "/scratch/global/maren/multiomics-data/full_dataset/"

# Read data from phase 2 of the competition
adata_gex = ad.read_h5ad(load_path + "openproblems_bmmc_multiome_phase2.manual_formatting.output_rna.h5ad")
adata_atac = ad.read_h5ad(load_path + "openproblems_bmmc_multiome_phase2.manual_formatting.output_mod2.h5ad")

# Compute HVGs
sc.pp.highly_variable_genes(adata_gex, flavor='cell_ranger', n_top_genes=4000)

# Filter HVGs from gex data set
adata_gex = adata_gex[:,adata_gex.var['highly_variable'] == True].copy()

# Save
adata_gex.write_h5ad(save_path + 'adata_gex_hvg_full.h5ad')

# Compute top peaks in ATAC data 
percent_cutoff = 0.75
featurecounts = adata_atac.layers['counts'].sum(axis=0) # featurecounts <- rowSums(x = object) # returns vector of length features 
# turn into array to calculate ECDF and percentiles
edist = ECDF(np.squeeze(np.asarray(featurecounts))) # e.dist <- ecdf(x = featurecounts)
percentiles = np.squeeze(np.asarray(edist(featurecounts)))
# make dataframe with feature names
atac_features = adata_atac.var["feature_types"].keys()
df = pd.DataFrame(percentiles, columns=['percentile'], index =atac_features)
# keep top 25% 
df['hvp'] = df['percentile'] >= percent_cutoff
keep_features=df.index[df['hvp']]
# sort according to percentiles 
df_sorted = df.sort_values(by=['percentile'], ascending=False)
# save as additional columns in .var to atacdata object
adata_atac.var['highly_variable']=df['hvp']
adata_atac.var['highly_variable_percentile']=df['percentile']

# Save as new adata object
adata_atac.write_h5ad(save_path + 'adata_atac_top_peaks_full.h5ad')

#----------------------------------------------------------------------------------------------------------------
# CITE-Seq data preparation
#----------------------------------------------------------------------------------------------------------------

# Read data from phase 2 of the competition
adata_cite_gex = ad.read_h5ad(load_path + 'cite/openproblems_bmmc_cite_phase2.manual_formatting.output_rna.h5ad')
adata_protein = ad.read_h5ad(load_path + 'cite/openproblems_bmmc_cite_phase2.manual_formatting.output_mod2.h5ad')

# Compute HVGs
sc.pp.highly_variable_genes(adata_cite_gex, flavor='cell_ranger', n_top_genes=4000)
#sc.pl.highly_variable_genes(adata_cite_gex)

# Filter HVGs from gex data set
adata_cite_gex = adata_cite_gex[:,adata_cite_gex.var['highly_variable'] == True].copy()

# Save as new adata object 
adata_cite_gex.write_h5ad(filename=save_path+'adata_cite_gex_hvg_full.h5ad')
adata_protein.write_h5ad(filename=save_path + 'adata_cite_protein_full.h5ad')