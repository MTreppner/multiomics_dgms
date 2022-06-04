import scvi
import anndata as ad
import scipy as sp
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gc
import multiprocessing
from tqdm import tqdm
from timeit import default_timer as timer

scvi.settings.seed = 420
def process(i, j):
    # read multiomic data
    adata_atac = sc.read("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
    adata_atac.var_names_make_unique()

    # Filter HVGs from gex data set
    adata_atac = adata_atac[:,adata_atac.var['highly_variable'] == True].copy()

    adata_rna = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
    adata_rna.var_names_make_unique()

    # Search for common barcodes (cells)
    common_barcodes = list(set(adata_rna.obs_names).intersection(adata_atac.obs_names))
    adata_rna_sub = adata_rna[common_barcodes].copy()
    adata_atac_sub = adata_atac[common_barcodes].copy()

    # Concatenate data sets 
    combined_dat = ad.concat([adata_rna_sub,adata_atac_sub], axis=1)
    combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
    combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
    combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
    combined_dat.obs["pseudotime_order_ATAC"] = adata_rna_sub.obs["pseudotime_order_ATAC"]
    combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
    combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
    combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
    combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
    combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Peaks",) * adata_atac_sub.shape[1])], axis=0).values
    combined_dat.var["highly_variable"] = pd.concat([adata_rna_sub.var["highly_variable"],adata_atac_sub.var["highly_variable"]], axis=0).values
        
    del adata_atac
    del adata_rna

    # Sort features 
    combined_dat = combined_dat[:, combined_dat.var["feature_types"].argsort()].copy()
    combined_dat.var

    # Start timer
    start = timer()

    # Initialize model
    scvi.model.MULTIVI.setup_anndata(combined_dat, batch_key='modality')

    mvi = scvi.model.MULTIVI(
        combined_dat,
        n_genes=(combined_dat.var['modality']=='Gene Expression').sum(),
        n_regions=(combined_dat.var['modality']=='Peaks').sum(),
        n_latent=10
    )

    # Train model
    mvi.train(max_epochs=100)

    # Save model
    mvi.save("/scratch/global/treppner/multi_omics_data/models/multivi_%s_cells_rep_%s" % (j,i))

    # Get latent representations from trained model
    latent = mvi.get_latent_representation()
    latent = pd.DataFrame(latent)     
        
    # Save to csv
    latent.to_csv("/scratch/global/treppner/multi_omics_data/neurips_data/embeddings/multivi_multiome_latent_embedding_%s_cells_rep_%s.csv" % (j,i))

    # End timer
    end = timer()

    extime = end - start
    f = open("/scratch/global/treppner/multi_omics_data/neurips_data/embeddings/executionTimes.txt", "a")
    f.write("multivi_time=%s_%s_cells_rep_%s\n" % (extime,j,i))
    f.close()

    del combined_dat
    del mvi
    gc.collect()
#

num_cores = 6
inputs = tqdm(range(10))

# Ugly parallelization due to memory issues
n_cells = 500
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(process)(i,n_cells) for i in inputs)

n_cells = 1000
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(process)(i,n_cells) for i in inputs)

n_cells = 2500
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(process)(i,n_cells) for i in inputs)

n_cells = 5000
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=4)(delayed(process)(i,n_cells) for i in inputs)

n_cells = 10000
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=2)(delayed(process)(i,n_cells) for i in inputs)


# Add cell names to latent reps
n_cells = [500, 1000, 2500, 5000, 10000]
scvi.settings.seed = 420
for j in n_cells:
    for i in range(10):
        # read multiomic data
        adata_atac = sc.read("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
        adata_atac.var_names_make_unique()
        # Filter HVGs from gex data set
        adata_atac = adata_atac[:,adata_atac.var['highly_variable'] == True].copy()
        adata_rna = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
        adata_rna.var_names_make_unique()
        # Search for common barcodes (cells)
        common_barcodes = list(set(adata_rna.obs_names).intersection(adata_atac.obs_names))
        adata_rna_sub = adata_rna[common_barcodes].copy()
        adata_atac_sub = adata_atac[common_barcodes].copy()
        # Concatenate data sets 
        combined_dat = ad.concat([adata_rna_sub,adata_atac_sub], axis=1)
        combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
        combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
        combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
        combined_dat.obs["pseudotime_order_ATAC"] = adata_rna_sub.obs["pseudotime_order_ATAC"]
        combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
        combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
        combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
        combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
        combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Peaks",) * adata_atac_sub.shape[1])], axis=0).values
        combined_dat.var["highly_variable"] = pd.concat([adata_rna_sub.var["highly_variable"],adata_atac_sub.var["highly_variable"]], axis=0).values
        # Sort features 
        combined_dat = combined_dat[:, combined_dat.var["feature_types"].argsort()].copy()
        combined_dat.var
        # Initialize model
        scvi.model.MULTIVI.setup_anndata(combined_dat, batch_key='modality')        
        # Load model
        vae = scvi.model.MULTIVI.load("/scratch/global/treppner/multi_omics_data/models/multivi_%s_cells_rep_%s" % (j,i),combined_dat)
        # Read csv
        embedding = vae.get_latent_representation()
        embedding = pd.DataFrame(embedding)
        embedding.index = vae.adata.obs_names
        print(j,i)
        # Save to csv
        embedding.to_csv("/scratch/global/treppner/multi_omics_data/multi_omics_review/multivi/csv/latent_subsample_%s_cells_rep_%s.csv" % (j,i),index=True)
        