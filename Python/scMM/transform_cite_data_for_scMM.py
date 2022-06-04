import anndata
import os
import numpy as np
from scipy.io import mmread, mmwrite
import scanpy as sc
import pandas as pd

# before starting: activate virtualenv 

"""
Required folder structure

RNA and protein count matrix should be stored in folder named `RNA-seq` and `CITE-seq` 
accomapnied with feature information stored in `gene.tsv` and `protein.tsv`, respectively. 
Also, single-cell barcode stored in `barcode.tsv` should be included. 

When running on chromatin accessibility data, name folder as `ATAC-seq` and feature file as `peak.tsv`. 

For example, folder structure looks like:
```
data/BMNC
     |---RNA-seq
     |   |---RNA_count.mtx
     |   |---gene.tsv
     |   |---barcode.tsv
     |---CITE-seq
         |---Protein_count.mtx
         |---protein.tsv
         |---barcode.tsv
```
"""

load_path = '/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/'
save_path = '/scratch/global/maren/multiomics-data/scMM/data/'

#
n_cells = [500, 1000, 2500, 5000, 10000]

for i in range(10):
    print('rep %s...'%i)
    for j in n_cells:
        print('%s cells...'%j)
        # 1) load subsampled data
        print('loading data...')
        rnadata = anndata.read_h5ad(load_path + 'adata_cite_gex_subsample_%s_cells_rep_%s.h5ad'%(j,i))
        proteindata = anndata.read_h5ad(load_path + 'adata_cite_protein_subsample_%s_cells_rep_%s.h5ad'%(j,i))
        # 2) extract counts, features and barcodes
        print('extracting info...')
        rna_counts = rnadata.layers["counts"]
        rna_features = rnadata.var["gene_ids"].keys()
        rna_barcodes = rnadata.obs["pct_counts_mt"].keys()
        #
        protein_counts = proteindata.layers["counts"]
        protein_features = proteindata.var["feature_types"].keys()
        protein_barcodes = proteindata.obs.index # names match rnadata object!! 
        # 3) some dimension checks 
        assert (protein_barcodes == rna_barcodes).sum() == len(protein_barcodes) # True 
        assert len(rna_barcodes) == rna_counts.shape[0]
        assert len(rna_features) == rna_counts.shape[1]
        assert len(protein_barcodes) == protein_counts.shape[0]
        assert len(protein_features) == protein_counts.shape[1]
        # 4) now save to .mtx und .csv
        # create directories if not yet existing
        save_data_path = save_path + '%s_cells_rep_%s/'%(j,i) # on server 
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        if not os.path.exists(save_data_path+'RNA-seq'):
            os.makedirs(save_data_path+'RNA-seq')
        if not os.path.exists(save_data_path+'CITE-seq'):
            os.makedirs(save_data_path+'CITE-seq')
        # define filenames for RNA + protein counts, features and barcodes, in accordance to the requirements of the scMM folder structure (see above)
        rna_counts_filename = save_data_path+"/RNA-seq/RNA_count.mtx"
        rna_features_filename = save_data_path+"/RNA-seq/gene.tsv"
        rna_barcodes_filename = save_data_path+"/RNA-seq/barcode.tsv"
        #
        protein_counts_filename = save_data_path+"CITE-seq/Protein_count.mtx"
        protein_features_filename = save_data_path+"CITE-seq/protein.tsv"
        protein_barcodes_filename = save_data_path+"CITE-seq/barcode.tsv"
        # write counts, features and barcodes as .mtx and tsv file
        print('saving files...')
        mmwrite(rna_counts_filename, rna_counts)
        mmwrite(protein_counts_filename, protein_counts)
        # write RNA features + barcode
        np.savetxt(rna_features_filename, rna_features.to_numpy(), delimiter="\t", fmt='"%s"')
        np.savetxt(rna_barcodes_filename, rna_barcodes.to_numpy(), delimiter="\t", fmt='"%s"')
        # write ATAC features + barcode
        np.savetxt(protein_features_filename, protein_features.to_numpy(), delimiter="\t", fmt='"%s"')
        np.savetxt(protein_barcodes_filename, protein_barcodes.to_numpy(), delimiter="\t", fmt='"%s"')
#