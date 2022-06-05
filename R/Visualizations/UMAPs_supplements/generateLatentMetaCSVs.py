import os as os
import scvi
import anndata as ad
import scipy as sp
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import exists
#os.environ['R_HOME'] = '/usr/lib/R'
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
from rpy2.rinterface_lib import openrlib
import scib

def prepareAdataObject(dataType, replicate, cellNumber):
    
    if dataType == "Multiome":
        adata_atac = sc.read("Desktop/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (cellNumber, replicate))
        adata_atac.var_names_make_unique()
        adata_atac = adata_atac[:,adata_atac.var['highly_variable'] == True].copy()
        adata_rna = sc.read("Desktop/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (cellNumber, replicate))
        adata_rna.var_names_make_unique()
    elif dataType == "CITE":
        adata_atac = sc.read("Desktop/adata_cite_protein_subsample_%s_cells_rep_%s.h5ad" % (cellNumber, replicate))
        adata_atac.var_names_make_unique()
        adata_rna = sc.read("Desktop/adata_cite_gex_subsample_%s_cells_rep_%s.h5ad" % (cellNumber, replicate))
        adata_rna.var_names_make_unique()

    common_barcodes = list(set(adata_rna.obs_names).intersection(adata_atac.obs_names))
    adata_rna_sub = adata_rna[common_barcodes].copy()
    adata_atac_sub = adata_atac[common_barcodes].copy()
    combined_dat = ad.concat([adata_rna_sub,adata_atac_sub], axis=1)
    combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
    combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]            
    combined_dat.obs["site"] = combined_dat.obs["batch"].str.split("d").str[0].str.replace('s','')
    combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
    
    if dataType == "Multiome":
        combined_dat.obs["pseudotime_order_ATAC"] = adata_rna_sub.obs["pseudotime_order_ATAC"]
        combined_dat.var["highly_variable"] = pd.concat([adata_rna_sub.var["highly_variable"],adata_atac_sub.var["highly_variable"]], axis=0).values
    elif dataType == "CITE":
        combined_dat.obs["pseudotime_order_ADT"] = adata_rna_sub.obs["pseudotime_order_ADT"]

    
    combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
    combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
    combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
    combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
    combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Peaks",) * adata_atac_sub.shape[1])], axis=0).values
    combined_dat.X = combined_dat.X.todense()
    return combined_dat
    
def attachLatentToAdataObject(combined_dat, dataType, replicate, tool, cellNumber):
    # Copy data to new object
    combined_dat_int = combined_dat.copy()
    # read embedding CSVs of various models
    # index_col = 0 treats first column as row names
    # first %s stand for model name, second for number of cells, third for repetition
    
    if dataType == "Multiome":
        embedding = pd.read_csv("Desktop/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (tool, cellNumber, replicate), index_col=0) 
    elif dataType == "CITE":
        embedding = pd.read_csv("Desktop/%s/csv_cite/latent_cite_subsample_%s_cells_rep_%s.csv" % (tool, cellNumber, replicate), index_col=0) 

    embedding = embedding[["0","1","2","3","4","5","6","7","8","9"]]
    embedding.index = embedding.index.str.replace(r'CITE~|Multiome~', '')
    embedding = embedding.loc[combined_dat_int.obs_names, :]
    embedding_df = embedding.copy()
    embedding = embedding.to_numpy()
    combined_dat_int.obsm['X_emb'] = embedding.copy()
    #sc.pp.neighbors(combined_dat)
    #sc.tl.leiden(combined_dat)
    sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
    sc.tl.leiden(combined_dat_int)
    
    
    meta_data_df =  pd.DataFrame.from_records(combined_dat_int.obs, index = combined_dat_int.obs_names)

    latent_with_meta = embedding_df.join(meta_data_df)
    
    return latent_with_meta
    
    
dataTypes = ["Multiome", "CITE"]   
replicate = 1
cellNumbers = [500,1000,2500,5000,10000]
toolsCITE = ["Cobolt", "scMM", "totalvi", "SCALEX"]
toolsMultiome = ["Cobolt", "scMM", "multivi", "scMVP"]


for dataType in dataTypes:
    print(dataType)
    for cellNumber in cellNumbers:
        print(str(cellNumber))
        combined_dat = prepareAdataObject(dataType = dataType, replicate = replicate, cellNumber = cellNumber)
        
        if dataType == "CITE":
            for toolCITE in toolsCITE:
                print(toolCITE)
                latent_with_meta =  attachLatentToAdataObject(combined_dat = combined_dat, dataType = dataType, replicate = replicate, tool = toolCITE, cellNumber = cellNumber)
                latent_with_meta.to_csv('dataType_' + dataType  + '_tool_' + toolCITE + '_cellNumber_'+ str(cellNumber) + '_replicate_'+ str(replicate) +'.csv')  
                
                # combined_dat_int = attachLatentToAdataObject(combined_dat = combined_dat, dataType = dataType, replicate = replicate, tool = toolCITE, cellNumber = cellNumber)

                # sc.pl.embedding(combined_dat_int, 
                #                 basis='X_emb',
                #                 color=combined_dat_int.obs.columns.difference(['modality']), 
                #                 ncols = 1,
                #                 legend_fontsize = 'xx-small',
                #                 save='_dataType_' + dataType  + '_tool_' + toolCITE + '_cellNumber_'+ str(cellNumber) + '_replicate_'+ str(replicate) +'.pdf')
                # del combined_dat_int
                
        elif dataType == "Multiome":
            for toolMultiome in toolsMultiome:
                print(toolMultiome)
                latent_with_meta = attachLatentToAdataObject(combined_dat = combined_dat, dataType = dataType, replicate = replicate, tool = toolMultiome, cellNumber = cellNumber)
                latent_with_meta.to_csv('dataType_' + dataType  + '_tool_' + toolMultiome + '_cellNumber_'+ str(cellNumber) + '_replicate_'+ str(replicate) +'.csv')  

                # combined_dat_int = attachLatentToAdataObject(combined_dat = combined_dat, dataType = dataType, replicate = replicate, tool = toolMultiome, cellNumber = cellNumber)
                # sc.pl.embedding(combined_dat_int, 
                #                 basis='X_emb',
                #                 color=combined_dat_int.obs.columns.difference(['modality']), 
                #                 ncols = 1,
                #                 legend_fontsize = 'xx-small',
                #                 save='_dataType_' + dataType  + '_tool_' + toolMultiome + '_cellNumber_'+ str(cellNumber) + '_replicate_'+ str(replicate) +'.pdf')
                # del combined_dat_int
                
        del combined_dat
             










    

