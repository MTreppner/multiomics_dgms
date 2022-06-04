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
os.environ['R_HOME'] = '/usr/lib/R'
import scib

# Connect to ssh
# sshfs treppner@homesync.imbi.uni-freiburg.de:/scratch/global/maren /Users/martintreppner/ssh_eval_maren
# sshfs treppner@homesync.imbi.uni-freiburg.de:/scratch/global/treppner /Users/martintreppner/ssh_eval

# Unmount
# sudo diskutil unmount force /Users/martintreppner/ssh_eval_maren
# sudo diskutil unmount force /Users/martintreppner/ssh_eval

############################################################################
############################ CITE-seq ######################################
############################################################################

n_cells = [500,1000,2500,5000,10000]
models = ["Cobolt", "scMM", "totalvi", "SCALEX"]
eval_all = []
scvi.settings.seed = 420
for j in n_cells:
    for i in range(10):
        for k in models:
            adata_protein = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_cite_protein_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            adata_protein.var_names_make_unique()
            print("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_cite_protein_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            adata_rna = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_cite_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            adata_rna.var_names_make_unique()
            print("/scratch/global/treppner/multi_omics_data/multi_omics_data/neurips_data/subsampled/adata_cite_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            # Search for common barcodes (cells)
            common_barcodes = list(set(adata_rna.obs_names).intersection(adata_protein.obs_names))
            adata_rna_sub = adata_rna[common_barcodes].copy()
            adata_protein_sub = adata_protein[common_barcodes].copy()
            # Concatenate data sets 
            combined_dat = ad.concat([adata_rna_sub,adata_protein_sub], axis=1)
            combined_dat.var_names_make_unique()
            combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
            combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
            combined_dat.obs["site"] = combined_dat.obs["batch"].str.split("d").str[0].str.replace('s','')
            combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
            combined_dat.obs["pseudotime_order_ADT"] = adata_rna_sub.obs["pseudotime_order_ADT"]
            combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
            combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
            combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
            combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
            combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Proteins",) * adata_protein_sub.shape[1])], axis=0).values
            combined_dat.obsm["protein_expression"] = adata_protein_sub.layers['counts'].toarray()
            combined_dat.X = combined_dat.X.todense()
            # Copy data to new object
            combined_dat_int = combined_dat.copy()
            # read embedding CSVs of various models
            # index_col = 0 treats first column as row names
            # first %s stand for model name, second for number of cells, third for repetition
            if os.path.exists("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv_cite/latent_cite_subsample_%s_cells_rep_%s.csv" % (k,j,i)) == True:
                embedding = pd.read_csv("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv_cite/latent_cite_subsample_%s_cells_rep_%s.csv" % (k,j,i), index_col=0)
                embedding = embedding[["0","1","2","3","4","5","6","7","8","9"]]
                print("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv_cite/latent_cite_subsample_%s_cells_rep_%s.csv" % (k,j,i))
                embedding.index = embedding.index.str.replace(r'CITE~|Multiome~', '')
                embedding = embedding.loc[combined_dat_int.obs_names, :]
                embedding = embedding.to_numpy()
                combined_dat_int.obsm['X_emb'] = embedding.copy()
                label_key = 'cell_type'
                batch_key = 'batch'
                sc.pp.neighbors(combined_dat)
                sc.tl.leiden(combined_dat)
                sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
                sc.tl.leiden(combined_dat_int)
                # Caluclate evaluation metrics
                # Set True/False depending on which metrics we need 
                # calculate metrics with metrics from scib 
                eval = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, isolated_labels_asw_=True,
                        silhouette_=True,
                        hvg_score_=False,
                        pcr_=True,
                        isolated_labels_f1_=True,
                        nmi_=True,
                        ari_=True,
                        graph_conn_=True,
                        embed = 'X_emb'
                )
                trajectory_score = scib.metrics.trajectory_conservation(
                        combined_dat,
                        combined_dat_int,
                        label_key=label_key,
                        batch_key = batch_key,
                        pseudotime_key = 'pseudotime_order_GEX'
                )
                # cell_cycle throws and error with sparse input
                cell_cycle = scib.metrics.cell_cycle(
                        combined_dat, 
                        combined_dat_int, 
                        batch_key,
                        embed='X_emb',
                        organism='human',
                        n_comps=50,
                )
                ASW_site = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key='site', label_key=label_key, 
                        silhouette_=True,
                        embed = 'X_emb'
                )
                # Transpose and add information (model, cells, repition)
                eval = eval.transpose()
                eval["trajectory"] = trajectory_score
                eval["cell_cycle_conservation"] = cell_cycle
                eval["ASW_label/site"] = ASW_site.loc['ASW_label/batch']
                eval["model"] = k
                eval["n_cells"] = j
                eval["rep"] = i
                # Concat to combined 
                eval_all.append(eval)
                del eval
                del adata_protein
                del adata_rna
                del combined_dat
                del combined_dat_int
            else:
                pass
    # Concat all models, cells, and reps
    eval_df = pd.concat(eval_all)
    # Write to csv
    eval_df.to_csv("/scratch/global/treppner/multi_omics_data/cite_evaluation_output_%s_cells.csv"  % (j))
    del eval_df

############################################################################
############################ Multiome ######################################
############################################################################

# No loop for memory reasons

# 500 cells
n_cells = [500]
models = ["Cobolt", "scMM", "multivi", "scMVP"]
eval_all = []
scvi.settings.seed = 123
for j in n_cells:
    for i in range(10):
        for k in models:
            # read atac data
            adata_atac = sc.read("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            adata_atac.var_names_make_unique()
            print("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            # Filter HVGs from gex data set
            adata_atac = adata_atac[:,adata_atac.var['highly_variable'] == True].copy()
            # read rna data
            adata_rna = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            adata_rna.var_names_make_unique()
            print("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            # Search for common barcodes (cells)
            common_barcodes = list(set(adata_rna.obs_names).intersection(adata_atac.obs_names))
            adata_rna_sub = adata_rna[common_barcodes].copy()
            adata_atac_sub = adata_atac[common_barcodes].copy()
            # Concatenate data sets 
            combined_dat = ad.concat([adata_rna_sub,adata_atac_sub], axis=1)
            combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
            combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
            combined_dat.obs["site"] = combined_dat.obs["batch"].str.split("d").str[0].str.replace('s','')
            combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
            combined_dat.obs["pseudotime_order_ATAC"] = adata_rna_sub.obs["pseudotime_order_ATAC"]
            combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
            combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
            combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
            combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
            combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Peaks",) * adata_atac_sub.shape[1])], axis=0).values
            combined_dat.var["highly_variable"] = pd.concat([adata_rna_sub.var["highly_variable"],adata_atac_sub.var["highly_variable"]], axis=0).values
            combined_dat.X = combined_dat.X.todense()
            # Copy data to new object
            combined_dat_int = combined_dat.copy()
            # read embedding CSVs of various models
            # index_col = 0 treats first column as row names
            # first %s stand for model name, second for number of cells, third for repetition
            embedding = pd.read_csv("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i), index_col=0) 
            print("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i))
            embedding.index = embedding.index.str.replace(r'CITE~|Multiome~', '')
            embedding = embedding.loc[combined_dat_int.obs_names, :]
            embedding = embedding.to_numpy()
            combined_dat_int.obsm['X_emb'] = embedding.copy()
            label_key = 'cell_type'
            batch_key = 'batch'
            sc.pp.neighbors(combined_dat)
            sc.tl.leiden(combined_dat)
            sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
            sc.tl.leiden(combined_dat_int)
            # Caluclate evaluation metrics
            # Set True/False depending on which metrics we need 
            # calculate metrics with metrics from scib 
            eval = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, isolated_labels_asw_=True,
                    silhouette_=True,
                    hvg_score_=False,
                    pcr_=True,
                    isolated_labels_f1_=True,
                    nmi_=True,
                    ari_=True,
                    graph_conn_=True,
                    embed = 'X_emb'
            )
            trajectory_score = scib.metrics.trajectory_conservation(
                    combined_dat,
                    combined_dat_int,
                    label_key=label_key,
                    batch_key = batch_key,
                    pseudotime_key = 'pseudotime_order_GEX'
            )
            # cell_cycle throws and error with sparse input
            cell_cycle = scib.metrics.cell_cycle(
                    combined_dat, 
                    combined_dat_int, 
                    batch_key,
                    embed='X_emb',
                    organism='human',
                    n_comps=50,
            )
            ASW_site = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key='site', label_key=label_key, 
                    silhouette_=True,
                    embed = 'X_emb'
            )            
            # Transpose and add information (model, cells, repition)
            eval = eval.transpose()
            eval["trajectory"] = trajectory_score
            eval["cell_cycle_conservation"] = cell_cycle
            eval["ASW_label/site"] = ASW_site.loc['ASW_label/batch']
            eval["model"] = k
            eval["n_cells"] = j
            eval["rep"] = i
            # Concat to combined 
            eval_all.append(eval)
            del eval
            del adata_atac
            del adata_rna
            del combined_dat
            del combined_dat_int
    # Concat all models, cells, and reps
    eval_df = pd.concat(eval_all)
    # Write to csv
    eval_df.to_csv("/scratch/global/treppner/multi_omics_data/multiome_evaluation_output_%s_cells.csv"  % (j))
    del eval_df

# 1000 cells
n_cells = [1000]
models = ["Cobolt", "scMM", "multivi", "scMVP"]
eval_all = []
scvi.settings.seed = 123
for j in n_cells:
    for i in range(10):
        for k in models:
            # read atac data
            adata_atac = sc.read("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            adata_atac.var_names_make_unique()
            print("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            # Filter HVGs from gex data set
            adata_atac = adata_atac[:,adata_atac.var['highly_variable'] == True].copy()
            # read rna data
            adata_rna = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            adata_rna.var_names_make_unique()
            print("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            # Search for common barcodes (cells)
            common_barcodes = list(set(adata_rna.obs_names).intersection(adata_atac.obs_names))
            adata_rna_sub = adata_rna[common_barcodes].copy()
            adata_atac_sub = adata_atac[common_barcodes].copy()
            # Concatenate data sets 
            combined_dat = ad.concat([adata_rna_sub,adata_atac_sub], axis=1)
            combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
            combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
            combined_dat.obs["site"] = combined_dat.obs["batch"].str.split("d").str[0].str.replace('s','')
            combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
            combined_dat.obs["pseudotime_order_ATAC"] = adata_rna_sub.obs["pseudotime_order_ATAC"]
            combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
            combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
            combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
            combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
            combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Peaks",) * adata_atac_sub.shape[1])], axis=0).values
            combined_dat.var["highly_variable"] = pd.concat([adata_rna_sub.var["highly_variable"],adata_atac_sub.var["highly_variable"]], axis=0).values
            combined_dat.X = combined_dat.X.todense()
            # Copy data to new object
            combined_dat_int = combined_dat.copy()
            # read embedding CSVs of various models
            # index_col = 0 treats first column as row names
            # first %s stand for model name, second for number of cells, third for repetition
            embedding = pd.read_csv("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i), index_col=0) 
            print("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i))
            embedding.index = embedding.index.str.replace(r'CITE~|Multiome~', '')
            embedding = embedding.loc[combined_dat_int.obs_names, :]
            embedding = embedding.to_numpy()
            combined_dat_int.obsm['X_emb'] = embedding.copy()
            label_key = 'cell_type'
            batch_key = 'batch'
            sc.pp.neighbors(combined_dat)
            sc.tl.leiden(combined_dat)
            sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
            sc.tl.leiden(combined_dat_int)
            # Caluclate evaluation metrics
            # Set True/False depending on which metrics we need 
            # calculate metrics with metrics from scib 
            eval = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, isolated_labels_asw_=True,
                    silhouette_=True,
                    hvg_score_=False,
                    pcr_=True,
                    isolated_labels_f1_=True,
                    nmi_=True,
                    ari_=True,
                    graph_conn_=True,
                    embed = 'X_emb'
            )
            trajectory_score = scib.metrics.trajectory_conservation(
                    combined_dat,
                    combined_dat_int,
                    label_key=label_key,
                    batch_key = batch_key,
                    pseudotime_key = 'pseudotime_order_GEX'
            )
            # cell_cycle throws and error with sparse input
            cell_cycle = scib.metrics.cell_cycle(
                    combined_dat, 
                    combined_dat_int, 
                    batch_key,
                    embed='X_emb',
                    organism='human',
                    n_comps=50,
            )
            ASW_site = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key='site', label_key=label_key, 
                    silhouette_=True,
                    embed = 'X_emb'
            )            
            # Transpose and add information (model, cells, repition)
            eval = eval.transpose()
            eval["trajectory"] = trajectory_score
            eval["cell_cycle_conservation"] = cell_cycle
            eval["ASW_label/site"] = ASW_site.loc['ASW_label/batch']
            eval["model"] = k
            eval["n_cells"] = j
            eval["rep"] = i
            # Concat to combined 
            eval_all.append(eval)
            del eval
            del adata_atac
            del adata_rna
            del combined_dat
            del combined_dat_int
    # Concat all models, cells, and reps
    eval_df = pd.concat(eval_all)
    # Write to csv
    eval_df.to_csv("/scratch/global/treppner/multi_omics_data/multiome_evaluation_output_%s_cells.csv"  % (j))
    del eval_df

# 2500 cells
n_cells = [2500]
models = ["Cobolt", "scMM", "multivi", "scMVP"]
eval_all = []
scvi.settings.seed = 123
for j in n_cells:
    for i in range(10):
        for k in models:
            # read atac data
            adata_atac = sc.read("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            adata_atac.var_names_make_unique()
            print("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            # Filter HVGs from gex data set
            adata_atac = adata_atac[:,adata_atac.var['highly_variable'] == True].copy()
            # read rna data
            adata_rna = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            adata_rna.var_names_make_unique()
            print("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            # Search for common barcodes (cells)
            common_barcodes = list(set(adata_rna.obs_names).intersection(adata_atac.obs_names))
            adata_rna_sub = adata_rna[common_barcodes].copy()
            adata_atac_sub = adata_atac[common_barcodes].copy()
            # Concatenate data sets 
            combined_dat = ad.concat([adata_rna_sub,adata_atac_sub], axis=1)
            combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
            combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
            combined_dat.obs["site"] = combined_dat.obs["batch"].str.split("d").str[0].str.replace('s','')
            combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
            combined_dat.obs["pseudotime_order_ATAC"] = adata_rna_sub.obs["pseudotime_order_ATAC"]
            combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
            combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
            combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
            combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
            combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Peaks",) * adata_atac_sub.shape[1])], axis=0).values
            combined_dat.var["highly_variable"] = pd.concat([adata_rna_sub.var["highly_variable"],adata_atac_sub.var["highly_variable"]], axis=0).values
            combined_dat.X = combined_dat.X.todense()
            # Copy data to new object
            combined_dat_int = combined_dat.copy()
            # read embedding CSVs of various models
            # index_col = 0 treats first column as row names
            # first %s stand for model name, second for number of cells, third for repetition
            embedding = pd.read_csv("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i), index_col=0) 
            print("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i))
            embedding.index = embedding.index.str.replace(r'CITE~|Multiome~', '')
            embedding = embedding.loc[combined_dat_int.obs_names, :]
            embedding = embedding.to_numpy()
            combined_dat_int.obsm['X_emb'] = embedding.copy()
            label_key = 'cell_type'
            batch_key = 'batch'
            sc.pp.neighbors(combined_dat)
            sc.tl.leiden(combined_dat)
            sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
            sc.tl.leiden(combined_dat_int)
            # Caluclate evaluation metrics
            # Set True/False depending on which metrics we need 
            # calculate metrics with metrics from scib 
            eval = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, isolated_labels_asw_=True,
                    silhouette_=True,
                    hvg_score_=False,
                    pcr_=True,
                    isolated_labels_f1_=True,
                    nmi_=True,
                    ari_=True,
                    graph_conn_=True,
                    embed = 'X_emb'
            )
            trajectory_score = scib.metrics.trajectory_conservation(
                    combined_dat,
                    combined_dat_int,
                    label_key=label_key,
                    batch_key = batch_key,
                    pseudotime_key = 'pseudotime_order_GEX'
            )
            # cell_cycle throws and error with sparse input
            cell_cycle = scib.metrics.cell_cycle(
                    combined_dat, 
                    combined_dat_int, 
                    batch_key,
                    embed='X_emb',
                    organism='human',
                    n_comps=50,
            )
            ASW_site = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key='site', label_key=label_key, 
                    silhouette_=True,
                    embed = 'X_emb'
            )            
            # Transpose and add information (model, cells, repition)
            eval = eval.transpose()
            eval["trajectory"] = trajectory_score
            eval["cell_cycle_conservation"] = cell_cycle
            eval["ASW_label/site"] = ASW_site.loc['ASW_label/batch']
            eval["model"] = k
            eval["n_cells"] = j
            eval["rep"] = i
            # Concat to combined 
            eval_all.append(eval)
            del eval
            del adata_atac
            del adata_rna
            del combined_dat
            del combined_dat_int
    # Concat all models, cells, and reps
    eval_df = pd.concat(eval_all)
    # Write to csv
    eval_df.to_csv("/scratch/global/treppner/multi_omics_data/multiome_evaluation_output_%s_cells.csv"  % (j))
    del eval_df

# 5000 cells
n_cells = [5000]
models = ["Cobolt", "scMM", "multivi", "scMVP"]
eval_all = []
scvi.settings.seed = 123
for j in n_cells:
    for i in range(6,10):
        for k in models:
            # read atac data
            adata_atac = sc.read("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            adata_atac.var_names_make_unique()
            print("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            # Filter HVGs from gex data set
            adata_atac = adata_atac[:,adata_atac.var['highly_variable'] == True].copy()
            # read rna data
            adata_rna = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            adata_rna.var_names_make_unique()
            print("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            # Search for common barcodes (cells)
            common_barcodes = list(set(adata_rna.obs_names).intersection(adata_atac.obs_names))
            adata_rna_sub = adata_rna[common_barcodes].copy()
            adata_atac_sub = adata_atac[common_barcodes].copy()
            # Concatenate data sets 
            combined_dat = ad.concat([adata_rna_sub,adata_atac_sub], axis=1)
            combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
            combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
            combined_dat.obs["site"] = combined_dat.obs["batch"].str.split("d").str[0].str.replace('s','')
            combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
            combined_dat.obs["pseudotime_order_ATAC"] = adata_rna_sub.obs["pseudotime_order_ATAC"]
            combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
            combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
            combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
            combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
            combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Peaks",) * adata_atac_sub.shape[1])], axis=0).values
            combined_dat.var["highly_variable"] = pd.concat([adata_rna_sub.var["highly_variable"],adata_atac_sub.var["highly_variable"]], axis=0).values
            combined_dat.X = combined_dat.X.todense()
            # Copy data to new object
            combined_dat_int = combined_dat.copy()
            # read embedding CSVs of various models
            # index_col = 0 treats first column as row names
            # first %s stand for model name, second for number of cells, third for repetition
            embedding = pd.read_csv("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i), index_col=0) 
            print("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i))
            embedding.index = embedding.index.str.replace(r'CITE~|Multiome~', '')
            embedding = embedding.loc[combined_dat_int.obs_names, :]
            embedding = embedding.to_numpy()
            combined_dat_int.obsm['X_emb'] = embedding.copy()
            label_key = 'cell_type'
            batch_key = 'batch'
            sc.pp.neighbors(combined_dat)
            sc.tl.leiden(combined_dat)
            sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
            sc.tl.leiden(combined_dat_int)
            # Caluclate evaluation metrics
            # Set True/False depending on which metrics we need 
            # calculate metrics with metrics from scib 
            eval = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, isolated_labels_asw_=True,
                    silhouette_=True,
                    hvg_score_=False,
                    pcr_=True,
                    isolated_labels_f1_=True,
                    nmi_=True,
                    ari_=True,
                    graph_conn_=True,
                    embed = 'X_emb'
            )
            trajectory_score = scib.metrics.trajectory_conservation(
                    combined_dat,
                    combined_dat_int,
                    label_key=label_key,
                    batch_key = batch_key,
                    pseudotime_key = 'pseudotime_order_GEX'
            )
            # cell_cycle throws and error with sparse input
            cell_cycle = scib.metrics.cell_cycle(
                    combined_dat, 
                    combined_dat_int, 
                    batch_key,
                    embed='X_emb',
                    organism='human',
                    n_comps=50,
            )
            ASW_site = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key='site', label_key=label_key, 
                    silhouette_=True,
                    embed = 'X_emb'
            )            
            # Transpose and add information (model, cells, repition)
            eval = eval.transpose()
            eval["trajectory"] = trajectory_score
            eval["cell_cycle_conservation"] = cell_cycle
            eval["ASW_label/site"] = ASW_site.loc['ASW_label/batch']
            eval["model"] = k
            eval["n_cells"] = j
            eval["rep"] = i
            # Concat to combined 
            eval_all.append(eval)
            del eval
            del adata_atac
            del adata_rna
            del combined_dat
            del combined_dat_int
    # Concat all models, cells, and reps
    eval_df = pd.concat(eval_all)
    # Write to csv
    eval_df.to_csv("/scratch/global/treppner/multi_omics_data/multiome_evaluation_output_%s_cells2.csv"  % (j))
    del eval_df

# 10000 cells
n_cells = [10000]
models = ["Cobolt", "scMM", "multivi", "scMVP"]
eval_all = []
scvi.settings.seed = 123
for j in n_cells:
    for i in range(9,10):
        for k in models:
            # read atac data
            adata_atac = sc.read("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            adata_atac.var_names_make_unique()
            print("/scratch/global/maren/multiomics-data/neurips_data_atac_top_peaks/adata_atac_subsample_top_peaks_%s_cells_rep_%s.h5ad" % (j,i))
            # Filter HVGs from gex data set
            adata_atac = adata_atac[:,adata_atac.var['highly_variable'] == True].copy()
            # read rna data
            adata_rna = sc.read("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            adata_rna.var_names_make_unique()
            print("/scratch/global/treppner/multi_omics_data/neurips_data/subsampled/adata_gex_subsample_%s_cells_rep_%s.h5ad" % (j,i))
            # Search for common barcodes (cells)
            common_barcodes = list(set(adata_rna.obs_names).intersection(adata_atac.obs_names))
            adata_rna_sub = adata_rna[common_barcodes].copy()
            adata_atac_sub = adata_atac[common_barcodes].copy()
            # Concatenate data sets 
            combined_dat = ad.concat([adata_rna_sub,adata_atac_sub], axis=1)
            combined_dat.obs["cell_type"] = adata_rna_sub.obs["cell_type"]
            combined_dat.obs["batch"] = adata_rna_sub.obs["batch"]
            combined_dat.obs["site"] = combined_dat.obs["batch"].str.split("d").str[0].str.replace('s','')
            combined_dat.obs["pseudotime_order_GEX"] = adata_rna_sub.obs["pseudotime_order_GEX"]
            combined_dat.obs["pseudotime_order_ATAC"] = adata_rna_sub.obs["pseudotime_order_ATAC"]
            combined_dat.obs["phase"] = adata_rna_sub.obs["phase"]
            combined_dat.obs["modality"] = pd.Series(("paired",) * adata_rna_sub.shape[0]).values
            combined_dat.obsm["X_pca"] = adata_rna_sub.obsm["X_pca"]
            combined_dat.obsm["X_umap"] = adata_rna_sub.obsm["X_umap"]
            combined_dat.var["modality"] = pd.concat([pd.Series(("Gene Expression",) * adata_rna_sub.shape[1]),pd.Series(("Peaks",) * adata_atac_sub.shape[1])], axis=0).values
            combined_dat.var["highly_variable"] = pd.concat([adata_rna_sub.var["highly_variable"],adata_atac_sub.var["highly_variable"]], axis=0).values
            combined_dat.X = combined_dat.X.todense()
            # Copy data to new object
            combined_dat_int = combined_dat.copy()
            # read embedding CSVs of various models
            # index_col = 0 treats first column as row names
            # first %s stand for model name, second for number of cells, third for repetition
            embedding = pd.read_csv("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i), index_col=0) 
            print("/scratch/global/treppner/multi_omics_data/multi_omics_review/%s/csv/latent_subsample_%s_cells_rep_%s.csv" % (k,j,i))
            embedding.index = embedding.index.str.replace(r'CITE~|Multiome~', '')
            embedding = embedding.loc[combined_dat_int.obs_names, :]
            embedding = embedding.to_numpy()
            combined_dat_int.obsm['X_emb'] = embedding.copy()
            label_key = 'cell_type'
            batch_key = 'batch'
            sc.pp.neighbors(combined_dat)
            sc.tl.leiden(combined_dat)
            sc.pp.neighbors(combined_dat_int,use_rep='X_emb')
            sc.tl.leiden(combined_dat_int)
            # Caluclate evaluation metrics
            # Set True/False depending on which metrics we need 
            # calculate metrics with metrics from scib 
            eval = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key=batch_key, label_key=label_key, isolated_labels_asw_=True,
                    silhouette_=True,
                    hvg_score_=False,
                    pcr_=True,
                    isolated_labels_f1_=True,
                    nmi_=True,
                    ari_=True,
                    graph_conn_=True,
                    embed = 'X_emb'
            )
            trajectory_score = scib.metrics.trajectory_conservation(
                    combined_dat,
                    combined_dat_int,
                    label_key=label_key,
                    batch_key = batch_key,
                    pseudotime_key = 'pseudotime_order_GEX'
            )
            # cell_cycle throws and error with sparse input
            cell_cycle = scib.metrics.cell_cycle(
                    combined_dat, 
                    combined_dat_int, 
                    batch_key,
                    embed='X_emb',
                    organism='human',
                    n_comps=50,
            )
            ASW_site = scib.metrics.metrics(combined_dat, combined_dat_int, batch_key='site', label_key=label_key, 
                    silhouette_=True,
                    embed = 'X_emb'
            )            
            # Transpose and add information (model, cells, repition)
            eval = eval.transpose()
            eval["trajectory"] = trajectory_score
            eval["cell_cycle_conservation"] = cell_cycle
            eval["ASW_label/site"] = ASW_site.loc['ASW_label/batch']
            eval["model"] = k
            eval["n_cells"] = j
            eval["rep"] = i
            # Concat to combined 
            eval_all.append(eval)
            del eval
            del adata_atac
            del adata_rna
            del combined_dat
            del combined_dat_int
    # Concat all models, cells, and reps
    eval_df = pd.concat(eval_all)
    # Write to csv
    eval_df.to_csv("/scratch/global/treppner/multi_omics_data/multiome_evaluation_output_%s_cells9.csv"  % (j))
    del eval_df

