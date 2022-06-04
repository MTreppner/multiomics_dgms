import logging
import os
import urllib
import numpy as np
import pandas as pd
import scipy.io as sp_io
from scipy.sparse import csr_matrix, issparse
import anndata

from scMVP.dataset.dataset import CellMeasurement, GeneExpressionDataset

logger = logging.getLogger(__name__)

available_specification = ["filtered", "raw"]


class LoadFromAnnData(GeneExpressionDataset):

    """
    Dataset format:
    dataset = {
    "gene_barcodes": xxx,
    "gene_expression": xxx,
    "gene_names": xxx,
    "atac_barcodes": xxx,
    "atac_expression": xxx,
    "atac_names": xxx,
    }
    OR
    dataset = {
    "gene_expression":xxx,
    "atac_expression":xxx,
    }
    """
    def __init__(self,
        rnadata: anndata = None, 
        atacdata: anndata = None,
        dataset: dict = None,
        dense: bool = False,
        measurement_names_column: int = 0,
        remove_extracted_data: bool = False,
        delayed_populating: bool = False,
        atac_threshold: float = 0.0001, # express in over 0.01%
        cell_threshold: int = 1, # filtering cells less than minimum count
        cell_meta: pd.DataFrame = None,
        ):

        self.dataset = dataset
        self.rnadata=rnadata
        self.atacdata=atacdata
        self.barcodes = None
        self.dense = dense
        self.measurement_names_column = measurement_names_column
        self.remove_extracted_data = remove_extracted_data
        self.atac_thres = atac_threshold
        self.cell_thres = cell_threshold
        self.cell_meta = cell_meta
        super().__init__()
        if not delayed_populating:
            self.populate()

    def populate(self):
        logger.info("Preprocessing joint profiling dataset.")
        joint_profiles = {}

        rna_counts = self.rnadata.layers["counts"]
        rna_features = self.rnadata.var["gene_ids"].keys()
        rna_barcodes = self.rnadata.obs["pct_counts_mt"].keys()
        #
        atacdata_hvp = self.atacdata[:,self.atacdata.var['highly_variable']]
        assert atacdata_hvp.var['highly_variable'].sum() == len(atacdata_hvp.var['highly_variable'])

        atac_counts = atacdata_hvp.layers["counts"]
        atac_features = atacdata_hvp.var["feature_types"].keys()
        atac_barcodes = atacdata_hvp.obs["atac_fragments"].keys() # names match rnadata object!! 

        joint_profiles = {}
        joint_profiles["gene_barcodes"] = pd.DataFrame(rna_barcodes)
        joint_profiles["gene_names"] = pd.DataFrame(rna_features)
        joint_profiles["gene_expression"] = rna_counts#.T

        joint_profiles["atac_barcodes"] = pd.DataFrame(atac_barcodes)
        joint_profiles["atac_names"] = pd.DataFrame(atac_features)
        joint_profiles["atac_expression"] = atac_counts#.T

        share_index, gene_barcode_index, atac_barcode_index = np.intersect1d(joint_profiles["gene_barcodes"].values,
                                                                    joint_profiles["atac_barcodes"].values,
                                                                    return_indices=True)

        if isinstance(self.cell_meta,pd.DataFrame):
            if self.cell_meta.shape[1] < 2:
                logger.info("Please use cell id in first column and give ata least 2 columns.")
                return
            meta_cell_id = self.cell_meta.iloc[:,0].values
            meta_share, meta_barcode_index, share_barcode_index =\
                np.intersect1d(meta_cell_id,
                share_index, return_indices=True)
            _gene_barcode_index = gene_barcode_index[share_barcode_index]
            _atac_barcode_index = atac_barcode_index[share_barcode_index]
            if len(_gene_barcode_index) < 2: # no overlaps
                logger.info("Inconsistent metadata to expression data.")
                return
            tmp = joint_profiles["gene_barcodes"]
            joint_profiles["gene_barcodes"] = tmp.loc[_gene_barcode_index, :]
            temp = joint_profiles["atac_barcodes"]
            joint_profiles["atac_barcodes"] = temp.loc[_atac_barcode_index, :]

        else:
            # reorder rnaseq cell meta
            tmp = joint_profiles["gene_barcodes"]
            joint_profiles["gene_barcodes"] = tmp.loc[gene_barcode_index,:]
            temp = joint_profiles["atac_barcodes"]
            joint_profiles["atac_barcodes"] = temp.loc[atac_barcode_index, :]

        gene_tab = joint_profiles["gene_expression"]
        if issparse(gene_tab):
            joint_profiles["gene_expression"] = gene_tab[gene_barcode_index, :].A
        else:
            joint_profiles["gene_expression"] = gene_tab[gene_barcode_index, :]

        temp = joint_profiles["atac_expression"]
        reorder_atac_exp = temp[atac_barcode_index, :]
        binary_index = reorder_atac_exp > 1
        reorder_atac_exp[binary_index] = 1
        # remove peaks > 10% of total cells
        high_count_atacs = ((reorder_atac_exp > 0).sum(axis=0).ravel() >= self.atac_thres * reorder_atac_exp.shape[0]) \
                           & ((reorder_atac_exp > 0).sum(axis=0).ravel() <= 0.1 * reorder_atac_exp.shape[0])

        if issparse(reorder_atac_exp):
            high_count_atacs_index = np.where(high_count_atacs)
            _tmp = reorder_atac_exp[:, high_count_atacs_index[1]]
            joint_profiles["atac_expression"] = _tmp.A
            joint_profiles["atac_names"] = joint_profiles["atac_names"].loc[high_count_atacs_index[1], :]

        else:
            _tmp = reorder_atac_exp[:, high_count_atacs]

            joint_profiles["atac_expression"] = _tmp
            joint_profiles["atac_names"] = joint_profiles["atac_names"].loc[high_count_atacs, :]

         # RNA-seq as the key
        Ys = []
        measurement = CellMeasurement(
            name="atac_expression",
            data=joint_profiles["atac_expression"],
            columns_attr_name="atac_names",
            columns=joint_profiles["atac_names"].astype(str),
        )
        Ys.append(measurement)
        # Add cell metadata
        if isinstance(self.cell_meta,pd.DataFrame):
            for l_index, label in enumerate(list(self.cell_meta.columns.values)):
                if l_index >0:
                    label_measurement = CellMeasurement(
                        name="{}_label".format(label),
                        data=self.cell_meta.iloc[meta_barcode_index,l_index],
                        columns_attr_name=label,
                        columns=self.cell_meta.iloc[meta_barcode_index, l_index]
                    )
                    Ys.append(label_measurement)
                    logger.info("Loading {} into dataset.".format(label))

        cell_attributes_dict = {
            "barcodes": np.squeeze(np.asarray(joint_profiles["gene_barcodes"], dtype=str))
        }

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=joint_profiles["gene_expression"],
            batch_indices=None,
            gene_names=joint_profiles["gene_names"].astype(str),
            cell_attributes_dict=cell_attributes_dict,
            Ys=Ys,
        )
        self.filter_cells_by_count(self.cell_thres)

    def _input_check(self):
        if len(self.dataset.keys()) == 2:
            for _key in self.dataset.keys():
                if _key not in self._minimum_input:
                    logger.info("Unknown input data type:{}".format(_key))
                    return False
                # if not self.dataset[_key].split(".")[-1] in ["txt","tsv","csv"]:
                #     logger.debug("scMVP only support two files input of txt, tsv or csv!")
                #     return False
        elif len(self.dataset.keys()) >= 6:
            for _key in self._allow_input:
                if not _key in self.dataset.keys():
                    logger.info("Data type {} missing.".format(_key))
                    return False
        else:
            logger.info("Incorrect input file number.")
            return False
        for _key in self.dataset.keys():
            if not os.path.exists(self.data_path):
                logger.info("{} do not exist!".format(self.data_path))
            if not os.path.exists("{}{}".format(self.data_path, self.dataset[_key])):
                logger.info("Cannot find {}{}!".format(self.data_path, self.dataset[_key]))
                return False
        return True

    def _download(self, url: str, save_path: str, filename: str):
        """Writes data from url to file."""
        if os.path.exists(os.path.join(save_path, filename)):
            logger.info("File %s already downloaded" % (os.path.join(save_path, filename)))
            return

        r = urllib.request.urlopen(url)
        logger.info("Downloading file at %s" % os.path.join(save_path, filename))

        def read_iter(file, block_size=1000):
            """Given a file 'file', returns an iterator that returns bytes of
            size 'blocksize' from the file, using read()."""
            while True:
                block = file.read(block_size)
                if not block:
                    break
                yield block

    def _add_cell_meta(self, cell_meta, filter=False):
        cell_ids = cell_meta.iloc[:,1].values
        share_index, meta_barcode_index, gene_barcode_index = \
            np.intersect1d(cell_ids,self.barcodes,return_indices=True)
        if len(share_index) <=1:
            logger.info("No consistent cell IDs!")
            return
        if len(share_index) < len(self.barcodes):
            logger.info("{} cells match metadata.".format(len(share_index)))
            return
