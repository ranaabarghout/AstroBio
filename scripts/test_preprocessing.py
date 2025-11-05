#!/usr/bin/env python3
"""
Test script to verify the data preprocessing pipeline works with a small sample.
"""

import logging

import cellxgene_census
import scanpy as sc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_census_connection():
    """Test basic connection to CellxGene Census."""
    logger.info("Testing CellxGene Census connection...")

    try:
        with cellxgene_census.open_soma(census_version="2025-01-30") as census:
            # Get a small sample of metadata
            cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
                value_filter="is_primary_data == True and disease == 'normal'",
                column_names=["soma_joinid", "cell_type", "tissue_general"]
            )

            cell_metadata = cell_metadata.concat()
            cell_metadata = cell_metadata.to_pandas()

            logger.info(f"Successfully retrieved {len(cell_metadata)} normal cells")
            logger.info(f"Sample cell types: {cell_metadata['cell_type'].value_counts().head()}")

            return True

    except (ConnectionError, RuntimeError, ValueError) as e:
        logger.error(f"Failed to connect to CellxGene Census: {e}")
        return False

def test_small_dataset():
    """Test processing a very small dataset."""
    logger.info("Testing small dataset processing...")

    try:
        with cellxgene_census.open_soma(census_version="2025-01-30") as census:
            # Get a tiny dataset for testing
            adata = cellxgene_census.get_anndata(
                census,
                organism="homo_sapiens",
                measurement_name="RNA",
                obs_value_filter="is_primary_data == True and disease == 'normal' and soma_joinid < 1000",
            )

            logger.info(f"Retrieved test dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")

            if adata.shape[0] == 0:
                logger.warning("No cells found in test dataset")
                return False

            # Test QC calculations
            adata.var["mt"] = adata.var_names.str.startswith("MT-")
            adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
            adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

            logger.info("QC metrics calculated successfully")
            logger.info(f"Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.1f}")
            logger.info(f"Mean total counts: {adata.obs['total_counts'].mean():.1f}")

            return True

    except (ConnectionError, RuntimeError, ValueError) as e:
        logger.error(f"Failed to process test dataset: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting data preprocessing pipeline tests...")

    # Test 1: Basic census connection
    if not test_census_connection():
        logger.error("Census connection test failed")
        return False

    # Test 2: Small dataset processing
    if not test_small_dataset():
        logger.error("Small dataset test failed")
        return False

    logger.info("All tests passed! The preprocessing pipeline should work correctly.")
    return True

if __name__ == "__main__":
    main()
