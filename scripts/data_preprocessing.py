#!/usr/bin/env python3
"""
Data preprocessing script for CellxGene Census data.

This script:
1. Pulls metadata and expression data from CellxGene Census
2. Filters for normal (non-diseased) tissue
3. Applies quality control metrics
4. Performs doublet detection
5. Normalizes the data
6. Subsamples to ~250,000 cells
7. Saves the processed data

Usage:
    python scripts/data_preprocessing.py
"""

import argparse
import logging
from pathlib import Path

import cellxgene_census
import numpy as np
import scanpy as sc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ORGANISM = "homo_sapiens"
MEASUREMENT = "RNA"
CENSUS_VERSION = "2025-01-30"
SEED = 42

# Set random seeds for reproducibility
np.random.seed(SEED)

# Metadata fields to include
METADATA_FIELDS = [
    "assay",
    "dataset_id",
    "cell_type",
    "development_stage",
    "disease",
    "self_reported_ethnicity",
    "sex",
    "tissue_general",
    "tissue",
    "soma_joinid"  # Need this for joining with expression data
]

# Quality control parameters
MIN_GENES_PER_CELL = 100
MIN_CELLS_PER_GENE = 3
TARGET_CELLS = 250000

EMBEDDING_NAME = "geneformer"

# Output paths
DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_normal_cell_metadata(census_version=CENSUS_VERSION):
    """
    Retrieve metadata for normal (non-diseased) cells from CellxGene Census.

    Returns:
        pd.DataFrame: Filtered metadata for normal cells
    """
    logger.info("Fetching metadata for normal cells from CellxGene Census...")

    with cellxgene_census.open_soma(census_version=census_version) as census:
        # Filter for primary data and normal tissue
        cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
            value_filter="is_primary_data == True and disease == 'normal'",
            column_names=METADATA_FIELDS
        )

        cell_metadata = cell_metadata.concat()
        cell_metadata = cell_metadata.to_pandas()

    logger.info(f"Found {len(cell_metadata)} normal cells")
    return cell_metadata


def get_expression_data(soma_joinids, census_version=CENSUS_VERSION):
    """
    Retrieve expression data for specified cells.

    Args:
        soma_joinids: List of soma_joinid values to retrieve
        census_version: CellxGene Census version

    Returns:
        anndata.AnnData: Expression data
    """
    logger.info(f"Fetching expression data for {len(soma_joinids)} cells...")

    # Create obs filter for the specific soma_joinids
    # Convert to sorted list for better performance
    sorted_joinids = sorted(list(soma_joinids))

    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism=ORGANISM,
            measurement_name=MEASUREMENT,
            obs_coords=sorted_joinids,
            var_value_filter="feature_type=='protein_coding'",
            obs_column_names=METADATA_FIELDS,  # Include metadata columns,
            obs_embeddings=[EMBEDDING_NAME]
        )

    logger.info(f"Retrieved expression data: {adata.shape[0]} cells x {adata.shape[1]} genes")

    if adata.shape[0] == 0:
        logger.error("No expression data retrieved! This might indicate an issue with soma_joinid filtering.")
        logger.info(f"Sample soma_joinids requested: {sorted_joinids[:10]}")

    return adata


def calculate_qc_metrics(adata):
    """
    Calculate quality control metrics for the AnnData object.

    Args:
        adata: AnnData object
    """
    logger.info("Calculating quality control metrics...")

    # Define gene sets for QC
    # Mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # Ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # Hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=True
    )

    logger.info("QC metrics calculated successfully")

    # Log some basic stats
    logger.info(f"Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.1f}")
    logger.info(f"Mean total counts per cell: {adata.obs['total_counts'].mean():.1f}")
    logger.info(f"Mean mitochondrial percentage: {adata.obs['pct_counts_mt'].mean():.1f}%")


def apply_basic_filters(adata, min_genes=MIN_GENES_PER_CELL, min_cells=MIN_CELLS_PER_GENE):
    """
    Apply basic quality control filters.

    Args:
        adata: AnnData object
        min_genes: Minimum genes per cell
        min_cells: Minimum cells per gene
    """
    logger.info("Applying basic quality control filters...")

    n_cells_before = adata.shape[0]
    n_genes_before = adata.shape[1]

    # Filter cells with too few genes
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Filter genes expressed in too few cells
    sc.pp.filter_genes(adata, min_cells=min_cells)

    n_cells_after = adata.shape[0]
    n_genes_after = adata.shape[1]

    logger.info(f"Filtered cells: {n_cells_before} -> {n_cells_after} ({n_cells_before - n_cells_after} removed)")
    logger.info(f"Filtered genes: {n_genes_before} -> {n_genes_after} ({n_genes_before - n_genes_after} removed)")


def detect_doublets(adata, batch_key="dataset_id"):
    """
    Detect doublets using Scrublet.

    Args:
        adata: AnnData object
        batch_key: Key for batch information
    """
    logger.info("Running doublet detection...")

    try:
        sc.pp.scrublet(adata, batch_key=batch_key)
        n_doublets = adata.obs["predicted_doublet"].sum()
        logger.info(f"Detected {n_doublets} doublets ({n_doublets/len(adata)*100:.1f}%)")
    except (KeyError, ValueError, RuntimeError) as e:
        logger.warning(f"Doublet detection failed: {e}")
        logger.warning("Continuing without doublet detection...")


def normalize_data(adata):
    """
    Normalize the expression data.

    Args:
        adata: AnnData object
    """
    logger.info("Normalizing expression data...")

    # Save raw counts
    adata.layers["counts"] = adata.X.copy()

    # Normalize to median total counts
    sc.pp.normalize_total(adata)

    # Log transform
    sc.pp.log1p(adata)

    logger.info("Normalization completed")


def subsample_data(adata, target_cells=TARGET_CELLS, random_state=SEED):
    """
    Subsample the data to target number of cells.

    Args:
        adata: AnnData object
        target_cells: Target number of cells
        random_state: Random seed

    Returns:
        anndata.AnnData: Subsampled data
    """
    if len(adata) <= target_cells:
        logger.info(f"Dataset has {len(adata)} cells, no subsampling needed")
        return adata

    logger.info(f"Subsampling from {len(adata)} to {target_cells} cells...")

    # Random subsampling
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(adata), size=target_cells, replace=False)
    adata_subset = adata[sample_indices].copy()

    logger.info(f"Subsampled to {len(adata_subset)} cells")
    return adata_subset


def save_data(adata, output_path):
    """
    Save the processed data.

    Args:
        adata: AnnData object
        output_path: Path to save the data
    """
    logger.info(f"Saving processed data to {output_path}")
    adata.write(output_path)
    logger.info("Data saved successfully")


def generate_qc_plots(adata, output_dir):
    """
    Generate quality control plots.

    Args:
        adata: AnnData object
        output_dir: Directory to save plots
    """
    logger.info("Generating QC plots...")

    # Set scanpy settings
    sc.settings.figdir = output_dir
    sc.settings.set_figure_params(dpi=80, facecolor='white')

    # Violin plots
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        save="_qc_violin.pdf"
    )

    # Scatter plot
    sc.pl.scatter(
        adata,
        "total_counts",
        "n_genes_by_counts",
        color="pct_counts_mt",
        save="_qc_scatter.pdf"
    )

    logger.info(f"QC plots saved to {output_dir}")


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description="Process CellxGene Census data")
    parser.add_argument("--target-cells", type=int, default=TARGET_CELLS,
                       help="Target number of cells to subsample")
    parser.add_argument("--min-genes", type=int, default=MIN_GENES_PER_CELL,
                       help="Minimum genes per cell")
    parser.add_argument("--min-cells", type=int, default=MIN_CELLS_PER_GENE,
                       help="Minimum cells per gene")
    parser.add_argument("--output-prefix", type=str, default="processed_data",
                       help="Output file prefix")

    args = parser.parse_args()

    logger.info("Starting data preprocessing pipeline...")

    # Step 1: Get metadata for normal cells
    metadata = get_normal_cell_metadata()

    # If we have too many cells, subsample metadata first to avoid memory issues
    if len(metadata) > args.target_cells * 2:  # Get 2x target to account for QC filtering
        logger.info(f"Pre-filtering metadata to {args.target_cells * 2} cells to manage memory...")
        metadata = metadata.sample(n=args.target_cells * 2, random_state=SEED)

    # Step 2: Get expression data
    adata = get_expression_data(metadata["soma_joinid"].tolist())

    # Check if we got any data
    if adata.shape[0] == 0:
        logger.error("No expression data retrieved. Trying alternative approach...")

        # Try getting data without specific soma_joinid filtering
        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            adata = cellxgene_census.get_anndata(
                census,
                organism=ORGANISM,
                measurement_name=MEASUREMENT,
                obs_value_filter="is_primary_data == True and disease == 'normal'",
                var_value_filter="feature_id != ''",
                column_names=METADATA_FIELDS
            )

        logger.info(f"Alternative retrieval: {adata.shape[0]} cells x {adata.shape[1]} genes")

        # If we still have too many cells, subsample
        if adata.shape[0] > args.target_cells * 2:
            logger.info(f"Subsampling expression data to {args.target_cells * 2} cells...")
            np.random.seed(SEED)
            sample_indices = np.random.choice(adata.shape[0], size=args.target_cells * 2, replace=False)
            adata = adata[sample_indices].copy()

    # Step 3: Add metadata to adata.obs (metadata should already be included)
    logger.info("Checking metadata integration...")

    # Check if metadata columns are already present
    missing_cols = [col for col in METADATA_FIELDS if col not in adata.obs.columns and col != "soma_joinid"]
    if missing_cols:
        logger.info(f"Adding missing metadata columns: {missing_cols}")
        # If metadata is missing, merge it
        metadata_indexed = metadata.set_index("soma_joinid")
        common_cells = adata.obs.index.intersection(metadata_indexed.index)

        if len(common_cells) == 0:
            logger.warning("No common cells found between metadata and expression data!")
            # Use adata as is, it should have metadata from get_anndata
        else:
            adata = adata[common_cells].copy()
            for col in missing_cols:
                if col in metadata_indexed.columns:
                    adata.obs[col] = metadata_indexed.loc[adata.obs.index, col]

    logger.info(f"Final dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Early exit if no cells
    if adata.shape[0] == 0:
        logger.error("No cells remaining after data retrieval and filtering!")
        logger.error("Please check your filtering criteria or try a different approach.")
        return

    # Step 4: Calculate QC metrics
    calculate_qc_metrics(adata)

    # Step 5: Apply basic filters
    apply_basic_filters(adata, min_genes=args.min_genes, min_cells=args.min_cells)

    # Step 6: Detect doublets
    detect_doublets(adata)

    # Step 7: Normalize data
    normalize_data(adata)

    # Step 8: Subsample if needed
    adata = subsample_data(adata, target_cells=args.target_cells)

    # Step 9: Generate QC plots
    plots_dir = DATA_DIR / "figures"
    plots_dir.mkdir(exist_ok=True)
    generate_qc_plots(adata, plots_dir)

    # Step 10: Save data
    output_path = DATA_DIR / f"{args.output_prefix}.h5ad"
    save_data(adata, output_path)

    # Save metadata summary
    metadata_summary = adata.obs.describe()
    metadata_summary.to_csv(DATA_DIR / f"{args.output_prefix}_metadata_summary.csv")

    logger.info("Pipeline completed successfully!")
    logger.info(f"Final dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
