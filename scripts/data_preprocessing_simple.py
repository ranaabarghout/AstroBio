#!/usr/bin/env python3
"""
Simplified data preprocessing script for CellxGene Census data.

This script uses a simpler approach to avoid the soma_joinid mismatch issue.
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

# Quality control parameters
MIN_GENES_PER_CELL = 100
MIN_CELLS_PER_GENE = 3
TARGET_CELLS = 250000

# Output paths
DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_data_direct(target_cells, census_version=CENSUS_VERSION):
    """
    Retrieve data directly from CellxGene Census with a simpler approach.

    Args:
        target_cells: Target number of cells
        census_version: CellxGene Census version

    Returns:
        anndata.AnnData: Expression data with metadata
    """
    logger.info("Fetching data directly from CellxGene Census...")

    with cellxgene_census.open_soma(census_version=census_version) as census:
        # Get a reasonable number of cells directly
        # Use a simple soma_joinid range for reliability
        max_cells = min(target_cells * 3, 100000)  # Get 3x target, max 100k for memory

        adata = cellxgene_census.get_anndata(
            census,
            organism=ORGANISM,
            measurement_name=MEASUREMENT,
            obs_value_filter=f"is_primary_data == True and disease == 'normal' and soma_joinid < {max_cells * 10}",
            var_value_filter="feature_id != ''",
        )

        logger.info(f"Retrieved {adata.shape[0]} cells x {adata.shape[1]} genes")

        # If we got more than we need, subsample
        if adata.shape[0] > target_cells * 2:
            logger.info(f"Subsampling to {target_cells * 2} cells...")
            np.random.seed(SEED)
            sample_indices = np.random.choice(adata.shape[0], size=target_cells * 2, replace=False)
            adata = adata[sample_indices].copy()
            logger.info(f"Subsampled to {adata.shape[0]} cells")

    return adata


def calculate_qc_metrics(adata):
    """Calculate quality control metrics."""
    logger.info("Calculating quality control metrics...")

    # Define gene sets for QC
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=True
    )

    logger.info("QC metrics calculated successfully")
    logger.info(f"Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.1f}")
    logger.info(f"Mean total counts per cell: {adata.obs['total_counts'].mean():.1f}")
    logger.info(f"Mean mitochondrial percentage: {adata.obs['pct_counts_mt'].mean():.1f}%")


def apply_basic_filters(adata, min_genes=MIN_GENES_PER_CELL, min_cells=MIN_CELLS_PER_GENE):
    """Apply basic quality control filters."""
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


def detect_doublets(adata):
    """Detect doublets using Scrublet."""
    logger.info("Running doublet detection...")

    try:
        # Try with dataset_id if available, otherwise skip batch correction
        batch_key = "dataset_id" if "dataset_id" in adata.obs.columns else None
        sc.pp.scrublet(adata, batch_key=batch_key)
        n_doublets = adata.obs["predicted_doublet"].sum()
        logger.info(f"Detected {n_doublets} doublets ({n_doublets/len(adata)*100:.1f}%)")
    except (KeyError, ValueError, RuntimeError) as e:
        logger.warning(f"Doublet detection failed: {e}")
        logger.warning("Continuing without doublet detection...")


def normalize_data(adata):
    """Normalize the expression data."""
    logger.info("Normalizing expression data...")

    # Save raw counts
    adata.layers["counts"] = adata.X.copy()

    # Normalize to median total counts
    sc.pp.normalize_total(adata)

    # Log transform
    sc.pp.log1p(adata)

    logger.info("Normalization completed")


def subsample_data(adata, target_cells=TARGET_CELLS, random_state=SEED):
    """Subsample the data to target number of cells."""
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
    """Save the processed data."""
    logger.info(f"Saving processed data to {output_path}")
    adata.write(output_path)
    logger.info("Data saved successfully")


def generate_qc_plots(adata, output_dir):
    """Generate quality control plots."""
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
    parser = argparse.ArgumentParser(description="Process CellxGene Census data (simplified)")
    parser.add_argument("--target-cells", type=int, default=TARGET_CELLS,
                       help="Target number of cells to subsample")
    parser.add_argument("--min-genes", type=int, default=MIN_GENES_PER_CELL,
                       help="Minimum genes per cell")
    parser.add_argument("--min-cells", type=int, default=MIN_CELLS_PER_GENE,
                       help="Minimum cells per gene")
    parser.add_argument("--output-prefix", type=str, default="processed_data_simple",
                       help="Output file prefix")

    args = parser.parse_args()

    logger.info("Starting simplified data preprocessing pipeline...")

    # Step 1: Get data directly
    adata = get_data_direct(args.target_cells)

    if adata.shape[0] == 0:
        logger.error("No cells retrieved! Check your filtering criteria.")
        return

    logger.info(f"Initial dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Step 2: Calculate QC metrics
    calculate_qc_metrics(adata)

    # Step 3: Apply basic filters
    apply_basic_filters(adata, min_genes=args.min_genes, min_cells=args.min_cells)

    if adata.shape[0] == 0:
        logger.error("No cells remaining after filtering!")
        return

    # Step 4: Detect doublets
    detect_doublets(adata)

    # Step 5: Normalize data
    normalize_data(adata)

    # Step 6: Subsample if needed
    adata = subsample_data(adata, target_cells=args.target_cells)

    # Step 7: Generate QC plots
    plots_dir = DATA_DIR / "figures"
    plots_dir.mkdir(exist_ok=True)
    generate_qc_plots(adata, plots_dir)

    # Step 8: Save data
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
