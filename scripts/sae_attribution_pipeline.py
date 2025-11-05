#!/usr/bin/env python3
"""
SAE Feature-Attribution Analysis Pipeline

This script integrates:
1. CellxGene Census data retrieval (geneformer embeddings + metadata)
2. Trained Sparse Autoencoder feature extraction
3. Comprehensive feature-attribution correlation analysis

Usage:
    python scripts/sae_attribution_pipeline.py --sample-size 10000 --output-dir results/
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import cellxgene_census
import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import torch

# Local imports
sys.path.append('src')

from feature_attribution_analysis import FeatureAttributionAnalyzer
from models import SparseAutoencoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ORGANISM = "homo_sapiens"
MEASUREMENT = "RNA"
CENSUS_VERSION = "2025-01-30"
EMBEDDING_NAME = "geneformer"

# Metadata fields to collect (comprehensive set from correlation notebook)
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


def get_census_data_with_embeddings(sample_size, census_version=CENSUS_VERSION):
    """
    Retrieve cell data with embeddings and comprehensive metadata from CellxGene Census.
    Uses the same approach as the correlation set notebook for consistency.
    """
    logger.info(f"Fetching {sample_size} cells with {EMBEDDING_NAME} embeddings and comprehensive metadata...")

    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism=ORGANISM,
            measurement_name=MEASUREMENT,
            obs_value_filter=f"soma_joinid < {sample_size}",  # Simple filtering approach
            var_value_filter="feature_type=='protein_coding'",
            obs_embeddings=[EMBEDDING_NAME],
            obs_column_names=METADATA_FIELDS,  # Explicit metadata collection
        )

        logger.info(f"Retrieved {adata.shape[0]} cells x {adata.shape[1]} genes")

        # Set feature names properly
        adata.var_names = adata.var["feature_name"]

    return adata


def load_sae_model(model_dir="models"):
    """
    Load trained Sparse Autoencoder model and hyperparameters.
    """
    logger.info("Loading trained SAE model...")

    # Load hyperparameters
    params_path = Path(model_dir) / "best_sparse_autoencoder_params.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    logger.info(f"SAE hyperparameters: {params}")

    # Initialize model
    sae_model = SparseAutoencoder(
        input_dim=512,  # Geneformer embedding dimension
        hidden_dim=params["hidden_dim"],
        expanded_ratio=params["expanded_ratio"],
        n_encoder_layers=params["n_encoder_layers"],
        n_decoder_layers=params["n_decoder_layers"]
    )

    # Load weights
    checkpoint_path = Path(model_dir) / "best_sparse_autoencoder.pt"
    sae_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    sae_model.eval()

    logger.info(f"Loaded SAE model from: {checkpoint_path}")
    logger.info(f"SAE expanded dimension: {sae_model.expanded_dim}")
    logger.info(f"SAE hidden dimension: {sae_model.hidden_dim}")

    return sae_model, params


def extract_sae_features(embeddings, model):
    """
    Extract SAE latent features from input embeddings.
    """
    logger.info("Extracting SAE latent features...")

    model.eval()

    with torch.no_grad():
        # Convert to tensor
        embeddings_tensor = torch.FloatTensor(embeddings)

        # Get latent features (encoder output)
        latent_features = model.encoder(embeddings_tensor)

        # Apply ReLU activation (sparsity)
        latent_features = torch.relu(latent_features)

    features = latent_features.numpy()
    sparsity = (features == 0).mean()

    logger.info(f"SAE features shape: {features.shape}")
    logger.info(f"SAE feature sparsity: {sparsity:.2%}")

    return features


def prepare_comprehensive_attributions(adata):
    """
    Prepare comprehensive cell attributions for analysis using the correlation set approach.

    This includes:
    - Technical QC metrics (library size, gene counts, mitochondrial/ribosomal content)
    - Cell cycle scoring using Hallmark gene sets
    - Pathway activity via ssGSEA (inflammation, hypoxia, apoptosis, etc.)
    - Existing metadata (cell type, tissue, disease, etc.)
    """
    logger.info("Preparing comprehensive biological attributions...")

    # Initialize comprehensive annotations dataframe
    annotations_df = pd.DataFrame(index=adata.obs_names)

    # Add existing metadata fields
    for field in METADATA_FIELDS:
        if field in adata.obs.columns:
            annotations_df[field] = adata.obs[field]

    logger.info(f"Added {len([f for f in METADATA_FIELDS if f in adata.obs.columns])} metadata fields")

    # Compute technical metrics
    logger.info("Computing technical QC metrics...")

    # Get expression matrix
    if issparse(adata.X):
        x = adata.X.toarray()
    else:
        x = adata.X

    # Basic technical metrics
    annotations_df['n_counts'] = x.sum(axis=1)
    annotations_df['n_genes'] = (x > 0).sum(axis=1)

    # Mitochondrial gene percentage
    mito_genes = adata.var_names.str.upper().str.startswith("MT-")
    annotations_df['pct_mito'] = x[:, mito_genes].sum(axis=1) / annotations_df['n_counts'] * 100

    # Ribosomal gene percentage
    ribo_genes = adata.var_names.str.startswith(("RPS","RPL"))
    annotations_df['pct_ribo'] = x[:, ribo_genes].sum(axis=1) / annotations_df['n_counts'] * 100

    logger.info(f"Technical metrics: mean counts={annotations_df['n_counts'].mean():.0f}, "
               f"mean genes={annotations_df['n_genes'].mean():.0f}")

    # Cell cycle scoring
    logger.info("Computing cell cycle scores...")
    try:
        hallmark_genesets = gp.get_library(name='MSigDB_Hallmark_2020', organism='Human')
        s_genes = [g for g in hallmark_genesets['E2F Targets'] if g in adata.var_names]
        g2m_genes = [g for g in hallmark_genesets['G2-M Checkpoint'] if g in adata.var_names]

        if len(s_genes) > 0 and len(g2m_genes) > 0:
            sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes, copy=False)
            annotations_df['S_score'] = adata.obs['S_score']
            annotations_df['G2M_score'] = adata.obs['G2M_score']
            annotations_df['phase'] = adata.obs['phase']
            logger.info(f"Cell cycle phases: {adata.obs['phase'].value_counts().to_dict()}")
        else:
            logger.warning("Insufficient cell cycle genes found, skipping cell cycle scoring")
    except Exception as e:
        logger.warning(f"Cell cycle scoring failed: {e}")

    # Pathway activity analysis via ssGSEA
    logger.info("Computing pathway activity scores...")
    try:
        # Define key biological pathways to analyze
        marker_sets = {
            "Inflammation_Response": hallmark_genesets['Inflammatory Response'],
            "Hypoxia": hallmark_genesets['Hypoxia'],
            "Apoptosis": hallmark_genesets['Apoptosis'],
            "DNA_Repair": hallmark_genesets['DNA Repair'],
            "Oxidative_Phosphorylation": hallmark_genesets['Oxidative Phosphorylation']
        }

        # Prepare expression data for ssGSEA
        expr_df = pd.DataFrame(x.T, index=adata.var_names, columns=adata.obs_names)

        # Compute ssGSEA scores for each pathway
        for pathway_name, genes in marker_sets.items():
            genes_present = [g for g in genes if g in adata.var_names]
            if len(genes_present) == 0:
                logger.warning(f"No genes from {pathway_name} found in dataset, skipping")
                continue

            try:
                ss_res = gp.ssgsea(
                    data=expr_df,
                    gene_sets={pathway_name: genes_present},
                    sample_norm_method="rank",
                    outdir=None,
                    no_plot=True,
                    permutation_num=0
                )

                # Extract scores and add to annotations
                sample_names = ss_res.res2d['Name']
                if sample_names.dtype != expr_df.columns.dtype:
                    sample_names = sample_names.astype(expr_df.columns.dtype)
                nes_series = pd.Series(data=ss_res.res2d['NES'].values, index=sample_names)

                annotations_df[pathway_name] = annotations_df.index.map(lambda i: nes_series[i])
                logger.info(f"Computed {pathway_name} scores (mean={annotations_df[pathway_name].mean():.3f})")

            except Exception as e:
                logger.warning(f"Error computing {pathway_name}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Pathway analysis failed: {e}")

    # Add all annotations to adata.obs
    for col in annotations_df.columns:
        adata.obs[col] = annotations_df[col]

    logger.info(f"Final annotations: {annotations_df.shape[1]} total attributes")

    # Smart categorization based on biological knowledge
    binary_attrs = []
    categorical_attrs = []
    continuous_attrs = []

    # Predefined categories based on biological knowledge
    known_categorical = [
        'cell_type', 'tissue_general', 'tissue', 'disease', 'development_stage',
        'assay', 'dataset_id', 'self_reported_ethnicity', 'phase'
    ]
    known_continuous = [
        'n_counts', 'n_genes', 'pct_mito', 'pct_ribo', 'S_score', 'G2M_score'
    ] + [col for col in adata.obs.columns if any(x in col for x in ['Inflammation', 'Hypoxia', 'Apoptosis', 'DNA_Repair', 'Oxidative'])]

    # Categorize based on data properties and biological knowledge
    for col in adata.obs.columns:
        if col.startswith('_') or col == 'soma_joinid':  # Skip internal columns
            continue

        n_unique = adata.obs[col].nunique()
        dtype = adata.obs[col].dtype
        missing_frac = adata.obs[col].isna().mean()

        # Skip if too many missing values
        if missing_frac > 0.5:
            continue

        # Categorize based on biological knowledge and data properties
        if col in known_continuous:
            continuous_attrs.append(col)
        elif col in known_categorical:
            if n_unique <= 50:  # Reasonable number of categories
                categorical_attrs.append(col)
        elif n_unique == 2 and dtype == 'object':
            binary_attrs.append(col)
        elif 2 < n_unique <= 50 and dtype == 'object':
            categorical_attrs.append(col)
        elif dtype in ['float64', 'int64'] and n_unique > 10:
            continuous_attrs.append(col)

    logger.info(f"Categorized attributes: {len(binary_attrs)} binary, "
               f"{len(categorical_attrs)} categorical, {len(continuous_attrs)} continuous")

    # Create attribution dictionary
    attributions = {
        'binary': {attr: adata.obs[attr] for attr in binary_attrs},
        'categorical': {attr: adata.obs[attr] for attr in categorical_attrs},
        'continuous': {attr: adata.obs[attr] for attr in continuous_attrs}
    }

    # Create confounders for residualization (technical factors)
    confounder_cols = ['n_counts', 'n_genes']
    if 'dataset_id' in adata.obs.columns:
        # One-hot encode dataset_id for batch effect control
        dataset_dummies = pd.get_dummies(adata.obs['dataset_id'], prefix='dataset')
        confounders = pd.concat([adata.obs[confounder_cols], dataset_dummies], axis=1)
    else:
        confounders = adata.obs[confounder_cols]

    return attributions, confounders


def save_results(results, sae_features, adata, output_dir):
    """
    Save analysis results.
    """
    logger.info(f"Saving results to {output_dir}...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save SAE features
    np.save(output_path / 'sae_features.npy', sae_features)

    # Save analysis results as JSON
    def make_json_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

    json_results = make_json_serializable(results)
    with open(output_path / 'analysis_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Save metadata
    adata.obs.to_csv(output_path / 'cell_metadata.csv')

    # Create summary report
    summary = create_summary_report(results, sae_features)
    with open(output_path / 'summary_report.txt', 'w') as f:
        f.write(summary)

    logger.info(f"Results saved to: {output_path}")


def create_summary_report(results, sae_features):
    """
    Create a summary report of the analysis.
    """
    report = []
    report.append("=== SAE FEATURE-ATTRIBUTION ANALYSIS SUMMARY ===\n")

    # Basic statistics
    report.append(f"SAE Features: {sae_features.shape[1]}")
    report.append(f"Cells Analyzed: {sae_features.shape[0]}")
    report.append(f"Overall Sparsity: {(sae_features == 0).mean():.2%}\n")

    # Univariate results
    report.append("=== UNIVARIATE ASSOCIATIONS ===")

    univar = results.get('univariate', {})
    total_significant = 0

    for attr_type in ['binary', 'categorical', 'continuous']:
        if attr_type in univar and len(univar[attr_type]) > 0:
            report.append(f"\n{attr_type.upper()} ATTRIBUTIONS:")

            type_significant = 0
            for attr_name, attr_results in univar[attr_type].items():
                if len(attr_results) > 0:
                    sig_count = sum(1 for result in attr_results if result.get('q_value', 1.0) < 0.05)
                    total_count = len(attr_results)
                    type_significant += sig_count

                    report.append(f"  {attr_name}: {sig_count}/{total_count} significant features")

            report.append(f"  Total significant: {type_significant}")
            total_significant += type_significant

    report.append(f"\nOverall significant associations: {total_significant}")

    # Linear probe results
    if 'multivariate' in results and 'linear_probes' in results['multivariate']:
        report.append("\n=== LINEAR PROBE PERFORMANCE ===")

        probes = results['multivariate']['linear_probes']
        for attr_name, probe_result in probes.items():
            if 'auc' in probe_result:
                report.append(f"{attr_name}: AUC = {probe_result['auc']:.3f}")
            elif 'r2' in probe_result:
                report.append(f"{attr_name}: R² = {probe_result['r2']:.3f}")

    return "\n".join(report)


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="SAE Feature-Attribution Analysis Pipeline")
    parser.add_argument("--sample-size", type=int, default=10000,
                       help="Number of cells to analyze")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory containing trained SAE model")
    parser.add_argument("--output-dir", type=str, default="data/processed/sae_attribution_analysis",
                       help="Output directory for results")
    parser.add_argument("--census-version", type=str, default=CENSUS_VERSION,
                       help="CellxGene Census version")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("Starting SAE Feature-Attribution Analysis Pipeline...")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Step 1: Load data from CellxGene Census
        adata = get_census_data_with_embeddings(args.sample_size, args.census_version)

        # Step 2: Load trained SAE model
        sae_model, sae_params = load_sae_model(args.model_dir)

        # Step 3: Extract SAE features
        geneformer_embeddings = adata.obsm[EMBEDDING_NAME]
        sae_features = extract_sae_features(geneformer_embeddings, sae_model)

        # Step 4: Prepare comprehensive attributions
        attributions, confounders = prepare_comprehensive_attributions(adata)

        # Step 5: Run analysis
        logger.info("Running comprehensive feature-attribution analysis...")
        analyzer = FeatureAttributionAnalyzer(
            n_permutations=1000,
            fdr_method='fdr_bh',
            cv_folds=5,
            random_state=42
        )

        results = analyzer.analyze_associations(
            features=sae_features,
            attributions=attributions,
            confounders=confounders.values if confounders.shape[1] > 0 else None
        )

        # Step 6: Save results
        save_results(results, sae_features, adata, args.output_dir)

        logger.info("Analysis completed successfully!")

        # Print summary
        summary = create_summary_report(results, sae_features)
        print("\n" + summary)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
