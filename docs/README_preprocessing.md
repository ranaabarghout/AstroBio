# Data Preprocessing Pipeline

This repository contains scripts for preprocessing single-cell RNA-seq data from CellxGene Census.

## Setup

The project uses `uv` for dependency management. Make sure uv is installed and the environment is set up:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Sync dependencies
uv sync

# Install Jupyter kernel
uv run python -m ipykernel install --user --name=astrobio-uv --display-name="AstroBio (uv)"
```

## Scripts

### `scripts/data_preprocessing.py`

Main preprocessing pipeline that:

1. **Data Retrieval**: Pulls metadata and expression data from CellxGene Census
2. **Filtering**: Selects only normal (non-diseased) tissue samples
3. **Quality Control**: Calculates QC metrics including:
   - Mitochondrial gene percentage (`MT-` genes)
   - Ribosomal gene percentage (`RPS`, `RPL` genes)
   - Hemoglobin gene percentage (`HB` genes)
4. **Basic Filtering**: Removes cells with < 100 genes and genes expressed in < 3 cells
5. **Doublet Detection**: Uses Scrublet to identify potential doublets
6. **Normalization**: Applies count depth normalization and log1p transformation
7. **Subsampling**: Randomly subsamples to target cell count (default: 250,000)
8. **Output**: Saves processed data and QC plots

#### Usage

```bash
# Basic usage with default parameters
uv run python scripts/data_preprocessing.py

# Custom parameters
uv run python scripts/data_preprocessing.py \
    --target-cells 100000 \
    --min-genes 200 \
    --min-cells 5 \
    --output-prefix my_data
```

#### Parameters

- `--target-cells`: Target number of cells to subsample (default: 250000)
- `--min-genes`: Minimum genes per cell (default: 100)
- `--min-cells`: Minimum cells per gene (default: 3)
- `--output-prefix`: Output file prefix (default: "processed_data")

#### Output Files

- `data/processed/{prefix}.h5ad`: Processed AnnData object
- `data/processed/{prefix}_metadata_summary.csv`: Summary statistics
- `data/processed/figures/`: QC plots (violin plots, scatter plots)

### `scripts/test_preprocessing.py`

Test script to verify the pipeline setup works correctly.

```bash
uv run python scripts/test_preprocessing.py
```

## Data Structure

The processed AnnData object contains:

### Observations (`.obs`)
- `cell_type`: Cell type annotation
- `tissue_general`: General tissue category
- `assay`: Assay type used
- `disease`: Disease status (filtered to 'normal' only)
- `sex`: Biological sex
- `development_stage`: Developmental stage
- `self_reported_ethnicity`: Self-reported ethnicity
- `dataset_id`: Source dataset identifier
- `n_genes_by_counts`: Number of genes with counts > 0
- `total_counts`: Total UMI counts per cell
- `pct_counts_mt`: Percentage of mitochondrial counts
- `pct_counts_ribo`: Percentage of ribosomal counts
- `pct_counts_hb`: Percentage of hemoglobin counts
- `predicted_doublet`: Doublet prediction (if available)
- `doublet_score`: Doublet score (if available)

### Variables (`.var`)
- `mt`: Boolean indicator for mitochondrial genes
- `ribo`: Boolean indicator for ribosomal genes
- `hb`: Boolean indicator for hemoglobin genes
- `n_cells_by_counts`: Number of cells expressing each gene

### Layers
- `counts`: Raw count data (before normalization)

### Expression Matrix (`.X`)
- Normalized and log-transformed expression data

## Quality Control Details

The pipeline implements standard single-cell QC practices:

1. **Gene Set Annotations**:
   - Mitochondrial genes: Start with "MT-"
   - Ribosomal genes: Start with "RPS" or "RPL"
   - Hemoglobin genes: Match pattern "^HB[^(P)]"

2. **QC Metrics**: Calculated using `scanpy.pp.calculate_qc_metrics()`
   - Total counts per cell
   - Number of genes detected per cell
   - Percentage of counts from specific gene sets

3. **Basic Filtering**:
   - Remove cells with < 100 genes (adjustable)
   - Remove genes expressed in < 3 cells (adjustable)

4. **Doublet Detection**:
   - Uses Scrublet algorithm
   - Accounts for batch effects using dataset_id

5. **Normalization**:
   - Count depth scaling to median total counts
   - Natural log transformation: log(x + 1)

## Memory Considerations

- The pipeline pre-filters metadata if > 500k cells to manage memory
- Consider adjusting `--target-cells` based on available RAM
- Large datasets may require HPC resources

## Troubleshooting

1. **Connection Issues**: Ensure internet connection for CellxGene Census access
2. **Memory Errors**: Reduce `--target-cells` parameter
3. **Doublet Detection Failures**: Pipeline continues without doublet detection if it fails
4. **No Data Found**: Check that normal tissue data exists in the census version

## Next Steps

After preprocessing, the data is ready for:
- Dimensionality reduction (PCA, UMAP)
- Clustering analysis
- Differential expression analysis
- Cell type annotation
- Trajectory analysis

Use the processed `.h5ad` file with scanpy, scvi-tools, or other single-cell analysis packages.
