# AstroBio

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A data science project for analyzing single-cell RNA sequencing data from the CZ CELLxGENE Census, focusing on SC research applications.

## Features

- **UV Environment Management**: Fast Python package management with UV
- **CELLxGENE Census Integration**: Scripts to download and analyze large-scale single-cell RNA-seq data
- **Data Pipeline**: Structured workflow for data processing and analysis
- **Jupyter Notebooks**: Interactive analysis and visualization tools

## Quick Start

### Prerequisites

- Python 3.13+
- UV package manager
- Internet connection for downloading Census data

### Installation

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Clone and setup the project**:
   ```bash
   git clone <repository-url>
   cd AstroBio
   ```

3. **Create and activate UV environment**:
   ```bash
   uv sync
   ```

4. **Test the installation**:
   ```bash
   uv run scripts/simple_cellxgene_test.py
   ```

### Running Scripts

- **Test CELLxGENE setup**: `uv run scripts/simple_cellxgene_test.py`
- **Download Census data**: `uv run scripts/download_cellxgene_data.py`
- **Launch Jupyter**: `uv run jupyter lab notebooks/`

## TODO
 - [x] Set up UV environment management
 - [x] Add CELLxGENE Census integration
 - [x] Create data download scripts
 - [ ] Structure src better
 - [ ] Add ruff/linter settings
 - [ ] Improve org data
 - [ ] Autorunning test

## Scripts Overview

### Data Download Scripts

#### `scripts/simple_cellxgene_test.py`
A lightweight test script that:
- Tests basic package imports (pandas, cellxgene-census)
- Creates sample data for testing
- Attempts to connect to CELLxGENE Census with timeout handling
- Generates mock data if census access fails
- **Usage**: `uv run scripts/simple_cellxgene_test.py`

#### `scripts/download_cellxgene_data.py`
Main data download script that:
- Queries cell metadata from CELLxGENE Census
- Downloads gene expression data for specific genes
- Filters for specific cell types (neurons, microglia) and conditions
- Saves data in multiple formats (CSV, H5AD)
- **Usage**: `uv run scripts/download_cellxgene_data.py`

### Data Files

The scripts generate several data files in `data/raw/`:
- `test_data.csv`: Sample data for testing
- `cell_metadata_*.csv`: Cell metadata from Census
- `expression_data_*.h5ad`: Gene expression data in AnnData format
- `gene_info.csv`: Gene annotation information
- `download_summary.txt`: Summary of downloaded data

## UV Environment Details

This project uses UV for fast and reliable Python package management. The environment includes:

**Core Data Science Packages:**
- `pandas`, `numpy`, `scipy`: Data manipulation and analysis
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Machine learning
- `jupyterlab`, `notebook`: Interactive analysis

**Single-cell Analysis:**
- `cellxgene-census`: Access to CZ CELLxGENE Census data
- `anndata`: Annotated data matrices
- `scanpy`: Single-cell analysis toolkit

**Development Tools:**
- `ruff`: Fast Python linter and formatter
- `pytest`: Testing framework
- `python-dotenv`: Environment variable management

### UV Commands

```bash
# Sync environment with pyproject.toml
uv sync

# Add new package
uv add package-name

# Run script with UV environment
uv run script.py

# Install development dependencies
uv sync --dev

# Check installed packages
uv pip list

# Update all packages
uv lock --upgrade
```

## Troubleshooting

### Common Issues

1. **TileDB Context Error**: If you see TileDB errors when accessing Census data:
   - This is often due to network or system configuration
   - The test script will generate mock data as fallback
   - Try running from a different network or system

2. **Disk Space Issues**: Large datasets require significant storage:
   - Monitor disk usage with `df -h`
   - Clean UV cache: `uv clean`
   - Use smaller data subsets for testing

3. **Import Errors**: If packages fail to import:
   - Ensure UV environment is activated
   - Run `uv sync` to reinstall dependencies
   - Check Python version compatibility

### Performance Tips

- Use `UV_LINK_MODE=copy` for better compatibility on shared filesystems
- Set reasonable timeouts for large data downloads
- Cache downloaded data to avoid repeated downloads

## Project Organization

```
├── LICENSE            <- MIT license
├── README.md          <- The top-level README for developers using this project
├── pyproject.toml     <- Project configuration file with package metadata and UV dependencies
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump (CELLxGENE Census downloads)
│
├── models             <- Trained and serialized models, model predictions, or model summaries
├── notebooks          <- Jupyter notebooks for exploration and analysis
├── scripts            <- Python scripts for data download and processing
│   ├── download_cellxgene_data.py    <- Main script to download Census data
│   └── simple_cellxgene_test.py      <- Test script for environment validation
│
├── results            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                <- Source code for use in this project
    ├── __init__.py             <- Makes src a Python module
    ├── paths_and_constants.py  <- Store paths and constants
    └── utils.py                <- Utility functions
```

## Getting Started with Analysis

1. **Test your setup**:
   ```bash
   uv run scripts/simple_cellxgene_test.py
   ```

2. **Download sample data**:
   ```bash
   uv run scripts/download_cellxgene_data.py
   ```

3. **Start Jupyter for analysis**:
   ```bash
   uv run jupyter lab notebooks/
   ```

4. **Check downloaded data**:
   ```bash
   ls -la data/raw/
   cat data/raw/download_summary.txt
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and test with UV: `uv run scripts/simple_cellxgene_test.py`
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
