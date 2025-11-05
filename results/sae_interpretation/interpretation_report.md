# SAE Feature Interpretation Report

## Overall Statistics
- Total SAE Features: 5120
- Cells Analyzed: 100
- Overall Sparsity: 43.61%
- Features with Significant Associations: 4613
- Median Feature Specificity: 1.000
- Mean Associations per Feature: 3.3
- Median Attribution Leakage: 0.165
- Mean Features per Attribution: 1161.6

## Top 10 Most Specific Features
- **Feature 0**: cell_type (Specificity: 1.000, Q-value: 6.32e-04)
- **Feature 2**: cell_type (Specificity: 1.000, Q-value: 4.84e-04)
- **Feature 3**: cell_type (Specificity: 1.000, Q-value: 3.29e-04)
- **Feature 4**: Inflammation_Response (Specificity: 1.000, Q-value: 1.56e-02)
- **Feature 6**: cell_type (Specificity: 1.000, Q-value: 2.18e-07)
- **Feature 8**: cell_type (Specificity: 1.000, Q-value: 2.22e-07)
- **Feature 9**: cell_type (Specificity: 1.000, Q-value: 1.48e-03)
- **Feature 10**: cell_type (Specificity: 1.000, Q-value: 8.92e-05)
- **Feature 11**: cell_type (Specificity: 1.000, Q-value: 1.24e-07)
- **Feature 14**: cell_type (Specificity: 1.000, Q-value: 4.14e-03)

## Top 10 Most Distributed Attributions
- **cell_type** (categorical): 4308 features (84.1% of total)
- **Inflammation_Response** (continuous): 1835 features (35.8% of total)
- **pct_ribo** (continuous): 1682 features (32.9% of total)
- **Hypoxia** (continuous): 1018 features (19.9% of total)
- **Oxidative_Phosphorylation** (continuous): 944 features (18.4% of total)
- **DNA_Repair** (continuous): 873 features (17.1% of total)
- **n_genes** (continuous): 843 features (16.5% of total)
- **Apoptosis** (continuous): 796 features (15.5% of total)
- **pct_mito** (continuous): 750 features (14.6% of total)
- **S_score** (continuous): 726 features (14.2% of total)

## Attribution Type Analysis
- **Categorical**: Mean leakage = 0.434 ± 0.576, Mean features affected = 2223.5
- **Continuous**: Mean leakage = 0.189 ± 0.081, Mean features affected = 968.5