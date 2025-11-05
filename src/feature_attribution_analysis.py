"""
Feature-Attribution Correlation Analysis Module

This module implements a comprehensive statistical toolkit to quantify how well
embedding/SAE features correlate with cell-level attributes (cell type, disease, etc.).

Classes:
    FeatureAttributionAnalyzer: Main analysis class for feature-attribution correlations

Functions:
    analyze_associations: Simplified interface for running the complete analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)


class FeatureAttributionAnalyzer:
    """
    Comprehensive analyzer for feature-attribution correlations.

    This class implements statistical methods to assess how well embedding features
    correlate with cell-level attributes, handling different data types and
    distribution mismatches.

    Args:
        n_permutations: Number of permutations for permutation tests
        fdr_method: Method for FDR correction ('fdr_bh', 'fdr_by', 'bonferroni')
        cv_folds: Number of cross-validation folds for linear probes
        random_state: Random seed for reproducibility
    """

    def __init__(self,
                 n_permutations: int = 1000,
                 fdr_method: str = 'fdr_bh',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """Initialize the analyzer with configuration parameters."""
        self.n_permutations = n_permutations
        self.fdr_method = fdr_method
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Set random seed
        np.random.seed(random_state)

        logger.info(f"Initialized FeatureAttributionAnalyzer with {n_permutations} permutations, "
                   f"FDR method: {fdr_method}, CV folds: {cv_folds}")

    def prepare_data(self,
                    features: np.ndarray,
                    attributions: Dict[str, Dict[str, Union[np.ndarray, pd.Series]]],
                    confounders: Optional[np.ndarray] = None) -> Dict:
        """
        Prepare and align data matrices for analysis.

        Args:
            features: Feature matrix (n_cells × n_features)
            attributions: Dictionary with 'binary', 'categorical', 'continuous' keys
            confounders: Optional confounder matrix (n_cells × n_confounders)

        Returns:
            Dictionary with prepared data matrices
        """
        logger.info("Preparing data matrices...")

        # Ensure features is a numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        # Process attributions
        processed_attrs = {
            'binary': {},
            'categorical': {},
            'continuous': {}
        }

        # Binary attributes
        if 'binary' in attributions:
            for attr_name, attr_values in attributions['binary'].items():
                if isinstance(attr_values, pd.Series):
                    attr_values = attr_values.values

                # Convert to binary (0/1) if not already
                unique_vals = np.unique(attr_values[~pd.isna(attr_values)])
                if len(unique_vals) == 2:
                    processed_attrs['binary'][attr_name] = (attr_values == unique_vals[1]).astype(int)
                else:
                    logger.warning(f"Binary attribute {attr_name} has {len(unique_vals)} unique values")

        # Categorical attributes
        if 'categorical' in attributions:
            for attr_name, attr_values in attributions['categorical'].items():
                if isinstance(attr_values, pd.Series):
                    attr_values = attr_values.values

                # Label encode categorical variables
                le = LabelEncoder()
                try:
                    # Handle missing values
                    mask = ~pd.isna(attr_values)
                    encoded = np.full(len(attr_values), -1, dtype=int)
                    encoded[mask] = le.fit_transform(attr_values[mask].astype(str))
                    processed_attrs['categorical'][attr_name] = encoded
                except ValueError as e:
                    logger.warning(f"Could not encode categorical attribute {attr_name}: {e}")

        # Continuous attributes
        if 'continuous' in attributions:
            for attr_name, attr_values in attributions['continuous'].items():
                if isinstance(attr_values, pd.Series):
                    attr_values = attr_values.values

                # Ensure numeric
                try:
                    processed_attrs['continuous'][attr_name] = attr_values.astype(float)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert continuous attribute {attr_name} to float")

        data = {
            'features': features,
            'attributions': processed_attrs,
            'confounders': confounders
        }

        logger.info(f"Prepared data: {features.shape[1]} features, "
                   f"{len(processed_attrs['binary'])} binary attrs, "
                   f"{len(processed_attrs['categorical'])} categorical attrs, "
                   f"{len(processed_attrs['continuous'])} continuous attrs")

        return data

    def residualize_features(self, features: np.ndarray, confounders: np.ndarray) -> np.ndarray:
        """
        Remove confounding effects from features using OLS regression.

        Args:
            features: Feature matrix (n_cells × n_features)
            confounders: Confounder matrix (n_cells × n_confounders)

        Returns:
            Residualized feature matrix
        """
        if confounders is None or confounders.size == 0:
            return features

        features_resid = np.zeros_like(features)

        for j in range(features.shape[1]):
            try:
                # Add intercept to confounders
                c_with_intercept = np.column_stack([np.ones(confounders.shape[0]), confounders])

                # Fit OLS regression: feature_j ~ 1 + confounders
                model = OLS(features[:, j], c_with_intercept).fit()
                features_resid[:, j] = model.resid
            except (np.linalg.LinAlgError, ValueError):
                # If regression fails, use original feature values
                features_resid[:, j] = features[:, j]

        return features_resid

    def point_biserial_correlation(self, continuous: np.ndarray, binary: np.ndarray) -> Tuple[float, float]:
        """Calculate point-biserial correlation between continuous and binary variables."""
        try:
            # Remove missing values
            mask = ~(pd.isna(continuous) | pd.isna(binary))
            if np.sum(mask) < 3:  # Need at least 3 valid observations
                return 0.0, 1.0

            r, p = pearsonr(continuous[mask], binary[mask])
            return r if not np.isnan(r) else 0.0, p if not np.isnan(p) else 1.0
        except (ValueError, ZeroDivisionError):
            return 0.0, 1.0

    def cliff_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cliff's delta effect size for binary comparisons.

        Cliff's delta = (# pairs where x1 > x2 - # pairs where x1 < x2) / (n1 * n2)

        Interpretation:
        - |δ| < 0.147: negligible
        - 0.147 ≤ |δ| < 0.33: small
        - 0.33 ≤ |δ| < 0.474: moderate
        - |δ| ≥ 0.474: large
        """
        # Remove missing values
        group1 = group1[~pd.isna(group1)]
        group2 = group2[~pd.isna(group2)]

        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0

        try:
            dominance = sum(np.sum(g1 > group2) - np.sum(g1 < group2) for g1 in group1)
            return dominance / (n1 * n2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def eta_squared(self, groups: List[np.ndarray]) -> float:
        """
        Calculate eta-squared effect size for categorical comparisons.

        η² = SS_between / SS_total

        Interpretation:
        - η² < 0.01: negligible
        - 0.01 ≤ η² < 0.06: small
        - 0.06 ≤ η² < 0.14: moderate
        - η² ≥ 0.14: large
        """
        try:
            # Remove missing values from each group
            groups = [g[~pd.isna(g)] for g in groups]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) < 2:
                return 0.0

            all_values = np.concatenate(groups)
            grand_mean = np.mean(all_values)

            ss_total = np.sum((all_values - grand_mean) ** 2)
            ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)

            return ss_between / ss_total if ss_total > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    def distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate distance correlation between two variables.

        Distance correlation captures both linear and non-linear associations.
        """
        try:
            # Remove missing values
            mask = ~(pd.isna(x) | pd.isna(y))
            if np.sum(mask) < 3:
                return 0.0

            x_clean = x[mask]
            y_clean = y[mask]

            def center_distance_matrix(d):
                row_means = d.mean(axis=1, keepdims=True)
                col_means = d.mean(axis=0, keepdims=True)
                grand_mean = d.mean()
                return d - row_means - col_means + grand_mean

            def distance_matrix(data):
                return squareform(pdist(data.reshape(-1, 1)))

            dx = distance_matrix(x_clean)
            dy = distance_matrix(y_clean)

            # Center distance matrices
            dx_centered = center_distance_matrix(dx)
            dy_centered = center_distance_matrix(dy)

            # Calculate distance covariance and variances
            dcov_xy = np.mean(dx_centered * dy_centered)
            dvar_x = np.mean(dx_centered * dx_centered)
            dvar_y = np.mean(dy_centered * dy_centered)

            # Calculate distance correlation
            if dvar_x > 0 and dvar_y > 0:
                dcorr = dcov_xy / np.sqrt(dvar_x * dvar_y)
                return max(0, dcorr)  # Ensure non-negative
            else:
                return 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    def univariate_associations(self, data: Dict) -> Dict:
        """
        Calculate univariate associations between features and attributes.

        Returns:
            Dictionary with association results for each attribute type
        """
        logger.info("Computing univariate associations...")

        features = data['features']
        attributions = data['attributions']
        confounders = data['confounders']

        # Residualize features against confounders
        features_resid = self.residualize_features(features, confounders)

        # Standardize residualized features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_resid)

        results = {
            'binary': {},
            'categorical': {},
            'continuous': {}
        }

        # Binary attributes: Point-biserial correlation + Mann-Whitney U
        for attr_name, attr_values in attributions['binary'].items():
            logger.info(f"Processing binary attribute: {attr_name}")

            attr_results = []

            # Add progress bar for feature iteration
            feature_iterator = tqdm(range(features_scaled.shape[1]),
                                  desc=f"Binary {attr_name}",
                                  leave=False)

            for j in feature_iterator:
                feature_vals = features_scaled[:, j]

                # Point-biserial correlation
                r, p_pb = self.point_biserial_correlation(feature_vals, attr_values)

                # Mann-Whitney U test
                group_0 = feature_vals[attr_values == 0]
                group_1 = feature_vals[attr_values == 1]

                if len(group_0) > 0 and len(group_1) > 0:
                    try:
                        statistic, p_mw = mannwhitneyu(group_1, group_0, alternative='two-sided')
                        # Cliff's delta
                        delta = self.cliff_delta(group_1, group_0)
                    except ValueError:
                        statistic, p_mw, delta = 0, 1.0, 0.0
                else:
                    statistic, p_mw, delta = 0, 1.0, 0.0

                attr_results.append({
                    'feature_idx': j,
                    'point_biserial_r': r,
                    'point_biserial_p': p_pb,
                    'mann_whitney_stat': statistic,
                    'mann_whitney_p': p_mw,
                    'cliff_delta': delta,
                    'effect_size': abs(delta),
                    'statistic': statistic,
                    'p_value': p_mw
                })

            results['binary'][attr_name] = attr_results

        # Categorical attributes: Kruskal-Wallis test
        for attr_name, attr_values in attributions['categorical'].items():
            logger.info(f"Processing categorical attribute: {attr_name}")

            unique_vals = np.unique(attr_values)
            unique_vals = unique_vals[unique_vals >= 0]  # Remove missing value marker (-1)

            if len(unique_vals) < 2:
                continue

            attr_results = []

            # Add progress bar for feature iteration
            feature_iterator = tqdm(range(features_scaled.shape[1]),
                                  desc=f"Categorical {attr_name}",
                                  leave=False)

            for j in feature_iterator:
                feature_vals = features_scaled[:, j]

                # Group feature values by category
                groups = [feature_vals[attr_values == val] for val in unique_vals]
                groups = [g for g in groups if len(g) > 0]  # Remove empty groups

                if len(groups) >= 2:
                    try:
                        # Kruskal-Wallis test (non-parametric ANOVA)
                        statistic, p_kw = kruskal(*groups)
                        # Eta-squared effect size
                        eta2 = self.eta_squared(groups)
                    except ValueError:
                        statistic, p_kw, eta2 = 0, 1.0, 0.0
                else:
                    statistic, p_kw, eta2 = 0, 1.0, 0.0

                attr_results.append({
                    'feature_idx': j,
                    'kruskal_wallis_stat': statistic,
                    'kruskal_wallis_p': p_kw,
                    'eta_squared': eta2,
                    'effect_size': eta2,
                    'statistic': statistic,
                    'p_value': p_kw
                })

            results['categorical'][attr_name] = attr_results

        # Continuous attributes: Spearman correlation + distance correlation
        for attr_name, attr_values in attributions['continuous'].items():
            logger.info(f"Processing continuous attribute: {attr_name}")

            attr_results = []

            # Add progress bar for feature iteration
            feature_iterator = tqdm(range(features_scaled.shape[1]),
                                  desc=f"Continuous {attr_name}",
                                  leave=False)

            for j in feature_iterator:
                feature_vals = features_scaled[:, j]

                # Spearman correlation (robust to non-normality)
                try:
                    r_s, p_s = spearmanr(feature_vals, attr_values, nan_policy='omit')
                    r_s = r_s if not np.isnan(r_s) else 0.0
                    p_s = p_s if not np.isnan(p_s) else 1.0
                except ValueError:
                    r_s, p_s = 0.0, 1.0

                # Distance correlation
                dcorr = self.distance_correlation(feature_vals, attr_values)

                attr_results.append({
                    'feature_idx': j,
                    'spearman_r': r_s,
                    'spearman_p': p_s,
                    'distance_correlation': dcorr,
                    'effect_size': abs(r_s),
                    'statistic': abs(r_s),
                    'p_value': p_s
                })

            results['continuous'][attr_name] = attr_results

        logger.info("Completed univariate associations")
        return results

    def linear_probes(self, data: Dict) -> Dict:
        """
        Train linear probes to predict attributes from all features jointly.

        This provides multivariate analysis complementing univariate associations.
        """
        logger.info("Training linear probes...")

        features = data['features']
        attributions = data['attributions']
        confounders = data['confounders']

        # Residualize and scale features
        features_resid = self.residualize_features(features, confounders)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_resid)

        results = {}

        # Binary classification probes
        logger.info("Training binary classification probes...")
        for attr_name, attr_values in tqdm(attributions['binary'].items(),
                                          desc="Binary probes",
                                          leave=False):
            if len(np.unique(attr_values)) == 2:  # Ensure binary
                try:
                    clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    scores = cross_val_score(clf, features_scaled, attr_values, cv=cv, scoring='roc_auc')

                    results[attr_name] = {
                        'auc': np.mean(scores),
                        'auc_std': np.std(scores),
                        'scores': scores.tolist()
                    }
                except (ValueError, np.linalg.LinAlgError):
                    results[attr_name] = {
                        'auc': 0.5,
                        'auc_std': 0.0,
                        'scores': [0.5] * self.cv_folds
                    }

        # Multi-class classification probes
        logger.info("Training multi-class classification probes...")
        for attr_name, attr_values in tqdm(attributions['categorical'].items(),
                                          desc="Categorical probes",
                                          leave=False):
            unique_vals = np.unique(attr_values)
            unique_vals = unique_vals[unique_vals >= 0]  # Remove missing value marker

            if len(unique_vals) > 2:  # Multi-class
                try:
                    clf = LogisticRegression(random_state=self.random_state, max_iter=1000, multi_class='ovr')
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    scores = cross_val_score(clf, features_scaled, attr_values, cv=cv, scoring='accuracy')

                    results[attr_name] = {
                        'accuracy': np.mean(scores),
                        'accuracy_std': np.std(scores),
                        'scores': scores.tolist()
                    }
                except (ValueError, np.linalg.LinAlgError):
                    results[attr_name] = {
                        'accuracy': 1.0 / len(unique_vals),  # Random chance
                        'accuracy_std': 0.0,
                        'scores': [1.0 / len(unique_vals)] * self.cv_folds
                    }

        # Regression probes
        logger.info("Training regression probes...")
        for attr_name, attr_values in tqdm(attributions['continuous'].items(),
                                          desc="Regression probes",
                                          leave=False):
            try:
                reg = Ridge(random_state=self.random_state)
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(reg, features_scaled, attr_values, cv=cv, scoring='r2')

                results[attr_name] = {
                    'r2': np.mean(scores),
                    'r2_std': np.std(scores),
                    'scores': scores.tolist()
                }
            except (ValueError, np.linalg.LinAlgError):
                results[attr_name] = {
                    'r2': 0.0,
                    'r2_std': 0.0,
                    'scores': [0.0] * self.cv_folds
                }

        logger.info("Completed linear probes")
        return results

    def multiple_testing_correction(self, univariate_results: Dict, alpha: float = 0.05) -> Dict:
        """
        Apply multiple testing correction to p-values.

        Args:
            univariate_results: Results from univariate_associations
            alpha: Significance level for FDR control

        Returns:
            Results with corrected p-values and significance flags
        """
        logger.info(f"Applying {self.fdr_method} correction at α = {alpha}")

        # Collect all p-values across all tests
        all_p_values = []
        p_value_locations = []

        for attr_type in ['binary', 'categorical', 'continuous']:
            if attr_type in univariate_results:
                for attr_name, attr_results in univariate_results[attr_type].items():
                    for i, result in enumerate(attr_results):
                        p_val = result.get('p_value', 1.0)
                        all_p_values.append(p_val)
                        p_value_locations.append((attr_type, attr_name, i))

        # Apply correction
        if len(all_p_values) > 0:
            rejected, p_corrected, _, _ = multipletests(
                all_p_values, alpha=alpha, method=self.fdr_method
            )

            # Map corrected p-values back to results
            for idx, (attr_type, attr_name, result_idx) in enumerate(p_value_locations):
                univariate_results[attr_type][attr_name][result_idx]['q_value'] = p_corrected[idx]
                univariate_results[attr_type][attr_name][result_idx]['significant'] = rejected[idx]

        logger.info("Completed multiple testing correction")
        return univariate_results

    def analyze_associations(self,
                           features: np.ndarray,
                           attributions: Dict[str, Dict[str, Union[np.ndarray, pd.Series]]],
                           confounders: Optional[np.ndarray] = None) -> Dict:
        """
        Run the complete feature-attribution association analysis.

        Args:
            features: Feature matrix (n_cells × n_features)
            attributions: Dictionary with 'binary', 'categorical', 'continuous' keys
            confounders: Optional confounder matrix (n_cells × n_confounders)

        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive feature-attribution analysis...")

        # Calculate total number of attributes for progress tracking
        n_binary = len(attributions.get('binary', {}))
        n_categorical = len(attributions.get('categorical', {}))
        n_continuous = len(attributions.get('continuous', {}))
        total_attrs = n_binary + n_categorical + n_continuous

        logger.info(f"Analysis overview: {features.shape[0]} cells, {features.shape[1]} features, "
                   f"{total_attrs} attributes ({n_binary} binary, {n_categorical} categorical, "
                   f"{n_continuous} continuous)")

        # Step 1: Prepare data
        logger.info("Step 1/5: Preparing data...")
        data = self.prepare_data(features, attributions, confounders)

        # Step 2: Univariate associations
        logger.info("Step 2/5: Computing univariate associations...")
        univariate_results = self.univariate_associations(data)

        # Step 3: Multiple testing correction
        logger.info("Step 3/5: Applying multiple testing correction...")
        corrected_results = self.multiple_testing_correction(univariate_results)

        # Step 4: Linear probes (multivariate analysis)
        logger.info("Step 4/5: Training linear probes...")
        probe_results = self.linear_probes(data)

        # Step 5: Compile final results
        logger.info("Step 5/5: Compiling final results...")
        final_results = {
            'data_info': {
                'n_cells': features.shape[0],
                'n_features': features.shape[1],
                'n_binary_attrs': len(attributions.get('binary', {})),
                'n_categorical_attrs': len(attributions.get('categorical', {})),
                'n_continuous_attrs': len(attributions.get('continuous', {})),
                'has_confounders': confounders is not None
            },
            'univariate': corrected_results,
            'multivariate': {
                'linear_probes': probe_results
            }
        }

        logger.info("Analysis completed successfully!")
        return final_results


def analyze_associations(features: np.ndarray,
                        attributions: Dict[str, Dict[str, Union[np.ndarray, pd.Series]]],
                        confounders: Optional[np.ndarray] = None,
                        n_permutations: int = 1000,
                        fdr_method: str = 'fdr_bh',
                        cv_folds: int = 5,
                        random_state: int = 42) -> Dict:
    """
    Simplified interface for running feature-attribution correlation analysis.

    Args:
        features: Feature matrix (n_cells × n_features)
        attributions: Dictionary with 'binary', 'categorical', 'continuous' keys
        confounders: Optional confounder matrix (n_cells × n_confounders)
        n_permutations: Number of permutations for permutation tests
        fdr_method: Method for FDR correction
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing all analysis results
    """
    analyzer = FeatureAttributionAnalyzer(
        n_permutations=n_permutations,
        fdr_method=fdr_method,
        cv_folds=cv_folds,
        random_state=random_state
    )

    return analyzer.analyze_associations(features, attributions, confounders)
