#!/usr/bin/env python3
"""
SAE Feature Interpretation and Visualization Module

This module provides comprehensive analysis and visualization tools for understanding:
1. What biological attributions each SAE feature represents
2. How attributions leak/spread across multiple features
3. Feature specificity and selectivity scores
4. Hierarchical organization of features and attributions
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class SAEFeatureInterpreter:
    """
    Comprehensive analysis and visualization of SAE feature interpretability.
    """

    def __init__(self, results: Dict, sae_features: np.ndarray, metadata: pd.DataFrame):
        """
        Initialize the interpreter with analysis results.

        Args:
            results: Output from FeatureAttributionAnalyzer
            sae_features: SAE feature matrix (n_cells x n_features)
            metadata: Cell metadata DataFrame
        """
        self.results = results
        self.sae_features = sae_features
        self.metadata = metadata
        self.n_cells, self.n_features = sae_features.shape

        # Extract significance matrices
        self.significance_matrices = self._extract_significance_matrices()
        self.effect_size_matrices = self._extract_effect_size_matrices()

        logger.info(f"Initialized SAE interpreter for {self.n_features} features across {self.n_cells} cells")

    def _extract_significance_matrices(self) -> Dict[str, np.ndarray]:
        """Extract q-value matrices for each attribution type."""
        matrices = {}

        univariate = self.results.get('univariate', {})

        for attr_type in ['binary', 'categorical', 'continuous']:
            if attr_type not in univariate:
                continue

            attr_data = univariate[attr_type]
            attr_names = list(attr_data.keys())

            if not attr_names:
                continue

            # Create matrix: features x attributions
            matrix = np.ones((self.n_features, len(attr_names)))

            for j, attr_name in enumerate(attr_names):
                attr_results = attr_data[attr_name]
                for result in attr_results:
                    if 'feature_idx' in result and 'q_value' in result:
                        feature_idx = result['feature_idx']
                        q_value = result['q_value']
                        if 0 <= feature_idx < self.n_features:
                            matrix[feature_idx, j] = q_value

            matrices[attr_type] = {
                'matrix': matrix,
                'attr_names': attr_names
            }

        return matrices

    def _extract_effect_size_matrices(self) -> Dict[str, np.ndarray]:
        """Extract effect size matrices for each attribution type."""
        matrices = {}

        univariate = self.results.get('univariate', {})

        for attr_type in ['binary', 'categorical', 'continuous']:
            if attr_type not in univariate:
                continue

            attr_data = univariate[attr_type]
            attr_names = list(attr_data.keys())

            if not attr_names:
                continue

            # Create matrix: features x attributions
            matrix = np.zeros((self.n_features, len(attr_names)))

            for j, attr_name in enumerate(attr_names):
                attr_results = attr_data[attr_name]
                for result in attr_results:
                    if 'feature_idx' in result and 'effect_size' in result:
                        feature_idx = result['feature_idx']
                        effect_size = result['effect_size']
                        if 0 <= feature_idx < self.n_features:
                            matrix[feature_idx, j] = abs(effect_size)  # Use absolute effect size

            matrices[attr_type] = {
                'matrix': matrix,
                'attr_names': attr_names
            }

        return matrices

    def compute_feature_specificity_scores(self, significance_threshold: float = 0.05) -> pd.DataFrame:
        """
        Compute specificity scores for each SAE feature.

        Specificity measures how selective a feature is:
        - High specificity: feature is strongly associated with few attributions
        - Low specificity: feature is associated with many attributions (less specific)
        """
        specificity_data = []

        for attr_type, data in self.significance_matrices.items():
            matrix = data['matrix']
            attr_names = data['attr_names']

            # Binary significance matrix
            sig_matrix = matrix < significance_threshold

            for feature_idx in range(self.n_features):
                feature_associations = sig_matrix[feature_idx, :]
                n_significant = feature_associations.sum()

                if n_significant > 0:
                    # Specificity: 1 / number of significant associations
                    # Higher values = more specific (fewer associations)
                    specificity = 1.0 / n_significant

                    # Get strongest associations
                    feature_qvals = matrix[feature_idx, :]
                    sig_indices = np.where(feature_associations)[0]

                    if len(sig_indices) > 0:
                        strongest_idx = sig_indices[np.argmin(feature_qvals[sig_indices])]
                        strongest_attr = attr_names[strongest_idx]
                        strongest_qval = feature_qvals[strongest_idx]

                        specificity_data.append({
                            'feature_idx': feature_idx,
                            'attr_type': attr_type,
                            'n_associations': n_significant,
                            'specificity_score': specificity,
                            'strongest_association': strongest_attr,
                            'strongest_qvalue': strongest_qval
                        })

        # Aggregate across attribution types for each feature
        if specificity_data:
            df = pd.DataFrame(specificity_data)

            # Get overall specificity per feature (minimum across types = most general)
            feature_specificity = df.groupby('feature_idx').agg({
                'specificity_score': 'min',  # Most general (lowest specificity)
                'n_associations': 'sum',     # Total associations
                'strongest_qvalue': 'min'    # Best q-value across all types
            }).reset_index()

            # Add the strongest single association info
            strongest_info = df.loc[df.groupby('feature_idx')['strongest_qvalue'].idxmin()]
            feature_specificity = feature_specificity.merge(
                strongest_info[['feature_idx', 'strongest_association', 'attr_type']],
                on='feature_idx'
            )

            return feature_specificity
        else:
            return pd.DataFrame()

    def compute_attribution_leakage_scores(self, significance_threshold: float = 0.05) -> pd.DataFrame:
        """
        Compute leakage scores for each attribution.

        Leakage measures how spread out an attribution is across features:
        - High leakage: attribution affects many features (distributed representation)
        - Low leakage: attribution affects few features (localized representation)
        """
        leakage_data = []

        for attr_type, data in self.significance_matrices.items():
            matrix = data['matrix']
            attr_names = data['attr_names']

            # Binary significance matrix
            sig_matrix = matrix < significance_threshold

            for attr_idx, attr_name in enumerate(attr_names):
                attr_associations = sig_matrix[:, attr_idx]
                n_features_affected = attr_associations.sum()

                if n_features_affected > 0:
                    # Leakage: how many features this attribution affects
                    leakage_score = n_features_affected / self.n_features  # Normalize by total features

                    # Get effect size information if available
                    if attr_type in self.effect_size_matrices:
                        effect_matrix = self.effect_size_matrices[attr_type]['matrix']
                        attr_effects = effect_matrix[:, attr_idx]
                        mean_effect = attr_effects[attr_associations].mean() if attr_associations.sum() > 0 else 0
                        max_effect = attr_effects.max()
                    else:
                        mean_effect = 0
                        max_effect = 0

                    leakage_data.append({
                        'attribution': attr_name,
                        'attr_type': attr_type,
                        'n_features_affected': n_features_affected,
                        'leakage_score': leakage_score,
                        'mean_effect_size': mean_effect,
                        'max_effect_size': max_effect
                    })

        return pd.DataFrame(leakage_data)

    def plot_feature_specificity_distribution(self, output_dir: str = "results/sae_interpretation"):
        """Plot distribution of feature specificity scores."""
        specificity_df = self.compute_feature_specificity_scores()

        if specificity_df.empty:
            logger.warning("No specificity data to plot")
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Specificity score distribution
        axes[0, 0].hist(specificity_df['specificity_score'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Specificity Score')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Distribution of Feature Specificity Scores')
        axes[0, 0].axvline(specificity_df['specificity_score'].median(), color='red', linestyle='--',
                          label=f'Median: {specificity_df["specificity_score"].median():.3f}')
        axes[0, 0].legend()

        # Number of associations distribution
        axes[0, 1].hist(specificity_df['n_associations'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Number of Significant Associations')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Distribution of Feature Association Counts')

        # Specificity vs associations scatter
        axes[1, 0].scatter(specificity_df['n_associations'], specificity_df['specificity_score'], alpha=0.6)
        axes[1, 0].set_xlabel('Number of Associations')
        axes[1, 0].set_ylabel('Specificity Score')
        axes[1, 0].set_title('Specificity vs Number of Associations')

        # Top specific features
        top_specific = specificity_df.nlargest(10, 'specificity_score')
        y_pos = np.arange(len(top_specific))
        axes[1, 1].barh(y_pos, top_specific['specificity_score'])
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"F{idx}: {attr}" for idx, attr in
                                   zip(top_specific['feature_idx'], top_specific['strongest_association'])])
        axes[1, 1].set_xlabel('Specificity Score')
        axes[1, 1].set_title('Top 10 Most Specific Features')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_specificity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save specificity data
        specificity_df.to_csv(f"{output_dir}/feature_specificity_scores.csv", index=False)
        logger.info(f"Feature specificity analysis saved to {output_dir}")

    def plot_attribution_leakage_analysis(self, output_dir: str = "results/sae_interpretation"):
        """Plot attribution leakage analysis."""
        leakage_df = self.compute_attribution_leakage_scores()

        if leakage_df.empty:
            logger.warning("No leakage data to plot")
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Leakage score distribution
        axes[0, 0].hist(leakage_df['leakage_score'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Leakage Score (Fraction of Features Affected)')
        axes[0, 0].set_ylabel('Number of Attributions')
        axes[0, 0].set_title('Distribution of Attribution Leakage Scores')
        axes[0, 0].axvline(leakage_df['leakage_score'].median(), color='red', linestyle='--',
                          label=f'Median: {leakage_df["leakage_score"].median():.3f}')
        axes[0, 0].legend()

        # Leakage by attribution type
        sns.boxplot(data=leakage_df, x='attr_type', y='leakage_score', ax=axes[0, 1])
        axes[0, 1].set_title('Leakage Score by Attribution Type')
        axes[0, 1].set_ylabel('Leakage Score')

        # Effect size vs leakage
        axes[1, 0].scatter(leakage_df['mean_effect_size'], leakage_df['leakage_score'],
                          c=leakage_df['attr_type'].astype('category').cat.codes, alpha=0.6)
        axes[1, 0].set_xlabel('Mean Effect Size')
        axes[1, 0].set_ylabel('Leakage Score')
        axes[1, 0].set_title('Effect Size vs Leakage')

        # Top leaky attributions
        top_leaky = leakage_df.nlargest(15, 'leakage_score')
        y_pos = np.arange(len(top_leaky))
        axes[1, 1].barh(y_pos, top_leaky['leakage_score'])
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"{attr} ({n})" for attr, n in
                                   zip(top_leaky['attribution'], top_leaky['n_features_affected'])])
        axes[1, 1].set_xlabel('Leakage Score')
        axes[1, 1].set_title('Top 15 Most Distributed Attributions')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/attribution_leakage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save leakage data
        leakage_df.to_csv(f"{output_dir}/attribution_leakage_scores.csv", index=False)
        logger.info(f"Attribution leakage analysis saved to {output_dir}")

    def create_feature_attribution_heatmap(self, top_features: int = 50, top_attributions: int = 30,
                                          output_dir: str = "results/sae_interpretation"):
        """Create comprehensive heatmap of feature-attribution associations."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Combine all significance matrices
        all_matrices = []
        all_attr_names = []
        all_attr_types = []

        for attr_type, data in self.significance_matrices.items():
            matrix = data['matrix']
            attr_names = data['attr_names']

            all_matrices.append(matrix)
            all_attr_names.extend(attr_names)
            all_attr_types.extend([attr_type] * len(attr_names))

        if not all_matrices:
            logger.warning("No significance matrices to plot")
            return

        # Combine matrices
        combined_matrix = np.concatenate(all_matrices, axis=1)

        # Convert to -log10(q-value) for visualization
        log_matrix = -np.log10(np.maximum(combined_matrix, 1e-10))

        # Select top features and attributions
        feature_importance = np.mean(log_matrix, axis=1)
        top_feature_indices = np.argsort(feature_importance)[-top_features:]

        attr_importance = np.mean(log_matrix, axis=0)
        top_attr_indices = np.argsort(attr_importance)[-top_attributions:]

        # Create subset matrix
        subset_matrix = log_matrix[np.ix_(top_feature_indices, top_attr_indices)]
        subset_features = [f"Feature_{i}" for i in top_feature_indices]
        subset_attrs = [all_attr_names[i] for i in top_attr_indices]
        subset_types = [all_attr_types[i] for i in top_attr_indices]

        # Create heatmap
        plt.figure(figsize=(20, 12))

        # Create color map for attribution types
        type_colors = {'binary': 'red', 'categorical': 'blue', 'continuous': 'green'}
        col_colors = [type_colors.get(t, 'gray') for t in subset_types]

        # Create clustermap
        g = sns.clustermap(subset_matrix,
                          xticklabels=subset_attrs,
                          yticklabels=subset_features,
                          cmap='viridis',
                          cbar_kws={'label': '-log10(q-value)'},
                          col_colors=col_colors,
                          figsize=(20, 12))

        # Rotate labels
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

        plt.savefig(f"{output_dir}/feature_attribution_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature-attribution heatmap saved to {output_dir}")

    def create_interactive_feature_explorer(self, output_dir: str = "results/sae_interpretation"):
        """Create interactive Plotly dashboard for feature exploration."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Compute feature embeddings using PCA and t-SNE
        feature_activity = self.sae_features.T  # Features as rows

        # PCA
        pca = PCA(n_components=2)
        feature_pca = pca.fit_transform(StandardScaler().fit_transform(feature_activity))

        # Get feature specificity and leakage info
        specificity_df = self.compute_feature_specificity_scores()

        # Create feature info dataframe
        feature_info = pd.DataFrame({
            'feature_idx': range(self.n_features),
            'pca_x': feature_pca[:, 0],
            'pca_y': feature_pca[:, 1],
            'sparsity': (self.sae_features == 0).mean(axis=0),
            'activation_mean': self.sae_features.mean(axis=0),
            'activation_std': self.sae_features.std(axis=0)
        })

        # Merge with specificity info
        if not specificity_df.empty:
            feature_info = feature_info.merge(specificity_df, on='feature_idx', how='left')
            feature_info['specificity_score'] = feature_info['specificity_score'].fillna(0)
            feature_info['strongest_association'] = feature_info['strongest_association'].fillna('None')
        else:
            feature_info['specificity_score'] = 0
            feature_info['strongest_association'] = 'None'

        # Create interactive plot
        fig = px.scatter(feature_info,
                        x='pca_x', y='pca_y',
                        color='specificity_score',
                        size='activation_mean',
                        hover_data=['feature_idx', 'sparsity', 'strongest_association'],
                        title='SAE Feature Landscape (PCA Projection)',
                        labels={'pca_x': 'PC1', 'pca_y': 'PC2'})

        fig.write_html(f"{output_dir}/interactive_feature_explorer.html")

        # Save feature info
        feature_info.to_csv(f"{output_dir}/feature_landscape_data.csv", index=False)

        logger.info(f"Interactive feature explorer saved to {output_dir}")

    def generate_feature_interpretation_report(self, output_dir: str = "results/sae_interpretation"):
        """Generate comprehensive interpretation report."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Compute all metrics
        specificity_df = self.compute_feature_specificity_scores()
        leakage_df = self.compute_attribution_leakage_scores()

        # Generate report
        report = []
        report.append("# SAE Feature Interpretation Report\n")

        # Overall statistics
        report.append("## Overall Statistics")
        report.append(f"- Total SAE Features: {self.n_features}")
        report.append(f"- Cells Analyzed: {self.n_cells}")
        report.append(f"- Overall Sparsity: {(self.sae_features == 0).mean():.2%}")

        if not specificity_df.empty:
            report.append(f"- Features with Significant Associations: {len(specificity_df)}")
            report.append(f"- Median Feature Specificity: {specificity_df['specificity_score'].median():.3f}")
            report.append(f"- Mean Associations per Feature: {specificity_df['n_associations'].mean():.1f}")

        if not leakage_df.empty:
            report.append(f"- Median Attribution Leakage: {leakage_df['leakage_score'].median():.3f}")
            report.append(f"- Mean Features per Attribution: {leakage_df['n_features_affected'].mean():.1f}")

        # Top specific features
        if not specificity_df.empty:
            report.append("\n## Top 10 Most Specific Features")
            top_specific = specificity_df.nlargest(10, 'specificity_score')
            for _, row in top_specific.iterrows():
                report.append(f"- **Feature {row['feature_idx']}**: {row['strongest_association']} "
                             f"(Specificity: {row['specificity_score']:.3f}, "
                             f"Q-value: {row['strongest_qvalue']:.2e})")

        # Top distributed attributions
        if not leakage_df.empty:
            report.append("\n## Top 10 Most Distributed Attributions")
            top_leaky = leakage_df.nlargest(10, 'leakage_score')
            for _, row in top_leaky.iterrows():
                report.append(f"- **{row['attribution']}** ({row['attr_type']}): "
                             f"{row['n_features_affected']} features "
                             f"({row['leakage_score']:.1%} of total)")

        # Attribution type analysis
        if not leakage_df.empty:
            report.append("\n## Attribution Type Analysis")
            type_stats = leakage_df.groupby('attr_type').agg({
                'leakage_score': ['mean', 'std'],
                'n_features_affected': 'mean'
            }).round(3)

            for attr_type in type_stats.index:
                mean_leakage = type_stats.loc[attr_type, ('leakage_score', 'mean')]
                std_leakage = type_stats.loc[attr_type, ('leakage_score', 'std')]
                mean_features = type_stats.loc[attr_type, ('n_features_affected', 'mean')]

                report.append(f"- **{attr_type.title()}**: "
                             f"Mean leakage = {mean_leakage:.3f} ± {std_leakage:.3f}, "
                             f"Mean features affected = {mean_features:.1f}")

        # Save report
        with open(f"{output_dir}/interpretation_report.md", 'w') as f:
            f.write('\n'.join(report))

        logger.info(f"Interpretation report saved to {output_dir}")

    def run_full_analysis(self, output_dir: str = "results/sae_interpretation"):
        """Run complete interpretability analysis."""
        logger.info("Running comprehensive SAE feature interpretation analysis...")

        # Create all visualizations
        self.plot_feature_specificity_distribution(output_dir)
        self.plot_attribution_leakage_analysis(output_dir)
        self.create_feature_attribution_heatmap(output_dir=output_dir)
        self.create_interactive_feature_explorer(output_dir)
        self.generate_feature_interpretation_report(output_dir)

        logger.info(f"Complete SAE interpretation analysis saved to {output_dir}")
