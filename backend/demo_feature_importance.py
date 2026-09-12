"""
Demonstration script for Feature Importance Analysis

This script demonstrates the complete functionality of the Feature Importance
implementation, showing all methods, visualizations, and use cases.
"""

import warnings
warnings.filterwarnings('ignore')

import json
from algorithms.ml.feature_importance import (
    FeatureImportanceModel,
    FeatureImportanceRequest,
    FeatureImportanceResponse
)

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS DEMONSTRATION")
print("=" * 80)

# Demo 1: Tree-based importance for quick analysis
print("\n" + "=" * 80)
print("DEMO 1: Quick Analysis with Tree-based Importance")
print("=" * 80)
print("\nUse Case: Fast feature ranking for initial data exploration")
print("Dataset: California Housing (20,640 samples, 8 features)")

request = FeatureImportanceRequest(
    method='tree',
    model_type='random_forest',
    dataset='housing',
    n_estimators=100,
    top_k=10,
    random_state=42
)

model = FeatureImportanceModel()
response = model.train(request)

print(f"\n✓ Analysis completed in {response.execution_time_ms:.2f}ms")
print(f"\n Model Performance:")
print(f"  - R² Score: {response.metrics['r2_score']:.4f}")
print(f"  - RMSE: {response.metrics['rmse']:.4f}")
print(f"  - MAE: {response.metrics['mae']:.4f}")

print(f"\n Top 5 Most Important Features (Tree-based):")
for rank_data in response.feature_rankings['tree'][:5]:
    print(f"  {rank_data['rank']}. {rank_data['feature']:15s} - {rank_data['importance']:.4f} ({rank_data['percentage']:.1f}%)")

print(f"\n Feature Statistics:")
stats = response.statistics['tree']
print(f"  - Mean importance: {stats['mean']:.4f}")
print(f"  - Std deviation: {stats['std']:.4f}")
print(f"  - Max importance: {stats['max']:.4f}")
print(f"  - Non-zero features: {stats['non_zero_features']}/{stats['total_features']}")

# Demo 2: Comprehensive analysis with all methods
print("\n" + "=" * 80)
print("DEMO 2: Comprehensive Analysis with Multiple Methods")
print("=" * 80)
print("\nUse Case: Robust feature selection comparing tree-based and permutation methods")
print("Dataset: Diabetes (442 samples, 10 features)")

request = FeatureImportanceRequest(
    method='all',
    model_type='gradient_boosting',
    dataset='diabetes',
    n_estimators=100,
    top_k=10,
    random_state=42
)

model = FeatureImportanceModel()
response = model.train(request)

print(f"\n✓ Analysis completed in {response.execution_time_ms:.2f}ms")
print(f"\n Methods Analyzed: {list(response.feature_importance.keys())}")
print(f"\n Model Performance:")
print(f"  - R² Score: {response.metrics['r2_score']:.4f}")
print(f"  - RMSE: {response.metrics['rmse']:.4f}")

print(f"\n Top 5 Features by Each Method:")
print(f"\n  TREE-BASED (MDI) - Fast, built-in to model:")
for rank_data in response.feature_rankings['tree'][:5]:
    print(f"    {rank_data['rank']}. {rank_data['feature']:10s} - {rank_data['importance']:.4f}")

print(f"\n  PERMUTATION - Model-agnostic, more reliable:")
for rank_data in response.feature_rankings['permutation'][:5]:
    print(f"    {rank_data['rank']}. {rank_data['feature']:10s} - {rank_data['importance']:.4f}")

print(f"\n Feature Comparison (Top 3):")
for i in range(3):
    tree_feat = response.feature_rankings['tree'][i]
    perm_feat = response.feature_rankings['permutation'][i]
    agreement = "✓" if tree_feat['feature'] == perm_feat['feature'] else "✗"
    print(f"  Rank {i+1}: Tree={tree_feat['feature']:10s} | Permutation={perm_feat['feature']:10s} {agreement}")

# Demo 3: Classification task
print("\n" + "=" * 80)
print("DEMO 3: Classification Task - Wine Dataset")
print("=" * 80)
print("\nUse Case: Feature importance for multi-class classification")
print("Dataset: Wine (178 samples, 13 features, 3 classes)")

request = FeatureImportanceRequest(
    method='all',
    model_type='random_forest',
    dataset='wine',
    n_estimators=100,
    top_k=10,
    random_state=42
)

model = FeatureImportanceModel()
response = model.train(request)

print(f"\n✓ Analysis completed in {response.execution_time_ms:.2f}ms")
print(f"\n Model Performance:")
print(f"  - Accuracy: {response.metrics['accuracy']:.4f}")
print(f"  - F1 Score: {response.metrics['f1_score']:.4f}")

print(f"\n Top 7 Features (Tree-based):")
for rank_data in response.feature_rankings['tree'][:7]:
    print(f"  {rank_data['rank']:2d}. {rank_data['feature']:35s} - {rank_data['importance']:.4f}")

# Demonstrate cumulative importance
print(f"\n Cumulative Feature Importance:")
cumulative_data = [d for d in response.cumulative_importance if d['method'] == 'tree']
for i, threshold in enumerate([0.5, 0.8, 0.9, 0.95]):
    for data in cumulative_data:
        if data['cumulative_importance'] >= threshold:
            print(f"  - {int(threshold*100)}% importance reached with {data['n_features']} features")
            break

# Demo 4: Visualization data structure
print("\n" + "=" * 80)
print("DEMO 4: Visualization Data Structure")
print("=" * 80)

print(f"\n Available Visualization Components:")
print(f"  1. Bar Chart Data: {len(response.visualization_data['bar_chart'])} data points")
print(f"  2. Correlation Heatmap: {len(response.correlation_matrix['data'])} correlations")
print(f"  3. Cumulative Curve: {len(response.cumulative_importance)} data points")
print(f"  4. Feature Rankings: {len(response.feature_rankings)} methods")
print(f"  5. Statistics: {len(response.statistics)} method summaries")

print(f"\n Sample Bar Chart Data (first 3 features):")
for data in response.visualization_data['bar_chart'][:3]:
    print(f"  - {data['feature']:20s} ({data['method']:12s}): {data['importance']:.4f}")

print(f"\n Sample Correlation Data (first 3):")
for i, data in enumerate(response.correlation_matrix['data'][:3]):
    if data['correlation'] != 1.0:  # Skip self-correlations
        print(f"  - {data['feature_x']:20s} ↔ {data['feature_y']:20s}: {data['correlation']:+.3f}")

# Demo 5: API Response Format
print("\n" + "=" * 80)
print("DEMO 5: API Response Format Sample")
print("=" * 80)

print("\n Sample API Response Structure:")
sample_response = {
    "metrics": response.metrics,
    "feature_importance": {k: dict(list(v.items())[:3]) for k, v in response.feature_importance.items()},
    "feature_rankings": {k: v[:2] for k, v in response.feature_rankings.items()},
    "execution_time_ms": response.execution_time_ms,
    "model_info": {
        "n_features": response.model_info['n_features'],
        "model_type": response.model_info['model_type'],
        "methods_used": response.model_info['methods_used']
    }
}

print(json.dumps(sample_response, indent=2))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
✅ Feature Importance Analysis is fully implemented with:

  • Multiple Methods: Tree-based (MDI) and Permutation Importance
  • Multiple Models: Random Forest, Gradient Boosting, XGBoost (optional)
  • Multiple Datasets: Housing (regression), Diabetes (regression), Wine (classification)
  • Rich Visualizations: Bar charts, heatmaps, cumulative curves, rankings
  • Comprehensive Metrics: Model performance and feature statistics
  • Flexible Parameters: Configurable estimators, top-k features, methods

  Ready for integration with frontend visualizations!
""")

print("=" * 80)
