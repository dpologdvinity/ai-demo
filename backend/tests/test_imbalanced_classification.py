"""Smoke test for Imbalanced Classification algorithm."""

from algorithms.ml.imbalanced_classification.model import ImbalancedClassificationModel
from algorithms.ml.imbalanced_classification.data import prepare_imbalanced_data


class TestImbalancedClassificationSmoke:
    def test_smote_basic(self):
        """Test SMOTE resampling."""
        X_train, X_test, y_train, y_test, _ = prepare_imbalanced_data(
            n_samples=200, imbalance_ratio=10, test_size=0.3
        )
        model = ImbalancedClassificationModel(model_type='random_forest')
        metrics, roc_data, pr_data = model.train_single_strategy(
            X_train, y_train, X_test, y_test, strategy='smote'
        )

        assert metrics.accuracy > 0
        assert metrics.f1_score >= 0
        assert roc_data['auc'] >= 0
        assert pr_data['auc'] >= 0
