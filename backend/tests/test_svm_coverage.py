"""Smoke test for SVM algorithm."""

from algorithms.ml.svm.model import SVMModel
from algorithms.ml.svm.data import load_iris_data


class TestSVMSmoke:
    def test_train_basic(self):
        """Test basic SVM training."""
        model = SVMModel()
        data = load_iris_data()
        result = model.train(data['X_train'], data['y_train'])

        assert 'n_support_vectors' in result
        assert result['n_support_vectors'] > 0
        assert result['training_time_ms'] > 0
        assert result['n_classes'] == 3
