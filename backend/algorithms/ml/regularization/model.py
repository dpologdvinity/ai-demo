"""Regularization Techniques model implementation using scikit-learn.

This module demonstrates L1 (Lasso), L2 (Ridge), Elastic Net, and Early Stopping
regularization techniques for preventing overfitting in machine learning models.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time


class RegularizationModel:
    """Regularization Techniques implementation for preventing overfitting.

    This class demonstrates four main regularization approaches:
    1. L1 (Lasso): Sparse coefficients, automatic feature selection
    2. L2 (Ridge): Small but non-zero coefficients, stable predictions
    3. Elastic Net: Combination of L1 and L2 penalties
    4. Early Stopping: Stop training when validation loss stops improving

    Attributes:
        technique: Selected regularization technique
        alpha: Regularization strength
        l1_ratio: Elastic Net mixing parameter
        max_iterations: Maximum training iterations
        models: Dictionary of trained models
        training_time_ms: Training time in milliseconds
    """

    def __init__(
        self,
        technique: str = 'l2',
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iterations: int = 1000,
        early_stopping_rounds: int = 10,
        random_state: int = 42
    ):
        """Initialize Regularization Model.

        Args:
            technique: Regularization type ('l1', 'l2', 'elastic_net', 'early_stopping', 'compare')
            alpha: Regularization strength (lambda parameter)
            l1_ratio: Elastic Net mixing (0.0 = Ridge, 1.0 = Lasso, 0.5 = balanced)
            max_iterations: Maximum number of training iterations
            early_stopping_rounds: Patience for early stopping
            random_state: Random seed for reproducibility
        """
        self.technique = technique
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.models = {}
        self.training_time_ms = 0.0
        self.scaler = StandardScaler()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train regularization model(s) based on selected technique.

        Args:
            X_train: Training features
            y_train: Training target values
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary containing training results and metrics
        """
        start_time = time.time()

        if self.technique == 'compare':
            results = self._train_comparison(X_train, y_train, X_test, y_test)
        elif self.technique == 'l1':
            results = self._train_l1(X_train, y_train, X_test, y_test)
        elif self.technique == 'l2':
            results = self._train_l2(X_train, y_train, X_test, y_test)
        elif self.technique == 'elastic_net':
            results = self._train_elastic_net(X_train, y_train, X_test, y_test)
        elif self.technique == 'early_stopping':
            results = self._train_early_stopping(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown technique: {self.technique}")

        self.training_time_ms = (time.time() - start_time) * 1000
        results['training_time_ms'] = self.training_time_ms

        return results

    def _train_l1(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train L1 (Lasso) regularization model."""
        model = Lasso(alpha=self.alpha, max_iter=self.max_iterations, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models['l1'] = model

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, model.coef_)

        # Generate coefficient path
        coefficient_path = self._generate_coefficient_path(X_train, y_train, 'l1')

        return {
            'model': model,
            'predictions': y_pred_test.tolist(),
            'metrics': metrics,
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'coefficient_path': coefficient_path,
            'sparsity': self._calculate_sparsity(model.coef_)
        }

    def _train_l2(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train L2 (Ridge) regularization model."""
        model = Ridge(alpha=self.alpha, max_iter=self.max_iterations, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models['l2'] = model

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, model.coef_)

        # Generate coefficient path
        coefficient_path = self._generate_coefficient_path(X_train, y_train, 'l2')

        return {
            'model': model,
            'predictions': y_pred_test.tolist(),
            'metrics': metrics,
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'coefficient_path': coefficient_path,
            'sparsity': self._calculate_sparsity(model.coef_)
        }

    def _train_elastic_net(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train Elastic Net (L1 + L2) regularization model."""
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iterations,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models['elastic_net'] = model

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, model.coef_)

        # Generate coefficient path
        coefficient_path = self._generate_coefficient_path(X_train, y_train, 'elastic_net')

        return {
            'model': model,
            'predictions': y_pred_test.tolist(),
            'metrics': metrics,
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'coefficient_path': coefficient_path,
            'sparsity': self._calculate_sparsity(model.coef_),
            'l1_ratio': self.l1_ratio
        }

    def _train_early_stopping(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train model with early stopping based on validation loss."""
        # Split training data to create validation set
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )

        model = SGDRegressor(
            penalty='l2',
            alpha=self.alpha,
            max_iter=1,
            warm_start=True,
            random_state=self.random_state,
            learning_rate='optimal'
        )

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_iteration = 0

        for iteration in range(self.max_iterations):
            # Train one iteration
            model.fit(X_tr, y_tr)

            # Calculate losses
            train_pred = model.predict(X_tr)
            val_pred = model.predict(X_val)
            train_loss = mean_squared_error(y_tr, train_pred)
            val_loss = mean_squared_error(y_val, val_pred)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iteration
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.early_stopping_rounds:
                break

        self.models['early_stopping'] = model

        # Make final predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, model.coef_)
        metrics['optimal_iterations'] = best_iteration + 1
        metrics['total_iterations'] = iteration + 1

        return {
            'model': model,
            'predictions': y_pred_test.tolist(),
            'metrics': metrics,
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_[0]) if hasattr(model.intercept_, '__iter__') else float(model.intercept_),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'optimal_iterations': best_iteration + 1,
            'sparsity': self._calculate_sparsity(model.coef_)
        }

    def _train_comparison(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train and compare all regularization techniques."""
        results = {}

        # Train L1 (Lasso)
        results['l1'] = self._train_l1(X_train, y_train, X_test, y_test)

        # Train L2 (Ridge)
        results['l2'] = self._train_l2(X_train, y_train, X_test, y_test)

        # Train Elastic Net
        results['elastic_net'] = self._train_elastic_net(X_train, y_train, X_test, y_test)

        # Train with Early Stopping
        results['early_stopping'] = self._train_early_stopping(X_train, y_train, X_test, y_test)

        # Compare sparsity
        sparsity_comparison = [
            {'technique': 'L1 (Lasso)', 'sparsity': results['l1']['sparsity'], 'r2_score': results['l1']['metrics']['test_r2']},
            {'technique': 'L2 (Ridge)', 'sparsity': results['l2']['sparsity'], 'r2_score': results['l2']['metrics']['test_r2']},
            {'technique': 'Elastic Net', 'sparsity': results['elastic_net']['sparsity'], 'r2_score': results['elastic_net']['metrics']['test_r2']},
            {'technique': 'Early Stopping', 'sparsity': results['early_stopping']['sparsity'], 'r2_score': results['early_stopping']['metrics']['test_r2']}
        ]

        # Use L2 predictions as default for response
        predictions = results['l2']['predictions']

        return {
            'comparison': results,
            'predictions': predictions,
            'sparsity_comparison': sparsity_comparison
        }

    def _calculate_metrics(
        self,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        coefficients: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation."""
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Overfitting gap
        overfitting_gap = train_r2 - test_r2

        return {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'test_mae': float(test_mae),
            'overfitting_gap': float(overfitting_gap),
            'n_features': len(coefficients),
            'n_nonzero_coefs': int(np.sum(np.abs(coefficients) > 1e-10))
        }

    def _calculate_sparsity(self, coefficients: np.ndarray) -> float:
        """Calculate sparsity (percentage of zero coefficients)."""
        n_zero = np.sum(np.abs(coefficients) < 1e-10)
        return float(n_zero / len(coefficients) * 100)

    def _generate_coefficient_path(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        technique: str
    ) -> List[Dict[str, Any]]:
        """Generate coefficient path for different alpha values."""
        alphas = np.logspace(-3, 2, 20)  # 20 alpha values from 0.001 to 100
        coefficient_path = []

        for alpha_val in alphas:
            if technique == 'l1':
                model = Lasso(alpha=alpha_val, max_iter=1000, random_state=self.random_state)
            elif technique == 'l2':
                model = Ridge(alpha=alpha_val, max_iter=1000, random_state=self.random_state)
            elif technique == 'elastic_net':
                model = ElasticNet(alpha=alpha_val, l1_ratio=self.l1_ratio, max_iter=1000, random_state=self.random_state)
            else:
                continue

            try:
                model.fit(X_train, y_train)
                coefficient_path.append({
                    'alpha': float(alpha_val),
                    'coefficients': model.coef_.tolist(),
                    'n_nonzero': int(np.sum(np.abs(model.coef_) > 1e-10)),
                    'sparsity': self._calculate_sparsity(model.coef_)
                })
            except Exception:
                continue

        return coefficient_path

    def predict(self, X: np.ndarray, technique: str = None) -> np.ndarray:
        """Make predictions using trained model.

        Args:
            X: Features to predict on
            technique: Which model to use (defaults to self.technique)

        Returns:
            Predicted values
        """
        tech = technique or self.technique
        if tech not in self.models:
            raise ValueError(f"Model '{tech}' has not been trained yet")

        return self.models[tech].predict(X)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models.

        Returns:
            Dictionary containing model configuration and parameters
        """
        info = {
            'technique': self.technique,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'max_iterations': self.max_iterations,
            'early_stopping_rounds': self.early_stopping_rounds,
            'models_trained': list(self.models.keys())
        }

        return info
