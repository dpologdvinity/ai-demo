"""Hyperparameter Tuning implementation using Grid Search, Random Search, and Bayesian Optimization.

This module implements multiple hyperparameter tuning strategies:
- Grid Search: Exhaustive search over parameter grid
- Random Search: Random sampling from parameter distributions
- Bayesian Optimization: Smart search using Optuna
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer

# Try to import optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .data import load_dataset
from .schema import TrialResult


class HyperparameterTuningModel:
    """Hyperparameter tuning using multiple search strategies.

    Supports:
        - Grid Search: Exhaustive search over parameter grid
        - Random Search: Random sampling from distributions
        - Bayesian Optimization: Smart search using Optuna (if available)
        - Comparison: Run all methods and compare results

    Attributes:
        method: Search method to use
        model_type: Type of model to tune
        best_params: Best hyperparameters found
        best_score: Best cross-validation score
        trials: List of all trial results
    """

    def __init__(self):
        """Initialize Hyperparameter Tuning model."""
        self.method = None
        self.model_type = None
        self.best_params = None
        self.best_score = None
        self.trials = []

    def _get_parameter_space(self, model_type: str) -> Dict[str, Any]:
        """Get parameter space definition for a model.

        Args:
            model_type: Type of model (random_forest, svm, xgboost, mlp).

        Returns:
            Dictionary with parameter distributions for grid/random search.
        """
        if model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        elif model_type == 'svm':
            return {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'degree': [2, 3, 4]  # Only for poly kernel
            }
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not installed")
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        elif model_type == 'mlp':
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [300, 500]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _get_base_model(self, model_type: str, random_state: int = 42):
        """Get base model instance.

        Args:
            model_type: Type of model.
            random_state: Random seed.

        Returns:
            Scikit-learn compatible model instance.
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(random_state=random_state)
        elif model_type == 'svm':
            return SVC(random_state=random_state)
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not installed")
            return XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        elif model_type == 'mlp':
            return MLPClassifier(random_state=random_state, early_stopping=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _get_scorer(self, scoring: str, n_classes: int):
        """Get appropriate scorer for the metric.

        Args:
            scoring: Metric name.
            n_classes: Number of classes.

        Returns:
            Scorer object.
        """
        if scoring == 'accuracy':
            return 'accuracy'
        elif scoring == 'f1':
            return 'f1_weighted' if n_classes > 2 else 'f1'
        elif scoring == 'precision':
            return 'precision_weighted' if n_classes > 2 else 'precision'
        elif scoring == 'recall':
            return 'recall_weighted' if n_classes > 2 else 'recall'
        elif scoring == 'roc_auc':
            if n_classes == 2:
                return 'roc_auc'
            else:
                return 'roc_auc_ovr_weighted'
        else:
            return 'accuracy'

    def _grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str,
        cv_folds: int,
        scoring: str,
        n_classes: int,
        random_state: int
    ) -> Dict[str, Any]:
        """Perform Grid Search.

        Args:
            X_train: Training features.
            y_train: Training labels.
            model_type: Model type.
            cv_folds: Number of CV folds.
            scoring: Scoring metric.
            n_classes: Number of classes.
            random_state: Random seed.

        Returns:
            Dictionary with search results.
        """
        base_model = self._get_base_model(model_type, random_state)
        param_grid = self._get_parameter_space(model_type)
        scorer = self._get_scorer(scoring, n_classes)

        # For grid search, limit parameter space for reasonable runtime
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        elif model_type == 'svm':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring=scorer,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        # Extract trial results
        trials = []
        for i, (params, score) in enumerate(zip(
            grid_search.cv_results_['params'],
            grid_search.cv_results_['mean_test_score']
        )):
            trials.append({
                'trial_number': i,
                'parameters': params,
                'score': float(score),
                'rank': int(grid_search.cv_results_['rank_test_score'][i])
            })

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'trials': trials,
            'cv_results': grid_search.cv_results_
        }

    def _random_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str,
        cv_folds: int,
        scoring: str,
        n_trials: int,
        n_classes: int,
        random_state: int
    ) -> Dict[str, Any]:
        """Perform Random Search.

        Args:
            X_train: Training features.
            y_train: Training labels.
            model_type: Model type.
            cv_folds: Number of CV folds.
            scoring: Scoring metric.
            n_trials: Number of trials.
            n_classes: Number of classes.
            random_state: Random seed.

        Returns:
            Dictionary with search results.
        """
        base_model = self._get_base_model(model_type, random_state)
        param_distributions = self._get_parameter_space(model_type)
        scorer = self._get_scorer(scoring, n_classes)

        random_search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_trials,
            cv=cv_folds,
            scoring=scorer,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )

        random_search.fit(X_train, y_train)

        # Extract trial results
        trials = []
        for i, (params, score) in enumerate(zip(
            random_search.cv_results_['params'],
            random_search.cv_results_['mean_test_score']
        )):
            trials.append({
                'trial_number': i,
                'parameters': params,
                'score': float(score),
                'rank': int(random_search.cv_results_['rank_test_score'][i])
            })

        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_,
            'trials': trials,
            'cv_results': random_search.cv_results_
        }

    def _bayesian_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str,
        cv_folds: int,
        scoring: str,
        n_trials: int,
        n_classes: int,
        random_state: int
    ) -> Dict[str, Any]:
        """Perform Bayesian Optimization using Optuna.

        Args:
            X_train: Training features.
            y_train: Training labels.
            model_type: Model type.
            cv_folds: Number of CV folds.
            scoring: Scoring metric.
            n_trials: Number of trials.
            n_classes: Number of classes.
            random_state: Random seed.

        Returns:
            Dictionary with search results.
        """
        if not OPTUNA_AVAILABLE:
            raise ValueError("Optuna is not installed. Cannot perform Bayesian optimization.")

        # Silence optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Define objective function
        def objective(trial):
            # Suggest hyperparameters based on model type
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                }
            elif model_type == 'svm':
                params = {
                    'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
                }
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 4)
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 9),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif model_type == 'mlp':
                n_layers = trial.suggest_int('n_layers', 1, 2)
                hidden_layer_sizes = []
                for i in range(n_layers):
                    hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 50, 150))
                params = {
                    'hidden_layer_sizes': tuple(hidden_layer_sizes),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                    'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Create and evaluate model
            model = self._get_base_model(model_type, random_state)
            model.set_params(**params)

            # Use cross-validation
            scorer = self._get_scorer(scoring, n_classes)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scorer, n_jobs=-1)

            return scores.mean()

        # Create study and optimize
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Get best parameters and train final model
        best_params = study.best_params
        best_model = self._get_base_model(model_type, random_state)
        best_model.set_params(**best_params)
        best_model.fit(X_train, y_train)

        # Extract trial results
        trials = []
        for i, trial in enumerate(study.trials):
            trials.append({
                'trial_number': i,
                'parameters': trial.params,
                'score': float(trial.value) if trial.value is not None else 0.0,
                'rank': 0  # Will be computed later
            })

        # Rank trials by score
        sorted_trials = sorted(trials, key=lambda x: x['score'], reverse=True)
        for rank, trial in enumerate(sorted_trials, 1):
            trial['rank'] = rank

        # Calculate parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            param_importance = {k: float(v) for k, v in importance.items()}
        except:
            param_importance = None

        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'best_estimator': best_model,
            'trials': trials,
            'study': study,
            'param_importance': param_importance
        }

    def _compare_methods(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str,
        cv_folds: int,
        scoring: str,
        n_trials: int,
        n_classes: int,
        random_state: int
    ) -> Dict[str, Any]:
        """Compare all tuning methods.

        Args:
            X_train: Training features.
            y_train: Training labels.
            model_type: Model type.
            cv_folds: Number of CV folds.
            scoring: Scoring metric.
            n_trials: Number of trials for random/bayesian.
            n_classes: Number of classes.
            random_state: Random seed.

        Returns:
            Dictionary with comparison results.
        """
        results = {}

        # Grid Search
        try:
            grid_result = self._grid_search(
                X_train, y_train, model_type, cv_folds, scoring, n_classes, random_state
            )
            results['grid'] = {
                'best_score': grid_result['best_score'],
                'best_params': grid_result['best_params'],
                'n_trials': len(grid_result['trials'])
            }
        except Exception as e:
            results['grid'] = {'error': str(e)}

        # Random Search
        try:
            random_result = self._random_search(
                X_train, y_train, model_type, cv_folds, scoring, n_trials, n_classes, random_state
            )
            results['random'] = {
                'best_score': random_result['best_score'],
                'best_params': random_result['best_params'],
                'n_trials': len(random_result['trials'])
            }
        except Exception as e:
            results['random'] = {'error': str(e)}

        # Bayesian Search (if available)
        if OPTUNA_AVAILABLE:
            try:
                bayesian_result = self._bayesian_search(
                    X_train, y_train, model_type, cv_folds, scoring, n_trials, n_classes, random_state
                )
                results['bayesian'] = {
                    'best_score': bayesian_result['best_score'],
                    'best_params': bayesian_result['best_params'],
                    'n_trials': len(bayesian_result['trials'])
                }
            except Exception as e:
                results['bayesian'] = {'error': str(e)}

        # Determine best method
        best_method = max(
            [k for k in results.keys() if 'error' not in results[k]],
            key=lambda k: results[k]['best_score']
        )

        return {
            'comparison': results,
            'best_method': best_method,
            'best_score': results[best_method]['best_score'],
            'best_params': results[best_method]['best_params']
        }

    def train(
        self,
        method: str = 'grid',
        model_type: str = 'random_forest',
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        dataset: str = 'iris',
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning.

        Args:
            method: Search method (grid, random, bayesian, compare).
            model_type: Model to tune (random_forest, svm, xgboost, mlp).
            n_trials: Number of trials for random/bayesian search.
            cv_folds: Number of cross-validation folds.
            scoring: Evaluation metric.
            dataset: Dataset to use.
            random_state: Random seed.

        Returns:
            Dictionary with tuning results.
        """
        start_time = time.time()

        try:
            # Load data
            data = load_dataset(dataset, random_state=random_state)
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']
            n_classes = data['n_classes']

            # Perform search based on method
            if method == 'grid':
                result = self._grid_search(
                    X_train, y_train, model_type, cv_folds, scoring, n_classes, random_state
                )
            elif method == 'random':
                result = self._random_search(
                    X_train, y_train, model_type, cv_folds, scoring, n_trials, n_classes, random_state
                )
            elif method == 'bayesian':
                result = self._bayesian_search(
                    X_train, y_train, model_type, cv_folds, scoring, n_trials, n_classes, random_state
                )
            elif method == 'compare':
                comparison = self._compare_methods(
                    X_train, y_train, model_type, cv_folds, scoring, n_trials, n_classes, random_state
                )
                execution_time_ms = (time.time() - start_time) * 1000

                return {
                    'success': True,
                    'method': 'compare',
                    'comparison_data': comparison['comparison'],
                    'best_method': comparison['best_method'],
                    'best_score': comparison['best_score'],
                    'best_params': comparison['best_params'],
                    'execution_time_ms': execution_time_ms,
                    'parameters_used': {
                        'method': method,
                        'model_type': model_type,
                        'cv_folds': cv_folds,
                        'scoring': scoring,
                        'dataset': dataset
                    }
                }
            else:
                raise ValueError(f"Unknown method: {method}")

            # Evaluate best model on test set
            best_estimator = result['best_estimator']
            y_pred = best_estimator.predict(X_test)

            if scoring == 'accuracy':
                test_score = accuracy_score(y_test, y_pred)
            elif scoring == 'f1':
                test_score = f1_score(y_test, y_pred, average='weighted' if n_classes > 2 else 'binary')
            elif scoring == 'precision':
                test_score = precision_score(y_test, y_pred, average='weighted' if n_classes > 2 else 'binary')
            elif scoring == 'recall':
                test_score = recall_score(y_test, y_pred, average='weighted' if n_classes > 2 else 'binary')
            else:
                test_score = accuracy_score(y_test, y_pred)

            # Prepare convergence data (best score over trials)
            convergence_data = []
            best_so_far = float('-inf')
            for trial in result['trials']:
                best_so_far = max(best_so_far, trial['score'])
                convergence_data.append({
                    'trial': trial['trial_number'],
                    'best_score': float(best_so_far),
                    'current_score': float(trial['score'])
                })

            # Prepare heatmap data for 2D parameter visualization
            heatmap_data = self._prepare_heatmap_data(result['trials'], model_type)

            execution_time_ms = (time.time() - start_time) * 1000

            return {
                'success': True,
                'method': method,
                'best_score': float(result['best_score']),
                'best_params': result['best_params'],
                'best_estimator_test_score': float(test_score),
                'trials': result['trials'],
                'convergence_data': convergence_data,
                'heatmap_data': heatmap_data,
                'parameter_importance': result.get('param_importance'),
                'execution_time_ms': execution_time_ms,
                'parameters_used': {
                    'method': method,
                    'model_type': model_type,
                    'n_trials': n_trials if method in ['random', 'bayesian'] else len(result['trials']),
                    'cv_folds': cv_folds,
                    'scoring': scoring,
                    'dataset': dataset,
                    'random_state': random_state
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': (time.time() - start_time) * 1000
            }

    def _prepare_heatmap_data(self, trials: List[Dict], model_type: str) -> Dict[str, Any]:
        """Prepare heatmap data for parameter space visualization.

        Args:
            trials: List of trial results.
            model_type: Model type.

        Returns:
            Dictionary with heatmap data.
        """
        if not trials:
            return None

        # Select two most important parameters for 2D visualization
        if model_type == 'random_forest':
            param_x = 'n_estimators'
            param_y = 'max_depth'
        elif model_type == 'svm':
            param_x = 'C'
            param_y = 'gamma'
        elif model_type == 'xgboost':
            param_x = 'learning_rate'
            param_y = 'max_depth'
        elif model_type == 'mlp':
            param_x = 'alpha'
            param_y = 'hidden_layer_sizes'
        else:
            return None

        # Extract parameter values and scores
        heatmap_points = []
        for trial in trials:
            params = trial['parameters']
            if param_x in params and param_y in params:
                x_val = params[param_x]
                y_val = params[param_y]

                # Convert to numeric if possible
                if isinstance(y_val, tuple):
                    y_val = y_val[0] if len(y_val) > 0 else 0

                heatmap_points.append({
                    'x': float(x_val) if isinstance(x_val, (int, float)) else hash(str(x_val)),
                    'y': float(y_val) if isinstance(y_val, (int, float)) else hash(str(y_val)),
                    'score': trial['score'],
                    'params': params
                })

        return {
            'param_x': param_x,
            'param_y': param_y,
            'points': heatmap_points
        }
