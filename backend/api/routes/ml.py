from fastapi import APIRouter, HTTPException
import time
from typing import Dict, Any

from algorithms.ml.linear_regression.model import LinearRegressionModel
from algorithms.ml.linear_regression.schema import (
    LinearRegressionRequest,
    LinearRegressionResponse
)
from algorithms.ml.linear_regression.data import (
    load_linear_regression_data,
    get_available_datasets
)
from algorithms.ml.random_forest.model import RandomForestModel
from algorithms.ml.random_forest.schema import (
    RandomForestRequest,
    RandomForestResponse
)
from algorithms.ml.random_forest.data import get_dataset_info
from algorithms.ml.ridge_regression import (
    RidgeRegressionModel,
    RidgeRegressionRequest,
    RidgeRegressionResponse
)
from algorithms.ml.ridge_regression.data import (
    load_housing_data,
    get_dataset_info as get_ridge_dataset_info
)
from algorithms.ml.lasso_regression import (
    LassoRegressionModel,
    LassoRegressionRequest,
    LassoRegressionResponse,
)
from algorithms.ml.lasso_regression.data import load_boston_data as load_lasso_data, get_dataset_info as get_lasso_dataset_info
from algorithms.ml.elastic_net import (
    ElasticNetModel,
    ElasticNetRequest,
    ElasticNetResponse,
)
from algorithms.ml.elastic_net.data import load_housing_data as load_elastic_net_data, get_dataset_info as get_elastic_net_dataset_info
from algorithms.ml.xgboost import XGBoostModel, XGBoostRequest, XGBoostResponse
from algorithms.ml.k_means import KMeansModel, KMeansParameters, KMeansResponse
from algorithms.ml.dbscan import DBSCANModel, DBSCANParameters, DBSCANResponse
from algorithms.ml.knn import train_knn, KNNRequest, KNNResponse
from algorithms.ml.naive_bayes import (
    NaiveBayesModel,
    NaiveBayesRequest,
    NaiveBayesResponse
)
from algorithms.ml.naive_bayes.data import get_supported_datasets as get_nb_datasets
from algorithms.ml.gmm import GaussianMixtureModelClass, GMMRequest, GMMResponse
from algorithms.ml.gmm.data import load_blobs_data, get_dataset_info as get_gmm_dataset_info
from algorithms.ml.adaboost import AdaBoostModel, AdaBoostRequest, AdaBoostResponse
from algorithms.ml.adaboost.data import prepare_data as prepare_adaboost_data, prepare_visualization_data as prepare_adaboost_viz
from algorithms.ml.adaboost.schema import AdaBoostMetrics, VisualizationData as AdaBoostVisualizationData
from algorithms.ml.regularization import (
    RegularizationModel,
    RegularizationRequest,
    RegularizationResponse,
)
from algorithms.ml.regularization.data import (
    create_overfitting_prone_dataset,
    get_dataset_info as get_regularization_dataset_info
)
from utils.algorithm_metadata import (
    AlgorithmMetadata,
    AlgorithmParameter,
    AlgorithmComplexity,
    AlgorithmCategory,
    DifficultyLevel,
    AlgorithmRegistry,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# Register Linear Regression metadata
linear_regression_metadata = AlgorithmMetadata(
    id="linear-regression",
    name="Linear Regression",
    slug="linear-regression",
    category=AlgorithmCategory.ML,
    description="Supervised learning algorithm for predicting continuous values using linear relationships",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["supervised", "regression", "linear-models"],
    use_cases=[
        "House price prediction",
        "Sales forecasting",
        "Risk assessment",
        "Trend analysis",
        "Economic modeling"
    ],
    complexity=AlgorithmComplexity(
        time="O(n²)",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="fit_intercept",
            label="Fit Intercept",
            type="select",
            default=True,
            options=[
                {"label": "True", "value": True},
                {"label": "False", "value": False}
            ],
            description="Whether to calculate the intercept for the model"
        ),
        AlgorithmParameter(
            name="normalize",
            label="Normalize Features",
            type="select",
            default=True,
            options=[
                {"label": "True", "value": True},
                {"label": "False", "value": False}
            ],
            description="Whether to normalize features before regression"
        ),
    ],
    dataset_name="boston",
    visualization_type="line_chart",
    theory=(
        "Linear Regression models the relationship between a dependent variable (target) "
        "and one or more independent variables (features) using a linear equation. "
        "It finds the best-fitting straight line through the data points by minimizing "
        "the sum of squared residuals (difference between predicted and actual values). "
        "The model learns coefficients (weights) for each feature and an optional intercept (bias) term."
    ),
    pros=[
        "Simple and easy to interpret",
        "Fast training and prediction",
        "Works well for linearly separable data",
        "Provides feature importance via coefficients",
        "No hyperparameters to tune"
    ],
    cons=[
        "Assumes linear relationship between features and target",
        "Sensitive to outliers",
        "Cannot capture complex non-linear patterns",
        "Assumes features are independent (no multicollinearity)",
        "Requires features to be scaled for best performance"
    ],
    related_algorithms=["ridge-regression", "lasso-regression", "polynomial-regression"]
)

AlgorithmRegistry.register(linear_regression_metadata)


@router.get("/")
async def ml_root():
    return {"message": "Machine Learning algorithms endpoint"}


@router.get("/algorithms")
async def list_ml_algorithms():
    """Get all registered ML algorithms.

    Returns:
        List of algorithm metadata for all registered ML algorithms
    """
    algorithms = AlgorithmRegistry.get_by_category(AlgorithmCategory.ML)
    return [algo.model_dump() for algo in algorithms]


@router.post("/linear-regression/train", response_model=LinearRegressionResponse)
async def train_linear_regression(request: LinearRegressionRequest):
    """Train a Linear Regression model.

    This endpoint trains a Linear Regression model on the specified dataset
    and returns predictions, metrics, and visualization data.

    Args:
        request: LinearRegressionRequest containing training parameters

    Returns:
        LinearRegressionResponse with training results and metrics

    Raises:
        HTTPException: If training fails due to invalid parameters or data issues
    """
    try:
        start_time = time.time()

        # Load data
        data = load_linear_regression_data(
            dataset_name=request.dataset_name,
            test_size=request.test_size,
            normalize=request.normalize
        )

        # Initialize and train model
        model = LinearRegressionModel(fit_intercept=request.fit_intercept)
        training_info = model.train(data['X_train'], data['y_train'])

        # Make predictions on test set
        predictions = model.predict(data['X_test'])

        # Evaluate model
        metrics = model.evaluate(data['X_test'], data['y_test'])

        # Get model info
        model_info = model.get_model_info()

        # Prepare visualization data
        chart_data = []
        for i, (pred, actual) in enumerate(zip(predictions[:100], data['y_test'][:100])):
            chart_data.append({
                "index": i,
                "predicted": float(pred),
                "actual": float(actual)
            })

        visualization_data = {
            "chart_data": chart_data,
            "x_label": "Sample Index",
            "y_label": "House Price (in $100k)",
            "title": "Predictions vs Actual Values"
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return LinearRegressionResponse(
            success=True,
            metrics=metrics,
            predictions=predictions.tolist(),
            actual=data['y_test'].tolist(),
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                "fit_intercept": request.fit_intercept,
                "normalize": request.normalize,
                "dataset_name": request.dataset_name,
                "test_size": request.test_size
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/linear-regression/info")
async def get_linear_regression_info() -> Dict[str, Any]:
    """Get Linear Regression algorithm information.

    Returns metadata, parameters, and available datasets for Linear Regression.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("linear-regression")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Linear Regression metadata not found"
        )

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": get_available_datasets()
    }


# Register Random Forest metadata
random_forest_metadata = AlgorithmMetadata(
    id="random-forest",
    name="Random Forest",
    slug="random-forest",
    category=AlgorithmCategory.ML,
    description="Ensemble method using multiple decision trees for robust predictions",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["supervised", "classification", "ensemble", "tree-based"],
    use_cases=[
        "Fraud detection",
        "Stock market analysis",
        "Healthcare predictions"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*log(n)*d*k)",
        space="O(k*n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_estimators",
            label="Number of Trees",
            type="range",
            default=100,
            min=10,
            max=500,
            step=10,
            description="Number of decision trees in the forest"
        ),
        AlgorithmParameter(
            name="max_depth",
            label="Maximum Depth",
            type="number",
            default=None,
            min=1,
            max=30,
            step=1,
            description="Maximum depth of each tree (None for unlimited)"
        ),
        AlgorithmParameter(
            name="min_samples_split",
            label="Min Samples Split",
            type="range",
            default=2,
            min=2,
            max=20,
            step=1,
            description="Minimum samples required to split an internal node"
        ),
        AlgorithmParameter(
            name="max_features",
            label="Max Features",
            type="select",
            default="sqrt",
            options=[
                {"label": "Square Root", "value": "sqrt"},
                {"label": "Log2", "value": "log2"},
                {"label": "All Features", "value": "None"}
            ],
            description="Number of features to consider for the best split"
        )
    ],
    dataset_name="wine",
    visualization_type="feature_importance",
    theory=(
        "Random Forest builds multiple decision trees and merges them together to get a more accurate "
        "and stable prediction. Each tree is trained on a random subset of the data (bootstrap sample) "
        "and at each split, a random subset of features is considered. This randomness helps to make "
        "the model more robust and prevents overfitting."
    ),
    pros=[
        "Reduces overfitting compared to single decision trees",
        "Handles missing values well",
        "Provides feature importance scores",
        "Works well with both classification and regression",
        "Robust to outliers and noise"
    ],
    cons=[
        "Can be slow to train with many trees",
        "Less interpretable than a single decision tree",
        "Requires more memory",
        "May not perform well on very small datasets"
    ],
    related_algorithms=["decision-tree", "gradient-boosting", "adaboost"]
)

AlgorithmRegistry.register(random_forest_metadata)


# Register Ridge Regression metadata
ridge_regression_metadata = AlgorithmMetadata(
    id="ridge-regression",
    name="Ridge Regression",
    slug="ridge-regression",
    category=AlgorithmCategory.ML,
    description="Linear regression with L2 regularization to prevent overfitting",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["supervised", "regression", "linear-models", "regularization"],
    use_cases=[
        "Multicollinear data",
        "High-dimensional regression",
        "Regularized prediction",
        "Feature selection with shrinkage",
        "Preventing overfitting in regression"
    ],
    complexity=AlgorithmComplexity(
        time="O(n²)",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="alpha",
            label="Regularization Strength (Alpha)",
            type="range",
            default=1.0,
            min=0.01,
            max=100.0,
            step=0.01,
            description="Regularization strength. Larger values specify stronger regularization."
        ),
        AlgorithmParameter(
            name="fit_intercept",
            label="Fit Intercept",
            type="select",
            default=True,
            options=[
                {"label": "True", "value": True},
                {"label": "False", "value": False}
            ],
            description="Whether to calculate the intercept for the model"
        ),
        AlgorithmParameter(
            name="solver",
            label="Solver",
            type="select",
            default="auto",
            options=[
                {"label": "Auto", "value": "auto"},
                {"label": "SVD", "value": "svd"},
                {"label": "Cholesky", "value": "cholesky"},
                {"label": "LSQR", "value": "lsqr"}
            ],
            description="Solver to use in the computational routines"
        ),
    ],
    dataset_name="boston",
    visualization_type="line_chart",
    theory=(
        "Ridge Regression is a regularized version of Linear Regression that adds an L2 penalty term "
        "to the loss function. This penalty term is proportional to the square of the magnitude of "
        "coefficients (controlled by the alpha parameter). Ridge Regression helps prevent overfitting "
        "by shrinking coefficients, making it especially useful when features are correlated "
        "(multicollinearity) or when the number of features is large relative to the number of samples. "
        "Unlike Lasso, Ridge keeps all features but reduces their impact, making it suitable when "
        "you believe all features contribute to the prediction."
    ),
    pros=[
        "Prevents overfitting through L2 regularization",
        "Handles multicollinearity well",
        "Works with high-dimensional data",
        "More stable than ordinary linear regression",
        "Computational efficiency with closed-form solution"
    ],
    cons=[
        "Does not perform feature selection (keeps all features)",
        "Requires careful tuning of alpha parameter",
        "Less interpretable than ordinary linear regression",
        "Assumes linear relationship between features and target",
        "All features should be on similar scales"
    ],
    related_algorithms=["linear-regression", "lasso-regression", "elastic-net"]
)

AlgorithmRegistry.register(ridge_regression_metadata)


# Register Lasso Regression metadata
lasso_regression_metadata = AlgorithmMetadata(
    id="lasso-regression",
    name="Lasso Regression",
    slug="lasso-regression",
    category=AlgorithmCategory.ML,
    description="Linear regression with L1 regularization for feature selection",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["supervised", "regression", "linear-models", "regularization", "feature-selection"],
    use_cases=[
        "Feature selection in high-dimensional data",
        "Sparse model creation",
        "Preventing overfitting in regression",
    ],
    complexity=AlgorithmComplexity(time="O(n²)", space="O(n)"),
    parameters=[
        AlgorithmParameter(
            name="alpha",
            label="Regularization Strength",
            type="range",
            default=1.0,
            min=0.01,
            max=10.0,
            step=0.1,
            description="L1 penalty strength. Higher values create sparser models."
        ),
        AlgorithmParameter(
            name="max_iter",
            label="Maximum Iterations",
            type="number",
            default=1000,
            min=100,
            max=5000,
            step=100,
            description="Maximum iterations for optimization algorithm."
        ),
        AlgorithmParameter(
            name="selection",
            label="Coefficient Update",
            type="select",
            default="cyclic",
            options=[
                {"value": "cyclic", "label": "Cyclic (Sequential)"},
                {"value": "random", "label": "Random (Faster)"},
            ],
            description="Strategy for updating coefficients during optimization."
        ),
    ],
    dataset_name="boston",
    visualization_type="line_chart",
    theory=(
        "Lasso (Least Absolute Shrinkage and Selection Operator) Regression adds "
        "L1 regularization to linear regression, which adds a penalty equal to the "
        "absolute value of the magnitude of coefficients. This encourages sparsity "
        "by driving some coefficients to exactly zero, effectively performing feature "
        "selection. The optimization problem is: minimize ||y - Xw||² + α||w||₁"
    ),
    pros=[
        "Automatic feature selection by setting coefficients to zero",
        "Prevents overfitting through regularization",
        "Produces interpretable sparse models",
        "Works well with high-dimensional data",
    ],
    cons=[
        "May arbitrarily select one feature from correlated group",
        "Can be unstable with highly correlated features",
        "Requires tuning of regularization parameter",
        "May underperform with small sample sizes",
    ],
    related_algorithms=["linear-regression", "ridge-regression", "elastic-net"],
)

AlgorithmRegistry.register(lasso_regression_metadata)


# Register Elastic Net metadata
elastic_net_metadata = AlgorithmMetadata(
    id="elastic-net",
    name="Elastic Net",
    slug="elastic-net",
    category=AlgorithmCategory.ML,
    description="Linear regression with combined L1 and L2 regularization",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["supervised", "regression", "regularization", "linear-models"],
    use_cases=[
        "High-dimensional regression with correlated features",
        "Grouped variable selection",
        "When both L1 and L2 regularization benefits are needed",
    ],
    complexity=AlgorithmComplexity(time="O(n²)", space="O(n)"),
    parameters=[
        AlgorithmParameter(
            name="alpha",
            label="Regularization Strength",
            type="range",
            default=1.0,
            min=0.01,
            max=10.0,
            step=0.1,
            description="Overall regularization strength. Higher values create more regularization."
        ),
        AlgorithmParameter(
            name="l1_ratio",
            label="L1 Ratio (Lasso ↔ Ridge Mix)",
            type="range",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Mix between L1 and L2. 0=Ridge, 1=Lasso, 0.5=Equal mix."
        ),
        AlgorithmParameter(
            name="max_iter",
            label="Maximum Iterations",
            type="number",
            default=1000,
            min=100,
            max=5000,
            step=100,
            description="Maximum iterations for optimization algorithm."
        ),
        AlgorithmParameter(
            name="fit_intercept",
            label="Fit Intercept",
            type="select",
            default=True,
            options=[
                {"value": True, "label": "Yes"},
                {"value": False, "label": "No"},
            ],
            description="Whether to calculate the intercept for the model."
        ),
    ],
    dataset_name="boston",
    visualization_type="line_chart",
    theory=(
        "Elastic Net linearly combines the L1 and L2 penalties of Lasso and Ridge regression. "
        "The optimization problem is: minimize ||y - Xw||² + α·ρ·||w||₁ + α·(1-ρ)/2·||w||² "
        "where α is the overall regularization strength and ρ (l1_ratio) controls the L1/L2 mix. "
        "This combination is particularly useful when there are multiple correlated features, as "
        "Lasso tends to pick one feature from a correlated group while Elastic Net tends to select "
        "all of them with distributed weights."
    ),
    pros=[
        "Combines benefits of both L1 (sparsity) and L2 (stability) regularization",
        "Handles correlated features better than Lasso",
        "Encourages grouped selection of correlated features",
        "More stable than Lasso with highly correlated features",
    ],
    cons=[
        "Two hyperparameters to tune (alpha and l1_ratio)",
        "More computationally expensive than Ridge or Lasso alone",
        "May be overkill for simple problems",
        "Requires cross-validation to find optimal parameter combination",
    ],
    related_algorithms=["linear-regression", "ridge-regression", "lasso-regression"],
)

AlgorithmRegistry.register(elastic_net_metadata)


@router.post("/random-forest/train", response_model=RandomForestResponse)
async def train_random_forest(request: RandomForestRequest):
    """Train a Random Forest model on the wine dataset.

    Args:
        request: RandomForestRequest containing model hyperparameters

    Returns:
        RandomForestResponse with metrics, predictions, and feature importance

    Raises:
        HTTPException: If training fails
    """
    try:
        model = RandomForestModel()
        response = model.train(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/random-forest/info")
async def get_random_forest_info() -> Dict[str, Any]:
    """Get metadata and dataset information for Random Forest algorithm.

    Returns:
        Dictionary containing algorithm metadata and dataset information
    """
    metadata = AlgorithmRegistry.get("random-forest")
    dataset_info = get_dataset_info()

    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    return {
        "algorithm": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/ridge-regression/train", response_model=RidgeRegressionResponse)
async def train_ridge_regression(request: RidgeRegressionRequest):
    """Train a Ridge Regression model.

    This endpoint trains a Ridge Regression model with L2 regularization on the
    California Housing dataset and returns predictions, metrics, and coefficient analysis.

    Args:
        request: RidgeRegressionRequest containing training parameters

    Returns:
        RidgeRegressionResponse with training results, metrics, and visualization data

    Raises:
        HTTPException: If training fails due to invalid parameters or data issues
    """
    try:
        start_time = time.time()

        # Load California Housing data
        data = load_housing_data()

        # Initialize and train model
        model = RidgeRegressionModel(
            alpha=request.alpha,
            fit_intercept=request.fit_intercept,
            solver=request.solver
        )
        training_info = model.train(data['X_train'], data['y_train'])

        # Make predictions on test set
        predictions = model.predict(data['X_test'])

        # Evaluate model
        metrics = model.evaluate(data['X_test'], data['y_test'])

        # Get model info
        model_info = model.get_model_info()

        # Get coefficient analysis
        coefficient_analysis = model.get_coefficient_analysis(
            feature_names=list(data['feature_names'])
        )

        # Prepare visualization data for predictions vs actual
        predictions_vs_actual = []
        for i, (pred, actual) in enumerate(zip(predictions[:100], data['y_test'][:100])):
            predictions_vs_actual.append({
                "index": i,
                "predicted": float(pred),
                "actual": float(actual)
            })

        # Prepare coefficient plot data
        coefficient_plot = []
        for feature, coef in coefficient_analysis['coefficients'].items():
            coefficient_plot.append({
                "feature": feature,
                "coefficient": float(coef),
                "coefficient_abs": float(coefficient_analysis['coefficients_abs'][feature])
            })
        # Sort by absolute value for better visualization
        coefficient_plot.sort(key=lambda x: x['coefficient_abs'], reverse=True)

        visualization_data = {
            "predictions_vs_actual": predictions_vs_actual,
            "coefficient_plot": coefficient_plot,
            "x_label": "Sample Index",
            "y_label": "House Price (in $100k)",
            "title": "Ridge Regression: Predictions vs Actual Values"
        }

        execution_time_ms = (time.time() - start_time) * 1000

        return RidgeRegressionResponse(
            metrics=metrics,
            predictions=predictions[:100].tolist(),  # Limit for response size
            actual_values=data['y_test'][:100].tolist(),
            coefficients=coefficient_analysis['coefficients'],
            coefficient_analysis=coefficient_analysis,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info
        )

    except ValueError as e:
        logger.error(f"ValueError in Ridge Regression training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in Ridge Regression training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/ridge-regression/info")
async def get_ridge_regression_info() -> Dict[str, Any]:
    """Get Ridge Regression algorithm information.

    Returns metadata, parameters, and dataset information for Ridge Regression.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("ridge-regression")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Ridge Regression metadata not found"
        )

    dataset_info = get_ridge_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/lasso-regression/train", response_model=LassoRegressionResponse)
async def train_lasso_regression(request: LassoRegressionRequest):
    """Train a Lasso Regression model with L1 regularization.

    This endpoint trains a Lasso Regression model on the California Housing dataset,
    demonstrating feature selection through L1 regularization.

    Args:
        request: Training parameters including alpha, max_iter, and selection strategy

    Returns:
        LassoRegressionResponse containing metrics, predictions, and visualization data

    Raises:
        HTTPException: If training fails
    """
    try:
        from utils.datasets import DatasetManager

        start_time = time.time()

        # Load dataset
        data = load_lasso_data()
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data['feature_names']

        # Normalize if requested
        if request.normalize:
            X_train, X_test = DatasetManager.normalize_data(X_train, X_test)

        # Create and train model
        model = LassoRegressionModel(
            alpha=request.alpha,
            max_iter=request.max_iter,
            selection=request.selection,
        )

        train_result = model.train(X_train, y_train)

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)

        # Make predictions
        predictions = model.predict(X_test)

        # Get feature importance
        feature_importance = model.get_feature_importance(feature_names)

        # Prepare visualization data
        chart_data = [
            {
                "index": i,
                "predicted": float(pred),
                "actual": float(actual)
            }
            for i, (pred, actual) in enumerate(zip(predictions[:100], y_test[:100]))
        ]

        # Prepare coefficient data showing sparsity
        coefficient_data = [
            {
                "feature": feature,
                "coefficient": float(coef),
                "selected": abs(coef) > 1e-10
            }
            for feature, coef in zip(feature_names, train_result['coefficients'])
        ]

        execution_time_ms = (time.time() - start_time) * 1000

        return LassoRegressionResponse(
            success=True,
            metrics=metrics,
            predictions=predictions.tolist()[:100],
            actual=y_test.tolist()[:100],
            visualization_data={
                "chart_data": chart_data,
                "coefficient_data": coefficient_data,
            },
            execution_time_ms=execution_time_ms,
            model_info=train_result,
            feature_importance=feature_importance,
            parameters_used={
                "alpha": request.alpha,
                "max_iter": request.max_iter,
                "selection": request.selection,
                "normalize": request.normalize,
            },
        )

    except ValueError as e:
        logger.error(f"ValueError in Lasso Regression training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in Lasso Regression training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/lasso-regression/info")
async def get_lasso_regression_info():
    """Get metadata and information about Lasso Regression algorithm.

    Returns:
        Algorithm metadata including parameters, complexity, and documentation
    """
    from algorithms.ml.lasso_regression.data import get_dataset_info

    metadata = AlgorithmRegistry.get("lasso-regression")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    dataset_info = get_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/elastic-net/train", response_model=ElasticNetResponse)
async def train_elastic_net(request: ElasticNetRequest):
    """Train an Elastic Net Regression model with combined L1 and L2 regularization.

    This endpoint trains an Elastic Net Regression model on the California Housing dataset,
    demonstrating the benefits of combining L1 (Lasso) and L2 (Ridge) regularization.

    Args:
        request: Training parameters including alpha, l1_ratio, max_iter, and fit_intercept

    Returns:
        ElasticNetResponse containing metrics, predictions, and visualization data

    Raises:
        HTTPException: If training fails
    """
    try:
        from utils.datasets import DatasetManager

        start_time = time.time()

        # Load dataset
        data = load_elastic_net_data()
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data['feature_names']

        # Normalize if requested
        if request.normalize:
            X_train, X_test = DatasetManager.normalize_data(X_train, X_test)

        # Create and train model
        model = ElasticNetModel(
            alpha=request.alpha,
            l1_ratio=request.l1_ratio,
            max_iter=request.max_iter,
            fit_intercept=request.fit_intercept,
        )

        train_result = model.train(X_train, y_train)

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)

        # Make predictions
        predictions = model.predict(X_test)

        # Get feature importance
        feature_importance = model.get_feature_importance(feature_names)

        # Prepare visualization data
        chart_data = [
            {
                "index": i,
                "predicted": float(pred),
                "actual": float(actual)
            }
            for i, (pred, actual) in enumerate(zip(predictions[:100], y_test[:100]))
        ]

        # Prepare coefficient data showing both sparsity and magnitude
        coefficient_data = [
            {
                "feature": feature,
                "coefficient": float(coef),
                "magnitude": abs(float(coef)),
                "selected": abs(coef) > 1e-10
            }
            for feature, coef in zip(feature_names, train_result['coefficients'])
        ]

        # Generate regularization path comparison
        comparison_result = model.compare_regularization_types(
            X_train, y_train, X_test, y_test, feature_names
        )

        # Format regularization path for visualization
        regularization_path = [
            {
                "l1_ratio": item["l1_ratio"],
                "type": "Ridge" if item["l1_ratio"] == 0.0 else "Lasso" if item["l1_ratio"] == 1.0 else "Elastic Net",
                "r2_score": item["r2_score"],
                "n_nonzero": item["n_nonzero_coefs"],
                "sparsity": item["sparsity"],
                "l2_norm": item["l2_norm"]
            }
            for item in comparison_result["comparison_data"]
        ]

        execution_time_ms = (time.time() - start_time) * 1000

        return ElasticNetResponse(
            success=True,
            metrics=metrics,
            predictions=predictions.tolist()[:100],
            actual=y_test.tolist()[:100],
            visualization_data={
                "chart_data": chart_data,
                "coefficient_data": coefficient_data,
                "regularization_path": regularization_path,
            },
            execution_time_ms=execution_time_ms,
            model_info=train_result,
            feature_importance=feature_importance,
            parameters_used={
                "alpha": request.alpha,
                "l1_ratio": request.l1_ratio,
                "max_iter": request.max_iter,
                "fit_intercept": request.fit_intercept,
                "normalize": request.normalize,
            },
        )

    except ValueError as e:
        logger.error(f"ValueError in Elastic Net training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in Elastic Net training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/elastic-net/info")
async def get_elastic_net_info():
    """Get metadata and information about Elastic Net Regression algorithm.

    Returns:
        Algorithm metadata including parameters, complexity, and documentation
    """
    from algorithms.ml.elastic_net.data import get_dataset_info

    metadata = AlgorithmRegistry.get("elastic-net")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    dataset_info = get_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register K-Means metadata
k_means_metadata = AlgorithmMetadata(
    id="k-means",
    name="K-Means Clustering",
    slug="k-means",
    category=AlgorithmCategory.ML,
    description="Unsupervised learning algorithm that groups data into K clusters",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["unsupervised", "clustering", "partitioning"],
    use_cases=[
        "Customer segmentation",
        "Image compression",
        "Document clustering"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*k*i)",
        space="O(n*k)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_clusters",
            label="Number of Clusters (K)",
            type="number",
            default=3,
            min=2,
            max=10,
            step=1,
            description="Number of clusters to form"
        ),
        AlgorithmParameter(
            name="max_iter",
            label="Maximum Iterations",
            type="number",
            default=300,
            min=50,
            max=1000,
            step=50,
            description="Maximum number of iterations for convergence"
        ),
        AlgorithmParameter(
            name="n_init",
            label="Number of Initializations",
            type="number",
            default=10,
            min=1,
            max=20,
            step=1,
            description="Number of times to run with different centroid seeds"
        ),
        AlgorithmParameter(
            name="random_state",
            label="Random Seed",
            type="number",
            default=42,
            min=0,
            max=100,
            step=1,
            description="Random seed for reproducibility"
        ),
        AlgorithmParameter(
            name="n_samples",
            label="Number of Samples",
            type="number",
            default=300,
            min=100,
            max=1000,
            step=50,
            description="Number of data points to generate"
        )
    ],
    dataset_name="blobs",
    visualization_type="scatter_plot",
    theory=(
        "K-Means is an iterative algorithm that divides data into K clusters. "
        "It works by: (1) Randomly initializing K centroids, (2) Assigning each "
        "point to the nearest centroid, (3) Updating centroids to the mean of "
        "assigned points, and (4) Repeating steps 2-3 until convergence."
    ),
    pros=[
        "Simple and intuitive algorithm",
        "Fast and efficient for large datasets",
        "Works well with spherical clusters"
    ],
    cons=[
        "Requires specifying K in advance",
        "Sensitive to initial centroid placement",
        "Struggles with non-spherical clusters",
        "Affected by outliers"
    ],
    related_algorithms=["dbscan", "hierarchical-clustering", "gaussian-mixture"]
)

AlgorithmRegistry.register(k_means_metadata)


@router.get("/k-means/info")
async def get_kmeans_info() -> Dict[str, Any]:
    """Get K-Means algorithm information and metadata."""
    metadata = AlgorithmRegistry.get("k-means")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["blobs"]
    }


@router.post("/k-means/train")
async def train_kmeans(params: KMeansParameters) -> KMeansResponse:
    """Train K-Means clustering model.

    Args:
        params: Training parameters including n_clusters, max_iter, etc.

    Returns:
        Training results including metrics and visualization data

    Raises:
        HTTPException: If training fails
    """
    try:
        logger.info(f"Training K-Means with parameters: {params.model_dump()}")
        model = KMeansModel()
        response = model.train(params)

        if not response.success:
            logger.error(f"K-Means training failed: {response.error}")
            raise HTTPException(status_code=400, detail=response.error)

        logger.info(
            f"K-Means training completed in {response.execution_time_ms:.2f}ms "
            f"with inertia={response.metrics.get('inertia', 0):.2f}"
        )
        return response

    except Exception as e:
        logger.error(f"Unexpected error during K-Means training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Register DBSCAN metadata
dbscan_metadata = AlgorithmMetadata(
    id="dbscan",
    name="DBSCAN",
    slug="dbscan",
    category=AlgorithmCategory.ML,
    description="Density-based clustering that can find arbitrarily shaped clusters and outliers",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["unsupervised", "clustering", "density-based"],
    use_cases=[
        "Anomaly detection",
        "Spatial data clustering",
        "Pattern recognition"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*log(n)) with indexing",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="eps",
            label="Epsilon (ε)",
            type="number",
            default=0.5,
            min=0.1,
            max=2.0,
            step=0.1,
            description="Maximum distance between samples in a neighborhood"
        ),
        AlgorithmParameter(
            name="min_samples",
            label="Minimum Samples",
            type="number",
            default=5,
            min=2,
            max=20,
            step=1,
            description="Minimum samples in neighborhood to form core point"
        ),
        AlgorithmParameter(
            name="metric",
            label="Distance Metric",
            type="select",
            default="euclidean",
            options=[
                {"label": "Euclidean", "value": "euclidean"},
                {"label": "Manhattan", "value": "manhattan"}
            ],
            description="Distance metric to use for neighborhood calculations"
        ),
        AlgorithmParameter(
            name="random_state",
            label="Random Seed",
            type="number",
            default=42,
            min=0,
            max=100,
            step=1,
            description="Random seed for reproducibility"
        ),
        AlgorithmParameter(
            name="n_samples",
            label="Number of Samples",
            type="number",
            default=300,
            min=100,
            max=1000,
            step=50,
            description="Number of data points to generate"
        ),
        AlgorithmParameter(
            name="noise",
            label="Noise Level",
            type="number",
            default=0.1,
            min=0.0,
            max=0.5,
            step=0.05,
            description="Standard deviation of Gaussian noise for dataset"
        )
    ],
    dataset_name="moons",
    visualization_type="scatter_plot",
    theory=(
        "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a "
        "density-based clustering algorithm that groups together points that are closely "
        "packed together, while marking points in low-density regions as outliers. "
        "It works by: (1) Defining core points as those with at least min_samples neighbors "
        "within eps distance, (2) Forming clusters by connecting core points that are within "
        "eps of each other, (3) Assigning border points to clusters, and (4) Marking remaining "
        "points as noise. Unlike K-Means, DBSCAN can discover clusters of arbitrary shape and "
        "automatically determines the number of clusters."
    ),
    pros=[
        "Discovers arbitrarily shaped clusters",
        "Identifies and handles noise/outliers",
        "Doesn't require specifying number of clusters",
        "Robust to outliers",
        "Works well with non-convex clusters"
    ],
    cons=[
        "Sensitive to eps and min_samples parameters",
        "Struggles with varying density clusters",
        "Not suitable for high-dimensional data",
        "Cannot predict cluster for new data points"
    ],
    related_algorithms=["k-means", "hierarchical-clustering", "hdbscan"]
)

AlgorithmRegistry.register(dbscan_metadata)


@router.get("/dbscan/info")
async def get_dbscan_info() -> Dict[str, Any]:
    """Get DBSCAN algorithm information and metadata."""
    metadata = AlgorithmRegistry.get("dbscan")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["moons"]
    }


@router.post("/dbscan/train")
async def train_dbscan(params: DBSCANParameters) -> DBSCANResponse:
    """Train DBSCAN clustering model.

    Args:
        params: Training parameters including eps, min_samples, metric

    Returns:
        Training results including metrics and visualization data

    Raises:
        HTTPException: If training fails
    """
    try:
        logger.info(f"Training DBSCAN with parameters: {params.model_dump()}")
        model = DBSCANModel()
        response = model.train(params)

        if not response.success:
            logger.error(f"DBSCAN training failed: {response.error}")
            raise HTTPException(status_code=400, detail=response.error)

        logger.info(
            f"DBSCAN training completed in {response.execution_time_ms:.2f}ms "
            f"with {response.metrics.get('n_clusters', 0)} clusters and "
            f"{response.metrics.get('n_noise', 0)} noise points"
        )
        return response

    except Exception as e:
        logger.error(f"Unexpected error during DBSCAN training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Register XGBoost metadata
xgboost_metadata = AlgorithmMetadata(
    id="xgboost",
    name="Gradient Boosting (XGBoost)",
    slug="xgboost",
    category=AlgorithmCategory.ML,
    description="Ensemble method that builds trees sequentially to correct errors",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["supervised", "classification", "ensemble", "boosting"],
    use_cases=[
        "Competition winning models",
        "Risk prediction",
        "Ranking problems"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*d*k*depth)",
        space="O(k*n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_estimators",
            label="Number of Estimators",
            type="range",
            default=100,
            min=10,
            max=500,
            step=10,
            description="Number of boosting rounds (trees to build)"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            description="Step size shrinkage used to prevent overfitting"
        ),
        AlgorithmParameter(
            name="max_depth",
            label="Max Depth",
            type="number",
            default=6,
            min=3,
            max=15,
            step=1,
            description="Maximum tree depth for base learners"
        ),
        AlgorithmParameter(
            name="subsample",
            label="Subsample Ratio",
            type="range",
            default=1.0,
            min=0.5,
            max=1.0,
            step=0.1,
            description="Subsample ratio of the training instances"
        )
    ],
    dataset_name="wine",
    visualization_type="feature_importance_learning_curves_confusion_matrix",
    theory=(
        "XGBoost (eXtreme Gradient Boosting) is an optimized distributed gradient boosting library "
        "designed to be highly efficient, flexible and portable. It implements machine learning "
        "algorithms under the Gradient Boosting framework. The algorithm works by building an ensemble "
        "of decision trees sequentially. Each new tree is trained to correct the errors made by the "
        "existing ensemble. Key innovations include regularized learning objective, parallel tree "
        "construction, handling of missing values, and built-in cross-validation."
    ),
    pros=[
        "State-of-the-art performance on structured data",
        "Built-in regularization to prevent overfitting",
        "Handles missing values automatically",
        "Fast training with parallel processing",
        "Feature importance analysis"
    ],
    cons=[
        "Can be sensitive to hyperparameters",
        "Requires careful tuning for optimal performance",
        "Less interpretable than single decision trees",
        "Memory intensive for large datasets",
        "May overfit on small datasets"
    ],
    related_algorithms=["random-forest", "gradient-boosting", "decision-tree"]
)

AlgorithmRegistry.register(xgboost_metadata)


@router.post("/xgboost/train", response_model=XGBoostResponse)
async def train_xgboost(request: XGBoostRequest) -> Dict[str, Any]:
    """Train XGBoost classifier with specified parameters.

    Args:
        request: XGBoost training request with parameters.

    Returns:
        Training results including metrics, predictions, and visualization data.

    Raises:
        HTTPException: If training fails or parameters are invalid.
    """
    try:
        model = XGBoostModel()
        result = model.train(
            n_estimators=request.n_estimators,
            learning_rate=request.learning_rate,
            max_depth=request.max_depth,
            subsample=request.subsample,
            dataset_name=request.dataset_name or "wine",
            normalize=request.normalize
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during training: {str(e)}"
        )


@router.get("/xgboost/info")
async def get_xgboost_info() -> Dict[str, Any]:
    """Get XGBoost algorithm information and metadata.

    Returns:
        Dictionary containing algorithm metadata and available datasets.
    """
    metadata = AlgorithmRegistry.get("xgboost")
    if not metadata:
        raise HTTPException(status_code=404, detail="XGBoost algorithm not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["wine", "iris", "digits"]
    }


# Import Decision Tree
import logging
from algorithms.ml.decision_tree import DecisionTreeModel, DecisionTreeRequest, DecisionTreeResponse

logger = logging.getLogger(__name__)


# Register Decision Tree metadata
decision_tree_metadata = AlgorithmMetadata(
    id="decision-tree",
    name="Decision Tree",
    slug="decision-tree",
    category=AlgorithmCategory.ML,
    description="Tree-based supervised learning algorithm for classification and regression",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["supervised", "classification", "tree-based"],
    use_cases=[
        "Medical diagnosis",
        "Credit risk assessment",
        "Customer behavior prediction"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*log(n)*d)",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="max_depth",
            label="Maximum Depth",
            type="number",
            default=None,
            min=1,
            max=20,
            step=1,
            description="Maximum depth of the decision tree (None for unlimited)"
        ),
        AlgorithmParameter(
            name="min_samples_split",
            label="Min Samples Split",
            type="number",
            default=2,
            min=2,
            max=20,
            step=1,
            description="Minimum number of samples required to split an internal node"
        ),
        AlgorithmParameter(
            name="min_samples_leaf",
            label="Min Samples Leaf",
            type="number",
            default=1,
            min=1,
            max=10,
            step=1,
            description="Minimum number of samples required to be at a leaf node"
        ),
        AlgorithmParameter(
            name="criterion",
            label="Split Criterion",
            type="select",
            default="gini",
            options=[
                {"label": "Gini Impurity", "value": "gini"},
                {"label": "Entropy (Information Gain)", "value": "entropy"}
            ],
            description="Function to measure the quality of a split"
        )
    ],
    dataset_name="iris",
    visualization_type="tree_confusion_matrix",
    theory=(
        "Decision Trees are hierarchical models that make predictions by learning simple "
        "decision rules from data features. The algorithm recursively partitions the feature "
        "space into regions, forming a tree structure where internal nodes represent decision "
        "rules based on feature values, branches represent the outcome of those decisions, and "
        "leaf nodes represent the final prediction. The tree is built top-down by selecting the "
        "best split at each node based on a criterion (Gini impurity or entropy) that measures "
        "the homogeneity of the resulting subsets."
    ),
    pros=[
        "Easy to understand and interpret",
        "Requires little data preprocessing",
        "Handles both numerical and categorical data",
        "Non-parametric (no assumptions about data distribution)",
        "Can capture non-linear relationships"
    ],
    cons=[
        "Prone to overfitting, especially with deep trees",
        "Can be unstable (small data changes affect structure)",
        "Biased toward features with more levels",
        "Not optimal for extrapolation",
        "Can create biased trees with imbalanced datasets"
    ],
    related_algorithms=["random-forest", "gradient-boosting", "xgboost"]
)

AlgorithmRegistry.register(decision_tree_metadata)


@router.post("/decision-tree/train", response_model=DecisionTreeResponse)
async def train_decision_tree(request: DecisionTreeRequest) -> DecisionTreeResponse:
    """Train a Decision Tree classifier.

    Args:
        request: Training parameters including max_depth, min_samples_split,
                min_samples_leaf, and criterion.

    Returns:
        Training results including metrics, predictions, and tree visualization data.

    Raises:
        HTTPException: If training fails or invalid parameters are provided.
    """
    try:
        logger.info(f"Training Decision Tree with parameters: {request.dict()}")

        # Create and train model
        model = DecisionTreeModel()
        result = model.train(
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            min_samples_leaf=request.min_samples_leaf,
            criterion=request.criterion,
            dataset_name=request.dataset_name,
            random_state=request.random_state
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        logger.info(
            f"Decision Tree training completed in {result['execution_time_ms']:.2f}ms "
            f"with accuracy: {result['metrics']['accuracy']:.4f}"
        )

        return DecisionTreeResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/decision-tree/info")
async def get_decision_tree_info() -> Dict[str, Any]:
    """Get Decision Tree algorithm information and metadata.

    Returns:
        Algorithm metadata including parameters, complexity, theory, and use cases.
    """
    metadata = AlgorithmRegistry.get("decision-tree")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["iris", "wine", "digits"]
    }


# Import and register Logistic Regression
from algorithms.ml.logistic_regression import LogisticRegressionModel, prepare_data, get_metadata as get_lr_metadata
from algorithms.ml.logistic_regression.data import prepare_visualization_data
from algorithms.ml.logistic_regression.schema import register_algorithm as register_lr
from utils.response_schemas import AlgorithmInfoResponse, TrainingRequest, TrainingResponse

# Import Hierarchical Clustering
from algorithms.ml.hierarchical_clustering import (
    HierarchicalClusteringModel,
    HierarchicalClusteringRequest,
    HierarchicalClusteringResponse
)
from algorithms.ml.hierarchical_clustering.data import get_sample_data
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Register Logistic Regression
register_lr()


@router.get("/logistic-regression/info", response_model=AlgorithmInfoResponse)
async def get_logistic_regression_info():
    """Get metadata and information about Logistic Regression algorithm.

    Returns:
        Algorithm metadata including parameters, complexity, and theory
    """
    metadata = get_lr_metadata()
    return AlgorithmInfoResponse(
        metadata=metadata.model_dump(),
        available_datasets=['iris', 'wine', 'digits']
    )


@router.post("/logistic-regression/train", response_model=TrainingResponse)
async def train_logistic_regression(request: TrainingRequest):
    """Train a Logistic Regression model with specified parameters.

    Args:
        request: Training request containing parameters and dataset configuration

    Returns:
        Training results including metrics, predictions, and visualization data

    Raises:
        HTTPException: If training fails or parameters are invalid
    """
    try:
        logger.info(f"Training Logistic Regression with parameters: {request.parameters}")

        # Extract parameters with defaults
        params = request.parameters
        C = params.get('C', 1.0)
        penalty = params.get('penalty', 'l2')
        max_iter = params.get('max_iter', 100)
        solver = params.get('solver', 'lbfgs')
        dataset_name = request.dataset_name or 'iris'
        normalize = request.normalize

        # Validate parameters
        if C <= 0:
            raise ValueError("C must be positive")
        if penalty not in ['l1', 'l2']:
            raise ValueError("penalty must be 'l1' or 'l2'")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if solver not in ['lbfgs', 'liblinear', 'saga']:
            raise ValueError("solver must be 'lbfgs', 'liblinear', or 'saga'")

        # Prepare data
        X_train, X_test, y_train, y_test, metadata = prepare_data(
            dataset_name=dataset_name,
            normalize=normalize
        )

        # Initialize and train model
        model = LogisticRegressionModel(
            C=C,
            penalty=penalty,
            max_iter=max_iter,
            solver=solver
        )

        results = model.train(X_train, y_train, X_test, y_test)

        # Prepare visualization data
        viz_data = prepare_visualization_data(
            X_test=X_test,
            y_test=y_test,
            y_pred=results['predictions'],
            probabilities=results['probabilities'],
            classes=results['classes'],
            feature_names=metadata.get('feature_names', [])
        )

        # Combine results
        return TrainingResponse(
            success=True,
            metrics=results['metrics'],
            predictions=results['predictions'],
            visualization_data=viz_data,
            execution_time_ms=results['execution_time_ms'],
            parameters_used={
                'C': C,
                'penalty': penalty,
                'max_iter': max_iter,
                'solver': solver,
                'dataset': dataset_name,
                'normalize': normalize
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Register Hierarchical Clustering metadata
hierarchical_clustering_metadata = AlgorithmMetadata(
    id="hierarchical-clustering",
    name="Hierarchical Clustering",
    slug="hierarchical-clustering",
    category=AlgorithmCategory.ML,
    description="Clustering method that builds a hierarchy of clusters using linkage",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["unsupervised", "clustering", "hierarchical"],
    use_cases=[
        "Taxonomy creation",
        "Gene sequence analysis",
        "Social network analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(n³)",
        space="O(n²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_clusters",
            label="Number of Clusters",
            type="number",
            default=3,
            min=2,
            max=10,
            step=1,
            description="Number of clusters to find"
        ),
        AlgorithmParameter(
            name="linkage",
            label="Linkage Method",
            type="select",
            default="ward",
            options=[
                {"value": "ward", "label": "Ward"},
                {"value": "complete", "label": "Complete"},
                {"value": "average", "label": "Average"},
                {"value": "single", "label": "Single"}
            ],
            description="Linkage criterion for computing distances between clusters"
        ),
        AlgorithmParameter(
            name="affinity",
            label="Distance Metric",
            type="select",
            default="euclidean",
            options=[
                {"value": "euclidean", "label": "Euclidean"},
                {"value": "manhattan", "label": "Manhattan"},
                {"value": "cosine", "label": "Cosine"}
            ],
            description="Metric used to compute the distances between samples"
        )
    ],
    dataset_name="blobs",
    visualization_type="dendrogram_scatter",
    theory="Hierarchical clustering builds a tree of clusters by either iteratively merging (agglomerative) or splitting (divisive) clusters. The agglomerative approach starts with each point as its own cluster and merges the closest pairs until only one cluster remains. The linkage criterion determines how the distance between clusters is computed.",
    pros=[
        "No need to specify number of clusters in advance (can cut dendrogram at any level)",
        "Creates a dendrogram that shows cluster hierarchy",
        "Works well with non-spherical clusters",
        "Deterministic results"
    ],
    cons=[
        "High computational complexity O(n³) makes it slow for large datasets",
        "High memory requirements O(n²)",
        "Sensitive to noise and outliers",
        "Cannot undo previous merging steps"
    ],
    related_algorithms=["k-means", "dbscan", "spectral-clustering"]
)

AlgorithmRegistry.register(hierarchical_clustering_metadata)


@router.post("/hierarchical-clustering/train", response_model=HierarchicalClusteringResponse)
async def train_hierarchical_clustering(request: HierarchicalClusteringRequest):
    """Train Hierarchical Clustering model and return results.

    Args:
        request: Training parameters including n_clusters, linkage, and affinity

    Returns:
        Training results including metrics, cluster assignments, and visualization data

    Raises:
        HTTPException: If training fails or invalid parameters provided
    """
    try:
        # Validate ward linkage with euclidean affinity
        if request.linkage == "ward" and request.affinity != "euclidean":
            raise HTTPException(
                status_code=400,
                detail="Ward linkage requires euclidean affinity"
            )

        # Load data
        data = get_sample_data(
            n_samples=request.n_samples,
            n_clusters=request.n_clusters
        )
        X = data['X']

        # Create and train model
        model = HierarchicalClusteringModel(
            n_clusters=request.n_clusters,
            linkage=request.linkage,
            affinity=request.affinity
        )

        training_result = model.train(X)

        # Get cluster assignments
        labels = model.get_cluster_labels()

        # Evaluate clustering
        metrics = model.evaluate(X)

        # Get dendrogram data
        dendrogram_data = model.get_dendrogram_data()

        # Get model info
        model_info = model.get_model_info()

        # Prepare visualization data
        visualization_data = {
            "scatter": {
                "X": X.tolist(),
                "labels": labels.tolist(),
                "n_clusters": request.n_clusters
            },
            "dendrogram": dendrogram_data
        }

        return HierarchicalClusteringResponse(
            metrics=metrics,
            predictions=labels.tolist(),
            visualization_data=visualization_data,
            execution_time_ms=training_result['training_time_ms'],
            model_info=model_info
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/hierarchical-clustering/info")
async def get_hierarchical_clustering_info():
    """Get metadata information about Hierarchical Clustering algorithm.

    Returns:
        Algorithm metadata including parameters, complexity, use cases, etc.
    """
    metadata = AlgorithmRegistry.get("hierarchical-clustering")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    return metadata.model_dump()


# Register KNN metadata
knn_metadata = AlgorithmMetadata(
    id="knn",
    name="K-Nearest Neighbors",
    slug="knn",
    category=AlgorithmCategory.ML,
    description="Instance-based learning algorithm that classifies based on nearest neighbors",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["supervised", "classification", "instance-based"],
    use_cases=[
        "Recommendation systems",
        "Pattern recognition",
        "Anomaly detection"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*d)",
        space="O(n*d)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_neighbors",
            label="Number of Neighbors (K)",
            type="range",
            default=5,
            min=1,
            max=20,
            step=1,
            description="Number of nearest neighbors to consider for classification"
        ),
        AlgorithmParameter(
            name="weights",
            label="Weight Function",
            type="select",
            default="uniform",
            options=[
                {"label": "Uniform (equal weights)", "value": "uniform"},
                {"label": "Distance (weighted by inverse distance)", "value": "distance"}
            ],
            description="How to weight neighbors in prediction"
        ),
        AlgorithmParameter(
            name="metric",
            label="Distance Metric",
            type="select",
            default="euclidean",
            options=[
                {"label": "Euclidean", "value": "euclidean"},
                {"label": "Manhattan", "value": "manhattan"},
                {"label": "Minkowski", "value": "minkowski"}
            ],
            description="Distance metric for finding nearest neighbors"
        ),
        AlgorithmParameter(
            name="p",
            label="Minkowski Power Parameter",
            type="range",
            default=2,
            min=1,
            max=5,
            step=1,
            description="Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)"
        )
    ],
    dataset_name="iris",
    visualization_type="scatter_confusion",
    theory=(
        "K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that makes "
        "predictions based on the K nearest training examples in the feature space. It's a "
        "non-parametric method, meaning it doesn't make assumptions about the underlying data "
        "distribution.\n\n"
        "The algorithm works by:\n"
        "1. Calculating the distance between the query point and all training points\n"
        "2. Finding the K nearest neighbors based on the chosen distance metric\n"
        "3. Classifying the query point based on the majority vote of its K neighbors\n\n"
        "KNN can use different distance metrics (Euclidean, Manhattan, Minkowski) and "
        "weighting schemes (uniform or distance-weighted) to make predictions."
    ),
    pros=[
        "Simple and intuitive algorithm",
        "No training phase (lazy learning)",
        "Naturally handles multi-class problems",
        "Can capture complex decision boundaries",
        "No assumptions about data distribution"
    ],
    cons=[
        "Slow prediction time with large datasets",
        "Sensitive to feature scaling",
        "Sensitive to irrelevant features",
        "Requires storage of all training data",
        "Performance depends heavily on K value"
    ],
    related_algorithms=["decision-tree", "svm", "naive-bayes"]
)

AlgorithmRegistry.register(knn_metadata)


@router.post("/knn/train", response_model=KNNResponse)
async def train_knn_endpoint(request: KNNRequest):
    """Train K-Nearest Neighbors classifier on Iris dataset.

    This endpoint trains a KNN model with the specified parameters,
    evaluates it on test data, and returns metrics and visualization data.

    Args:
        request: KNNRequest with algorithm parameters

    Returns:
        KNNResponse with training results, metrics, and visualization data

    Raises:
        HTTPException: If training fails
    """
    try:
        logger.info(f"Training KNN with parameters: {request.model_dump()}")
        response = train_knn(request)
        logger.info(
            f"KNN training completed in {response.execution_time_ms:.2f}ms "
            f"with accuracy: {response.metrics.accuracy:.4f}"
        )
        return response
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/knn/info")
async def get_knn_info():
    """Get metadata information about the KNN algorithm.

    Returns algorithm description, parameters, complexity, use cases,
    and other metadata for display in the frontend.

    Returns:
        Dictionary with algorithm metadata

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("knn")
    if metadata is None:
        raise HTTPException(status_code=404, detail="KNN algorithm metadata not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["iris"]
    }


# Register GMM metadata
gmm_metadata = AlgorithmMetadata(
    id="gmm",
    name="Gaussian Mixture Model",
    slug="gmm",
    category=AlgorithmCategory.ML,
    description="Probabilistic clustering assuming data is mixture of Gaussians",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["unsupervised", "clustering", "probabilistic", "generative"],
    use_cases=[
        "Density estimation",
        "Anomaly detection",
        "Image segmentation",
        "Customer segmentation",
        "Speech recognition"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*k*d²*i)",
        space="O(k*d²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_components",
            label="Number of Components",
            type="number",
            default=3,
            min=2,
            max=10,
            step=1,
            description="Number of Gaussian components (clusters) to fit"
        ),
        AlgorithmParameter(
            name="covariance_type",
            label="Covariance Type",
            type="select",
            default="full",
            options=[
                {"value": "full", "label": "Full - Each component has its own covariance matrix"},
                {"value": "tied", "label": "Tied - All components share the same covariance"},
                {"value": "diag", "label": "Diagonal - Only diagonal elements"},
                {"value": "spherical", "label": "Spherical - Single variance per component"}
            ],
            description="Type of covariance parameters to use"
        ),
        AlgorithmParameter(
            name="max_iter",
            label="Max Iterations",
            type="number",
            default=100,
            min=10,
            max=500,
            step=10,
            description="Maximum number of EM algorithm iterations"
        ),
        AlgorithmParameter(
            name="n_samples",
            label="Number of Samples",
            type="number",
            default=300,
            min=100,
            max=1000,
            step=50,
            description="Number of samples in the generated dataset"
        )
    ],
    dataset_name="blobs",
    visualization_type="scatter_plot",
    theory=(
        "Gaussian Mixture Model (GMM) is a probabilistic model that assumes data points "
        "are generated from a mixture of a finite number of Gaussian distributions with "
        "unknown parameters. GMM uses the Expectation-Maximization (EM) algorithm to find "
        "maximum likelihood estimates of the parameters. Unlike K-Means, GMM provides soft "
        "clustering where each point has a probability of belonging to each cluster."
    ),
    pros=[
        "Provides probabilistic cluster assignments",
        "Can model elliptical clusters with different shapes and orientations",
        "Naturally handles uncertainty in cluster membership",
        "Can be used for density estimation and anomaly detection",
        "Flexible covariance structures for different data patterns"
    ],
    cons=[
        "Computationally expensive for large datasets",
        "Sensitive to initialization",
        "Requires specification of number of components",
        "May converge to local optima",
        "Assumes Gaussian distribution of clusters"
    ],
    related_algorithms=["k-means", "dbscan", "hierarchical-clustering"]
)

AlgorithmRegistry.register(gmm_metadata)


@router.post("/gmm/train", response_model=GMMResponse)
async def train_gmm(request: GMMRequest) -> GMMResponse:
    """Train a Gaussian Mixture Model on blob dataset.

    This endpoint trains a GMM with the specified parameters on synthetic
    blob data and returns clustering results, metrics, and visualization data.

    Args:
        request: GMM training parameters including n_components, covariance_type,
                max_iter, n_samples, and random_state.

    Returns:
        GMMResponse containing metrics, predictions, probabilities,
        visualization data, parameters, and execution time.

    Raises:
        HTTPException: If training fails or parameters are invalid.
    """
    try:
        start_time = time.time()

        # Load data
        data = load_blobs_data(
            n_samples=request.n_samples,
            centers=request.n_components,
            random_state=request.random_state
        )
        X = data['X']

        # Initialize and train model
        model = GaussianMixtureModelClass(
            n_components=request.n_components,
            covariance_type=request.covariance_type,
            max_iter=request.max_iter,
            random_state=request.random_state
        )

        # Fit model
        fit_info = model.fit(X)

        # Get predictions and probabilities
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Evaluate model
        metrics = model.evaluate(X)

        # Get learned parameters
        parameters = model.get_parameters()

        # Generate contour data for visualization
        contour_data = model.generate_contour_data(X, n_points=50)

        # Prepare visualization data
        visualization_data = {
            "x": X[:, 0].tolist(),
            "y": X[:, 1].tolist(),
            "labels": predictions.tolist(),
            "contour": contour_data,
            "means": parameters["means"],
            "true_labels": data['y'].tolist()  # Include true labels for comparison
        }

        # Get model info
        model_info = model.get_model_info()

        # Calculate total execution time
        execution_time_ms = (time.time() - start_time) * 1000

        return GMMResponse(
            metrics=metrics,
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            visualization_data=visualization_data,
            parameters=parameters,
            execution_time_ms=execution_time_ms,
            model_info=model_info
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/gmm/info")
async def get_gmm_info() -> Dict[str, Any]:
    """Get information about the Gaussian Mixture Model algorithm.

    Returns:
        Dictionary containing algorithm metadata including parameters,
        complexity, use cases, and theory.
    """
    metadata = AlgorithmRegistry.get('gmm')
    if not metadata:
        raise HTTPException(status_code=404, detail="GMM algorithm not found")

    dataset_info = get_gmm_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Naive Bayes metadata
naive_bayes_metadata = AlgorithmMetadata(
    id="naive-bayes",
    name="Naive Bayes",
    slug="naive-bayes",
    category=AlgorithmCategory.ML,
    description="Probabilistic classifier based on Bayes' theorem with independence assumption",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["supervised", "classification", "probabilistic"],
    use_cases=[
        "Spam filtering",
        "Document classification",
        "Sentiment analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*d)",
        space="O(d*c)"
    ),
    parameters=[
        AlgorithmParameter(
            name="var_smoothing",
            label="Variance Smoothing",
            type="range",
            default=1e-9,
            min=1e-10,
            max=1e-8,
            step=1e-10,
            description="Portion of largest variance added to variances for stability"
        ),
        AlgorithmParameter(
            name="normalize",
            label="Normalize Features",
            type="boolean",
            default=True,
            description="Whether to normalize features before training"
        )
    ],
    dataset_name="iris",
    visualization_type="probability_distribution,confusion_matrix",
    theory=(
        "Naive Bayes is a probabilistic classifier based on applying Bayes' theorem "
        "with strong (naive) independence assumptions between features. Despite the "
        "simplifying assumption that features are conditionally independent given the "
        "class, Naive Bayes classifiers often perform surprisingly well in practice. "
        "The Gaussian variant assumes continuous features follow a normal distribution "
        "within each class."
    ),
    pros=[
        "Fast training and prediction",
        "Works well with high-dimensional data",
        "Requires small amount of training data",
        "Not sensitive to irrelevant features",
        "Provides probability estimates"
    ],
    cons=[
        "Assumes feature independence (rarely true in practice)",
        "Sensitive to feature scaling for Gaussian variant",
        "Can be outperformed by more complex models",
        "May produce biased probability estimates"
    ],
    related_algorithms=["logistic-regression", "k-nearest-neighbors", "decision-tree"]
)

AlgorithmRegistry.register(naive_bayes_metadata)


@router.post("/naive-bayes/train", response_model=NaiveBayesResponse)
async def train_naive_bayes(request: NaiveBayesRequest) -> NaiveBayesResponse:
    """Train Naive Bayes classifier on specified dataset.

    This endpoint trains a Gaussian Naive Bayes classifier using the provided
    parameters and returns comprehensive results including metrics, predictions,
    and visualization data.

    Args:
        request: Training configuration with parameters and dataset selection

    Returns:
        Training results with metrics, predictions, and visualization data

    Raises:
        HTTPException: If training fails or invalid parameters provided
    """
    try:
        # Train model using the from_dataset class method
        results = NaiveBayesModel.from_dataset(
            dataset_name=request.dataset_name,
            var_smoothing=request.var_smoothing,
            priors=request.priors,
            normalize=request.normalize
        )

        return NaiveBayesResponse(
            success=True,
            metrics=results['metrics'],
            predictions=results['predictions'],
            probabilities=results['probabilities'],
            feature_contributions=results.get('feature_contributions'),
            visualization_data=results['visualization_data'],
            execution_time_ms=results['execution_time_ms'],
            parameters_used=results['parameters_used']
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/naive-bayes/info")
async def get_naive_bayes_info() -> Dict[str, Any]:
    """Get metadata and configuration information for Naive Bayes algorithm.

    Returns:
        Algorithm metadata including parameters, complexity, theory, and
        available datasets

    Raises:
        HTTPException: If algorithm metadata not found
    """
    metadata = AlgorithmRegistry.get("naive-bayes")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Algorithm metadata not found"
        )

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": get_nb_datasets()
    }


# Import and register SVM
from algorithms.ml.svm import SVMModel, SVMRequest, SVMResponse
from algorithms.ml.svm.data import load_iris_data as load_svm_iris_data


# Register SVM metadata
svm_metadata = AlgorithmMetadata(
    id="svm",
    name="Support Vector Machine",
    slug="svm",
    category=AlgorithmCategory.ML,
    description="Supervised learning algorithm that finds optimal hyperplane for classification",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["supervised", "classification", "kernel-methods"],
    use_cases=[
        "Image classification",
        "Text categorization",
        "Bioinformatics",
    ],
    complexity=AlgorithmComplexity(
        time="O(n²) to O(n³)",
        space="O(n²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="C",
            label="Regularization Parameter",
            type="range",
            default=1.0,
            min=0.1,
            max=10.0,
            step=0.1,
            description="Controls the trade-off between smooth decision boundary and classifying training points correctly"
        ),
        AlgorithmParameter(
            name="kernel",
            label="Kernel Type",
            type="select",
            default="rbf",
            options=[
                {"value": "linear", "label": "Linear"},
                {"value": "poly", "label": "Polynomial"},
                {"value": "rbf", "label": "RBF (Radial Basis Function)"},
                {"value": "sigmoid", "label": "Sigmoid"}
            ],
            description="Kernel function used to transform the data into higher dimensions"
        ),
        AlgorithmParameter(
            name="gamma",
            label="Kernel Coefficient",
            type="select",
            default="scale",
            options=[
                {"value": "scale", "label": "Scale (1 / (n_features * X.var()))"},
                {"value": "auto", "label": "Auto (1 / n_features)"}
            ],
            description="Kernel coefficient for RBF, polynomial and sigmoid kernels"
        ),
        AlgorithmParameter(
            name="degree",
            label="Polynomial Degree",
            type="range",
            default=3,
            min=2,
            max=5,
            step=1,
            description="Degree of polynomial kernel function (ignored by other kernels)"
        ),
    ],
    dataset_name="iris",
    visualization_type="scatter_plot",
    theory=(
        "Support Vector Machine (SVM) is a powerful supervised learning algorithm that finds the optimal hyperplane "
        "to separate different classes in the feature space. The key idea is to maximize the margin (distance) between "
        "the hyperplane and the nearest data points from each class, called support vectors. SVMs can handle both linear "
        "and non-linear classification problems through the use of kernel functions that transform the input data into "
        "higher-dimensional spaces where linear separation becomes possible."
    ),
    pros=[
        "Effective in high-dimensional spaces",
        "Memory efficient (only uses support vectors)",
        "Versatile through different kernel functions",
        "Works well with clear margin of separation"
    ],
    cons=[
        "Not suitable for large datasets (training time scales poorly)",
        "Sensitive to feature scaling",
        "Requires careful kernel selection and parameter tuning",
        "Doesn't directly provide probability estimates"
    ],
    related_algorithms=["logistic-regression", "random-forest", "k-nearest-neighbors"]
)

AlgorithmRegistry.register(svm_metadata)


@router.post("/svm/train", response_model=SVMResponse)
async def train_svm(request: SVMRequest):
    """Train a Support Vector Machine model on the Iris dataset.

    This endpoint trains an SVM classifier with the specified parameters
    and returns performance metrics along with visualization data.

    Args:
        request: SVM training configuration parameters

    Returns:
        SVMResponse containing metrics, model info, and visualization data

    Raises:
        HTTPException: If training fails or invalid parameters provided
    """
    try:
        start_time = time.time()

        # Load dataset
        data = load_svm_iris_data()
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        # Initialize and train model
        model = SVMModel(
            C=request.C,
            kernel=request.kernel,
            gamma=request.gamma,
            degree=request.degree
        )

        train_info = model.train(X_train, y_train)

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)

        # Get predictions for visualization
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Get support vectors
        support_vector_info = model.get_support_vectors()
        support_vector_indices = support_vector_info['support_vector_indices']

        # Prepare training data for visualization (using first 2 features)
        training_data = []
        for i, (features, label, pred) in enumerate(zip(X_train, y_train, y_train_pred)):
            is_support_vector = i in support_vector_indices
            training_data.append({
                'x': float(features[0]),  # First feature (sepal length)
                'y': float(features[1]),  # Second feature (sepal width)
                'label': int(label),
                'predicted': int(pred),
                'is_support_vector': is_support_vector,
                'type': 'train'
            })

        # Prepare test data for visualization
        test_data = []
        for features, label, pred in zip(X_test, y_test, y_test_pred):
            test_data.append({
                'x': float(features[0]),
                'y': float(features[1]),
                'label': int(label),
                'predicted': int(pred),
                'is_support_vector': False,
                'type': 'test'
            })

        # Prepare support vectors for visualization
        support_vectors_viz = []
        for sv in support_vector_info['support_vectors']:
            support_vectors_viz.append({
                'x': float(sv[0]),
                'y': float(sv[1]),
                'is_support_vector': True
            })

        # Get decision boundary data (for 2D visualization)
        try:
            decision_boundary = model.get_decision_boundary_data(X_train, resolution=50)
        except Exception as e:
            # If decision boundary generation fails, continue without it
            decision_boundary = None

        # Calculate total execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Get model info
        model_info = model.get_model_info()

        # Build response
        response = SVMResponse(
            metrics={
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "confusion_matrix": metrics["confusion_matrix"]
            },
            model_info={
                "C": model_info["C"],
                "kernel": model_info["kernel"],
                "gamma": model_info["gamma"],
                "degree": model_info["degree"],
                "n_support_vectors": train_info["n_support_vectors"],
                "support_vectors_per_class": train_info["support_vectors_per_class"],
                "n_features": train_info["n_features"],
                "n_classes": train_info["n_classes"],
                "classes": model_info["classes"]
            },
            visualization_data={
                "training_data": training_data,
                "test_data": test_data,
                "support_vectors": support_vectors_viz,
                "decision_boundary": decision_boundary,
                "feature_names": data['feature_names'][:2],  # Only first 2 for 2D viz
                "target_names": data['target_names']
            },
            execution_time_ms=execution_time_ms
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/svm/info")
async def get_svm_info():
    """Get SVM algorithm metadata and information.

    Returns detailed information about the SVM algorithm including
    parameters, complexity, use cases, and educational content.

    Returns:
        Algorithm metadata dictionary
    """
    metadata = AlgorithmRegistry.get("svm")
    if metadata is None:
        raise HTTPException(status_code=404, detail="SVM algorithm metadata not found")
    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["iris"]
    }


# Import and register PCA
from algorithms.ml.pca import PCAModel, PCARequest, PCAResponse
from algorithms.ml.pca.data import load_digits_data as load_pca_digits_data


# Register PCA metadata
pca_metadata = AlgorithmMetadata(
    id="pca",
    name="Principal Component Analysis",
    slug="pca",
    category=AlgorithmCategory.ML,
    description="Dimensionality reduction technique that finds principal components",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["unsupervised", "dimensionality-reduction", "linear"],
    use_cases=[
        "Data visualization",
        "Noise reduction",
        "Feature extraction"
    ],
    complexity=AlgorithmComplexity(
        time="O(min(n²*d, d²*n))",
        space="O(n*d)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_components",
            label="Number of Components",
            type="range",
            default=2,
            min=2,
            max=10,
            step=1,
            description="Number of principal components to compute"
        ),
        AlgorithmParameter(
            name="whiten",
            label="Whiten Components",
            type="boolean",
            default=False,
            description="Whether to whiten components for unit variance"
        ),
    ],
    dataset_name="digits",
    visualization_type="scatter_plot",
    theory="PCA is a dimensionality reduction technique that transforms data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they explain. It finds orthogonal directions of maximum variance in high-dimensional data.",
    pros=[
        "Reduces data dimensionality while preserving variance",
        "Removes correlated features",
        "Fast computation for moderate datasets",
        "Interpretable principal components"
    ],
    cons=[
        "Assumes linear relationships",
        "Sensitive to data scaling",
        "May lose important information in lower components",
        "Principal components may be hard to interpret"
    ],
    related_algorithms=["t-sne", "lda", "autoencoder"]
)

AlgorithmRegistry.register(pca_metadata)


@router.get("/pca/info")
async def get_pca_info() -> Dict[str, Any]:
    """Get PCA algorithm information and metadata."""
    metadata = AlgorithmRegistry.get("pca")
    if not metadata:
        raise HTTPException(status_code=404, detail="PCA algorithm not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["digits"]
    }


@router.post("/pca/train", response_model=PCAResponse)
async def train_pca(request: PCARequest) -> PCAResponse:
    """Train PCA model and perform dimensionality reduction.

    This endpoint loads the digits dataset, applies PCA for dimensionality
    reduction, and returns the transformed data along with explained variance.

    Args:
        request: PCA training request with n_components and whiten parameters

    Returns:
        PCAResponse containing transformed data and visualization data
    """
    try:
        logger.info(f"Training PCA with parameters: {request.dict()}")
        start_time = time.time()

        # Load digits dataset
        data = load_pca_digits_data()
        X_train = data['X_train']
        y_train = data['y_train']

        # Initialize and fit PCA model
        model = PCAModel(
            n_components=request.n_components,
            whiten=request.whiten
        )

        # Fit and transform the data
        result = model.fit_transform(X_train)

        # Get model info
        model_info = model.get_model_info()

        # Prepare transformed data for response
        transformed_data = result["transformed_data"].tolist()
        labels = y_train.tolist()

        # Prepare visualization data
        # 1. Scatter plot data (first 2 or 3 components)
        scatter_data = []
        for i, point in enumerate(result["transformed_data"]):
            data_point = {
                "pc1": float(point[0]),
                "pc2": float(point[1]) if len(point) > 1 else 0.0,
                "label": int(y_train[i])
            }
            if len(point) > 2:
                data_point["pc3"] = float(point[2])
            scatter_data.append(data_point)

        # 2. Explained variance plot data
        variance_data = [
            {
                "component": f"PC{i+1}",
                "variance": float(var),
                "cumulative": float(result["cumulative_variance_ratio"][i])
            }
            for i, var in enumerate(result["explained_variance_ratio"])
        ]

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"PCA training completed in {execution_time_ms:.2f}ms "
            f"with {request.n_components} components explaining "
            f"{result['cumulative_variance_ratio'][-1]:.2%} variance"
        )

        return PCAResponse(
            success=True,
            transformed_data=transformed_data,
            labels=labels,
            explained_variance=result["explained_variance"],
            explained_variance_ratio=result["explained_variance_ratio"],
            cumulative_variance_ratio=result["cumulative_variance_ratio"],
            visualization_data={
                "scatter_data": scatter_data,
                "variance_data": variance_data,
                "n_samples": len(scatter_data),
                "n_features_original": X_train.shape[1]
            },
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                "n_components": request.n_components,
                "whiten": request.whiten,
                "random_state": request.random_state
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"PCA training error: {str(e)}", exc_info=True)
        return PCAResponse(
            success=False,
            error=f"An error occurred during PCA training: {str(e)}"
        )


# Import and register t-SNE
from algorithms.ml.tsne import TSNEModel, TSNERequest, TSNEResponse
from algorithms.ml.tsne.data import load_digits_data as load_tsne_data, get_dataset_info as get_tsne_dataset_info

# Register t-SNE metadata
tsne_metadata = AlgorithmMetadata(
    id="tsne",
    name="t-SNE",
    slug="tsne",
    category=AlgorithmCategory.ML,
    description="Non-linear dimensionality reduction for visualization of high-dimensional data",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["unsupervised", "dimensionality-reduction", "visualization"],
    use_cases=[
        "High-dimensional data visualization",
        "Cluster discovery and analysis",
        "Feature analysis and exploration",
        "Pattern recognition in complex datasets",
        "Exploratory data analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(n²)",
        space="O(n²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_components",
            label="Embedding Dimensions",
            type="select",
            default=2,
            options=[
                {"label": "2D", "value": 2},
                {"label": "3D", "value": 3}
            ],
            description="Number of dimensions for the embedded space (2D or 3D)"
        ),
        AlgorithmParameter(
            name="perplexity",
            label="Perplexity",
            type="range",
            default=30.0,
            min=5.0,
            max=50.0,
            step=5.0,
            description="Balance between local and global structure (related to number of nearest neighbors)"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=200.0,
            min=10.0,
            max=1000.0,
            step=10.0,
            description="Learning rate for gradient descent optimization"
        ),
        AlgorithmParameter(
            name="n_iter",
            label="Iterations",
            type="range",
            default=1000,
            min=250,
            max=5000,
            step=250,
            description="Number of optimization iterations (minimum 250)"
        )
    ],
    dataset_name="digits",
    visualization_type="scatter_2d_3d",
    theory=(
        "t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality "
        "reduction technique designed specifically for visualizing high-dimensional data. "
        "It works by converting similarities between data points into joint probabilities and "
        "minimizing the Kullback-Leibler divergence between these probabilities in high-dimensional "
        "and low-dimensional spaces. Unlike linear methods like PCA, t-SNE can capture complex "
        "non-linear relationships and reveal cluster structure. The algorithm uses a Student's "
        "t-distribution in the low-dimensional space to alleviate the crowding problem, allowing "
        "dissimilar points to be modeled far apart. The perplexity parameter balances attention "
        "between local and global aspects of the data, effectively determining the number of "
        "nearest neighbors considered."
    ),
    pros=[
        "Excellent for visualizing high-dimensional data",
        "Preserves local neighborhood structure very well",
        "Reveals clusters and patterns not visible with linear methods",
        "Handles non-linear relationships effectively",
        "Widely used and well-understood in practice"
    ],
    cons=[
        "Computationally expensive for large datasets (O(n²) complexity)",
        "Non-deterministic (different runs produce different results)",
        "Cannot be applied to new data (no out-of-sample extension)",
        "Sensitive to hyperparameters (perplexity, learning rate)",
        "Global structure may be distorted to preserve local structure",
        "Distances in embedded space are not meaningful for interpretation"
    ],
    related_algorithms=["pca", "umap", "mds"]
)

AlgorithmRegistry.register(tsne_metadata)


@router.post("/tsne/train", response_model=TSNEResponse)
async def train_tsne(request: TSNERequest) -> TSNEResponse:
    """Apply t-SNE dimensionality reduction for visualization.

    Args:
        request: t-SNE parameters including n_components, perplexity,
                learning_rate, and n_iter.

    Returns:
        t-SNE results including embedded data, visualization data, and metrics.

    Raises:
        HTTPException: If training fails or invalid parameters are provided.
    """
    try:
        logger.info(f"Running t-SNE with parameters: {request.model_dump()}")

        # Load data
        data = load_tsne_data()
        X_train = data['X_train']
        y_train = data['y_train']

        # Create and fit t-SNE model
        model = TSNEModel(
            n_components=request.n_components,
            perplexity=request.perplexity,
            learning_rate=request.learning_rate,
            n_iter=request.n_iter,
            random_state=request.random_state
        )

        # Fit and transform
        result = model.fit_transform(X_train, y_train)

        # Prepare visualization data
        viz_data = model.prepare_visualization_data(
            result['embedded_data'],
            y_train
        )

        # Get model info
        model_info = model.get_model_info()

        execution_time_ms = result['training_time_ms']

        logger.info(
            f"t-SNE completed in {execution_time_ms:.2f}ms "
            f"with KL divergence: {result['kl_divergence']:.4f}"
        )

        return TSNEResponse(
            success=True,
            embedded_data=result['embedded_data'].tolist(),
            labels=result['labels'],
            kl_divergence=result['kl_divergence'],
            visualization_data=viz_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                "n_components": request.n_components,
                "perplexity": request.perplexity,
                "learning_rate": request.learning_rate,
                "n_iter": request.n_iter,
                "random_state": request.random_state
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"t-SNE training error: {str(e)}", exc_info=True)
        return TSNEResponse(
            success=False,
            error=f"An error occurred during t-SNE: {str(e)}"
        )


@router.get("/tsne/info")
async def get_tsne_info() -> Dict[str, Any]:
    """Get t-SNE algorithm information and metadata.

    Returns:
        Algorithm metadata including parameters, complexity, theory, and use cases.
    """
    metadata = AlgorithmRegistry.get("tsne")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    dataset_info = get_tsne_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset_info": dataset_info
    }


# Register AdaBoost metadata
adaboost_metadata = AlgorithmMetadata(
    id="adaboost",
    name="AdaBoost",
    slug="adaboost",
    category=AlgorithmCategory.ML,
    description="Ensemble method that combines weak learners with adaptive weights",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["supervised", "classification", "ensemble", "boosting"],
    use_cases=[
        "Face detection",
        "Text classification",
        "Imbalanced datasets"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*d*k)",
        space="O(k*n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_estimators",
            label="Number of Estimators",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of weak learners to train sequentially"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=1.0,
            min=0.1,
            max=2.0,
            step=0.1,
            description="Weight applied to each classifier (shrinkage parameter)"
        ),
        AlgorithmParameter(
            name="algorithm",
            label="Boosting Algorithm",
            type="select",
            default="SAMME.R",
            options=[
                {"label": "SAMME (discrete)", "value": "SAMME"},
                {"label": "SAMME.R (real, uses probabilities)", "value": "SAMME.R"}
            ],
            description="Boosting algorithm variant to use"
        )
    ],
    dataset_name="iris",
    visualization_type="learning_curve_feature_importance_confusion_matrix",
    theory=(
        "AdaBoost (Adaptive Boosting) is an ensemble learning method that combines multiple "
        "weak learners (typically decision stumps) into a strong classifier. It works iteratively:\n\n"
        "1. Train a weak learner on the training data\n"
        "2. Increase weights for misclassified samples\n"
        "3. Train the next learner focusing on the harder examples\n"
        "4. Combine all weak learners with weighted voting\n\n"
        "Each weak learner is assigned a weight based on its accuracy, and the algorithm "
        "adaptively focuses on samples that are harder to classify. The final prediction is "
        "a weighted majority vote of all weak learners. SAMME.R uses real-valued predictions "
        "and probability estimates, while SAMME uses discrete predictions."
    ),
    pros=[
        "Simple to implement and understand",
        "Works well with weak learners",
        "Adaptive focus on difficult samples",
        "Built-in feature selection",
        "Less prone to overfitting than some other methods"
    ],
    cons=[
        "Sensitive to noisy data and outliers",
        "Can be slow to train sequentially",
        "Vulnerable to uniform noise",
        "Performance depends on weak learner choice",
        "Less effective on very complex problems"
    ],
    related_algorithms=["random-forest", "xgboost", "gradient-boosting"]
)

AlgorithmRegistry.register(adaboost_metadata)


@router.post("/adaboost/train", response_model=AdaBoostResponse)
async def train_adaboost_endpoint(request: AdaBoostRequest):
    """Train AdaBoost classifier on Iris dataset.

    AdaBoost (Adaptive Boosting) sequentially trains weak learners,
    with each learner focusing on samples that previous learners misclassified.
    The final prediction is a weighted combination of all weak learners.

    Args:
        request: AdaBoostRequest with algorithm parameters

    Returns:
        AdaBoostResponse with training results, metrics, and visualization data

    Raises:
        HTTPException: If training fails
    """
    try:
        logger.info(f"Training AdaBoost with parameters: {request.model_dump()}")

        # Prepare data
        X_train, X_test, y_train, y_test, metadata = prepare_adaboost_data(
            dataset_name='iris',
            normalize=True,
            test_size=request.test_size,
            random_state=request.random_state
        )

        # Create and train model
        model = AdaBoostModel(
            n_estimators=request.n_estimators,
            learning_rate=request.learning_rate,
            algorithm=request.algorithm,
            random_state=request.random_state
        )

        # Train and get results
        results = model.train(X_train, y_train, X_test, y_test)

        # Prepare visualization data
        viz_data = prepare_adaboost_viz(
            feature_importance=results['feature_importance'],
            feature_names=metadata['feature_names'],
            confusion_matrix_data=results['confusion_matrix'],
            class_labels=metadata['target_names'],
            learning_curve_data=results['learning_curve']
        )

        logger.info(
            f"AdaBoost training completed in {results['execution_time_ms']:.2f}ms "
            f"with accuracy: {results['metrics']['accuracy']:.4f}"
        )

        # Build response
        return AdaBoostResponse(
            success=True,
            metrics=AdaBoostMetrics(**results['metrics']),
            predictions=results['predictions'],
            actual=y_test.tolist(),
            visualization_data=AdaBoostVisualizationData(**viz_data),
            execution_time_ms=results['execution_time_ms'],
            parameters={
                'n_estimators': request.n_estimators,
                'learning_rate': request.learning_rate,
                'algorithm': request.algorithm,
                'test_size': request.test_size,
                'random_state': request.random_state
            },
            message=f"Successfully trained AdaBoost with {request.n_estimators} estimators"
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/adaboost/info")
async def get_adaboost_info():
    """Get metadata information about the AdaBoost algorithm.

    Returns algorithm description, parameters, complexity, use cases,
    and other metadata for display in the frontend.

    Returns:
        Dictionary with algorithm metadata

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("adaboost")
    if metadata is None:
        raise HTTPException(status_code=404, detail="AdaBoost algorithm metadata not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["iris"]
    }


# Import and register Isolation Forest
from algorithms.ml.isolation_forest import (
    IsolationForestModel,
    IsolationForestParameters,
    IsolationForestResponse,
    AnomalyInfo
)

# Import and register Anomaly Detection
from algorithms.ml.anomaly_detection import (
    AnomalyDetectionModel,
    AnomalyDetectionParameters,
    AnomalyDetectionResponse,
    AnomalyInfo as AnomalyDetectionInfo,
    MethodComparison
)
from algorithms.ml.anomaly_detection.data import get_dataset_info as get_anomaly_dataset_info

# Register Isolation Forest metadata
isolation_forest_metadata = AlgorithmMetadata(
    id="isolation-forest",
    name="Isolation Forest",
    slug="isolation-forest",
    category=AlgorithmCategory.ML,
    description="Anomaly detection algorithm using isolation trees",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["unsupervised", "anomaly-detection", "ensemble"],
    use_cases=[
        "Fraud detection",
        "Network intrusion detection",
        "Quality control",
        "Outlier detection in datasets",
        "System health monitoring"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*log(n))",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="n_estimators",
            label="Number of Trees",
            type="range",
            default=100,
            min=50,
            max=300,
            step=10,
            description="Number of isolation trees in the forest"
        ),
        AlgorithmParameter(
            name="contamination",
            label="Contamination (Expected Outlier Fraction)",
            type="range",
            default=0.1,
            min=0.01,
            max=0.5,
            step=0.01,
            description="Expected proportion of outliers in the dataset"
        ),
        AlgorithmParameter(
            name="max_samples",
            label="Max Samples per Tree",
            type="select",
            default="auto",
            options=[
                {"label": "Auto", "value": "auto"},
                {"label": "100", "value": 100},
                {"label": "256", "value": 256},
                {"label": "512", "value": 512}
            ],
            description="Number of samples to draw to train each tree"
        ),
        AlgorithmParameter(
            name="random_state",
            label="Random Seed",
            type="number",
            default=42,
            min=0,
            max=100,
            step=1,
            description="Random seed for reproducibility"
        ),
        AlgorithmParameter(
            name="n_samples",
            label="Number of Samples",
            type="number",
            default=300,
            min=100,
            max=1000,
            step=50,
            description="Total number of data points to generate"
        ),
        AlgorithmParameter(
            name="n_outliers_ratio",
            label="Outlier Injection Ratio",
            type="range",
            default=0.1,
            min=0.01,
            max=0.3,
            step=0.01,
            description="Ratio of outliers to inject into the dataset"
        )
    ],
    dataset_name="blobs_with_outliers",
    visualization_type="scatter_plot_anomaly_scores",
    theory=(
        "Isolation Forest is an unsupervised anomaly detection algorithm that works by isolating "
        "anomalies rather than profiling normal points. It builds an ensemble of isolation trees "
        "where anomalies are points that have short average path lengths (are easier to isolate). "
        "The algorithm works by:\n\n"
        "1. Randomly selecting a feature and a split value between min and max\n"
        "2. Recursively partitioning the data by creating isolation trees\n"
        "3. Measuring the path length from root to leaf for each point\n"
        "4. Anomalies have shorter average path lengths across all trees\n\n"
        "The key insight is that anomalies are 'few and different', making them easier to isolate "
        "than normal points. The anomaly score is computed based on the average path length, "
        "normalized by the expected path length of unsuccessful searches in a Binary Search Tree. "
        "Unlike distance-based or density-based methods, Isolation Forest has linear time complexity "
        "and works well in high-dimensional spaces."
    ),
    pros=[
        "Efficient for large datasets with linear time complexity O(n*log(n))",
        "Handles high-dimensional data well",
        "No need for labeled anomaly data (unsupervised)",
        "Provides anomaly scores for ranking",
        "Robust to irrelevant features",
        "Low memory requirements"
    ],
    cons=[
        "Less effective for datasets with many normal points in low-density regions",
        "Contamination parameter must be specified",
        "May not work well with datasets that have multiple normal densities",
        "Performance depends on contamination parameter accuracy",
        "Less interpretable than rule-based methods"
    ],
    related_algorithms=["dbscan", "local-outlier-factor", "one-class-svm"]
)

AlgorithmRegistry.register(isolation_forest_metadata)


@router.get("/isolation-forest/info")
async def get_isolation_forest_info() -> Dict[str, Any]:
    """Get Isolation Forest algorithm information and metadata.

    Returns algorithm description, parameters, complexity, theory, use cases,
    and other metadata for display in the frontend.

    Returns:
        Dictionary containing algorithm metadata

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("isolation-forest")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["blobs_with_outliers"]
    }


@router.post("/isolation-forest/train")
async def train_isolation_forest(params: IsolationForestParameters) -> IsolationForestResponse:
    """Train Isolation Forest anomaly detection model.

    This endpoint trains an Isolation Forest model on synthetic data with
    injected outliers and returns anomaly detection results including predictions,
    anomaly scores, and visualization data.

    Args:
        params: Training parameters including n_estimators, contamination,
               max_samples, and data generation parameters

    Returns:
        Training results including metrics, anomaly info, and visualization data

    Raises:
        HTTPException: If training fails

    Example:
        >>> params = IsolationForestParameters(n_estimators=100, contamination=0.1)
        >>> response = await train_isolation_forest(params)
        >>> print(f"Detected {response.anomaly_info.n_anomalies} anomalies")
    """
    try:
        logger.info(f"Training Isolation Forest with parameters: {params.model_dump()}")
        model = IsolationForestModel()
        response = model.train(params)

        if not response.success:
            logger.error(f"Isolation Forest training failed: {response.error}")
            raise HTTPException(status_code=400, detail=response.error)

        logger.info(
            f"Isolation Forest training completed in {response.execution_time_ms:.2f}ms "
            f"with {response.anomaly_info.n_anomalies} anomalies detected "
            f"({response.anomaly_info.anomaly_ratio:.2%} anomaly ratio)"
        )
        return response

    except Exception as e:
        logger.error(f"Unexpected error during Isolation Forest training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Register Anomaly Detection metadata
anomaly_detection_metadata = AlgorithmMetadata(
    id="anomaly-detection",
    name="Anomaly Detection",
    slug="anomaly-detection",
    category=AlgorithmCategory.ML,
    description="Detect outliers and anomalies using multiple unsupervised methods",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["unsupervised", "anomaly-detection", "outlier-detection", "isolation-forest"],
    use_cases=[
        "Fraud detection",
        "Network security",
        "Manufacturing defect detection",
        "System health monitoring",
        "Medical diagnosis",
        "Sensor data validation"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*log(n)) for Isolation Forest",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="method",
            label="Detection Method",
            type="select",
            default="isolation_forest",
            options=[
                {"label": "Isolation Forest", "value": "isolation_forest"},
                {"label": "One-Class SVM", "value": "one_class_svm"},
                {"label": "Local Outlier Factor (LOF)", "value": "lof"},
                {"label": "Compare All Methods", "value": "compare"}
            ],
            description="Anomaly detection method to use"
        ),
        AlgorithmParameter(
            name="contamination",
            label="Expected Anomaly Ratio",
            type="range",
            default=0.1,
            min=0.01,
            max=0.5,
            step=0.01,
            description="Expected proportion of anomalies in the dataset"
        ),
        AlgorithmParameter(
            name="n_estimators",
            label="Number of Trees (Isolation Forest)",
            type="range",
            default=100,
            min=50,
            max=500,
            step=10,
            description="Number of isolation trees for Isolation Forest method"
        ),
        AlgorithmParameter(
            name="kernel",
            label="SVM Kernel",
            type="select",
            default="rbf",
            options=[
                {"label": "RBF (Radial Basis Function)", "value": "rbf"},
                {"label": "Linear", "value": "linear"},
                {"label": "Polynomial", "value": "poly"},
                {"label": "Sigmoid", "value": "sigmoid"}
            ],
            description="Kernel type for One-Class SVM"
        ),
        AlgorithmParameter(
            name="n_neighbors",
            label="Number of Neighbors (LOF)",
            type="range",
            default=20,
            min=5,
            max=50,
            step=1,
            description="Number of neighbors to consider for Local Outlier Factor"
        ),
        AlgorithmParameter(
            name="dataset",
            label="Dataset Type",
            type="select",
            default="synthetic",
            options=[
                {"label": "Synthetic 2D Data", "value": "synthetic"},
                {"label": "Credit Card Fraud", "value": "fraud"},
                {"label": "Network Intrusion", "value": "network"}
            ],
            description="Type of dataset to use for demonstration"
        )
    ],
    dataset_name="synthetic",
    visualization_type="scatter_heatmap_roc",
    theory=(
        "Anomaly Detection identifies rare items, events, or observations that differ significantly "
        "from the majority of the data. This implementation provides three complementary methods:\n\n"
        "1. **Isolation Forest**: Uses an ensemble of isolation trees to identify anomalies based on "
        "path length. Anomalies are easier to isolate (shorter paths) than normal points. Fast and "
        "scalable with O(n*log(n)) complexity.\n\n"
        "2. **One-Class SVM**: Learns a decision boundary around normal data in feature space. Points "
        "outside this boundary are classified as anomalies. Effective for finding global outliers with "
        "various kernel functions for non-linear boundaries.\n\n"
        "3. **Local Outlier Factor (LOF)**: Compares local density of a point with densities of its "
        "neighbors. Points in low-density regions relative to neighbors are anomalies. Excellent for "
        "finding local density-based anomalies.\n\n"
        "Each method has different strengths: Isolation Forest is fast and scalable, One-Class SVM "
        "handles non-linear boundaries well, and LOF excels at local anomalies. The 'Compare' mode "
        "runs all three methods and shows performance comparison."
    ),
    pros=[
        "Multiple methods for different anomaly types",
        "No labeled anomaly data required (unsupervised)",
        "Handles high-dimensional data well",
        "Provides interpretable anomaly scores",
        "Fast and scalable (especially Isolation Forest)",
        "Works with diverse data types and distributions"
    ],
    cons=[
        "Contamination parameter must be estimated",
        "May struggle with varying density clusters",
        "Performance depends on method-data match",
        "LOF is computationally expensive for large datasets",
        "One-Class SVM sensitive to kernel choice",
        "Difficult to validate without ground truth"
    ],
    related_algorithms=["isolation-forest", "dbscan", "k-means"]
)

AlgorithmRegistry.register(anomaly_detection_metadata)


@router.get("/anomaly-detection/info")
async def get_anomaly_detection_info() -> Dict[str, Any]:
    """Get Anomaly Detection algorithm information and metadata.

    Returns algorithm description, parameters, complexity, theory, use cases,
    and other metadata for display in the frontend. Also provides information
    about available datasets and detection methods.

    Returns:
        Dictionary containing algorithm metadata and dataset information

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("anomaly-detection")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    # Get dataset information
    datasets = {
        'synthetic': get_anomaly_dataset_info('synthetic'),
        'fraud': get_anomaly_dataset_info('fraud'),
        'network': get_anomaly_dataset_info('network')
    }

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": datasets,
        "methods": [
            {
                "name": "isolation_forest",
                "display_name": "Isolation Forest",
                "description": "Tree-based path length anomaly scoring",
                "complexity": "O(n*log(n))",
                "best_for": "Fast, scalable detection on large datasets"
            },
            {
                "name": "one_class_svm",
                "display_name": "One-Class SVM",
                "description": "Margin-based outlier detection",
                "complexity": "O(n²) to O(n³)",
                "best_for": "Non-linear boundaries and global outliers"
            },
            {
                "name": "lof",
                "display_name": "Local Outlier Factor",
                "description": "Density-based anomaly detection",
                "complexity": "O(n²)",
                "best_for": "Local density-based anomalies"
            }
        ]
    }


@router.post("/anomaly-detection/train")
async def train_anomaly_detection(params: AnomalyDetectionParameters) -> AnomalyDetectionResponse:
    """Train Anomaly Detection model using specified method.

    This endpoint trains an anomaly detection model using one of three methods:
    Isolation Forest, One-Class SVM, or Local Outlier Factor (LOF). It can also
    compare all three methods when method='compare'.

    The endpoint generates synthetic data with injected anomalies based on the
    selected dataset type, trains the model(s), and returns comprehensive results
    including anomaly predictions, scores, visualization data, and performance metrics.

    Args:
        params: Training parameters including method, contamination, algorithm-specific
               parameters, dataset type, and data generation parameters

    Returns:
        Training results including:
        - Anomaly predictions and scores
        - Performance metrics (precision, recall, F1, ROC-AUC)
        - Visualization data (scatter plots, score distributions, ROC curves)
        - Method comparison (if compare mode used)

    Raises:
        HTTPException: If training fails or invalid parameters provided

    Example:
        >>> params = AnomalyDetectionParameters(
        ...     method='isolation_forest',
        ...     contamination=0.1,
        ...     n_estimators=100,
        ...     dataset='synthetic'
        ... )
        >>> response = await train_anomaly_detection(params)
        >>> print(f"Detected {response.anomaly_info.n_anomalies} anomalies")
    """
    try:
        logger.info(f"Training Anomaly Detection with parameters: {params.model_dump()}")
        model = AnomalyDetectionModel()
        response = model.train(params)

        if not response.success:
            logger.error(f"Anomaly Detection training failed: {response.error}")
            raise HTTPException(status_code=400, detail=response.error)

        if params.method == 'compare':
            logger.info(
                f"Anomaly Detection comparison completed in {response.execution_time_ms:.2f}ms"
            )
            if response.method_comparison:
                for comp in response.method_comparison:
                    logger.info(
                        f"  {comp.method_name}: {comp.n_anomalies} anomalies, "
                        f"F1={comp.f1_score:.4f}, Precision={comp.precision:.4f}, "
                        f"Recall={comp.recall:.4f}"
                    )
        else:
            logger.info(
                f"Anomaly Detection ({params.method}) completed in {response.execution_time_ms:.2f}ms "
                f"with {response.anomaly_info.n_anomalies} anomalies detected "
                f"({response.anomaly_info.anomaly_ratio:.2%} anomaly ratio)"
            )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during Anomaly Detection training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Import and register Ensemble Methods
from algorithms.ml.ensemble_methods import (
    EnsembleMethodsModel,
    EnsembleMethodsRequest,
    EnsembleMethodsResponse
)
from algorithms.ml.ensemble_methods.data import (
    prepare_data as prepare_ensemble_data,
    prepare_visualization_data as prepare_ensemble_viz
)


# Register Ensemble Methods metadata
ensemble_methods_metadata = AlgorithmMetadata(
    id="ensemble-methods",
    name="Ensemble Methods",
    slug="ensemble-methods",
    category=AlgorithmCategory.ML,
    description="Combine multiple models to improve prediction accuracy and robustness",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["supervised", "classification", "ensemble", "ml", "bagging", "boosting", "stacking", "model-combination"],
    use_cases=[
        "Kaggle competitions",
        "Production ML systems",
        "Reducing overfitting",
        "Improving generalization",
        "Model uncertainty quantification",
        "High-stakes predictions"
    ],
    complexity=AlgorithmComplexity(
        time="O(n_estimators*base_complexity)",
        space="O(n_estimators*base_space)"
    ),
    parameters=[
        AlgorithmParameter(
            name="method",
            label="Ensemble Method",
            type="select",
            default="voting",
            options=[
                {"label": "Bagging (Bootstrap Aggregating)", "value": "bagging"},
                {"label": "Boosting (Gradient Boosting)", "value": "boosting"},
                {"label": "Stacking (Meta-Learning)", "value": "stacking"},
                {"label": "Voting (Soft Voting)", "value": "voting"},
                {"label": "Compare All Methods", "value": "all"}
            ],
            description="Ensemble method to use for combining models"
        ),
        AlgorithmParameter(
            name="n_estimators",
            label="Number of Base Models",
            type="range",
            default=10,
            min=3,
            max=100,
            step=1,
            description="Number of base models in the ensemble"
        ),
        AlgorithmParameter(
            name="base_model",
            label="Base Estimator",
            type="select",
            default="decision_tree",
            options=[
                {"label": "Decision Tree", "value": "decision_tree"},
                {"label": "Support Vector Machine", "value": "svm"},
                {"label": "K-Nearest Neighbors", "value": "knn"},
                {"label": "Logistic Regression", "value": "logistic"}
            ],
            description="Type of base estimator to use (for Bagging)"
        ),
        AlgorithmParameter(
            name="max_samples",
            label="Bagging Sample Ratio",
            type="range",
            default=0.8,
            min=0.5,
            max=1.0,
            step=0.1,
            description="Ratio of samples to draw for each base model (Bagging only)"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Boosting Learning Rate",
            type="range",
            default=1.0,
            min=0.1,
            max=2.0,
            step=0.1,
            description="Learning rate for boosting (shrinkage parameter)"
        )
    ],
    dataset_name="wine",
    visualization_type="performance_comparison_diversity_metrics_voting_patterns",
    theory=(
        "Ensemble Methods combine multiple machine learning models to create a stronger "
        "predictor than any single model. The key insight is that different models make "
        "different errors, and by combining them appropriately, we can reduce overall error.\n\n"
        "**Bagging (Bootstrap Aggregating)**: Trains multiple models on random subsets of the "
        "training data (bootstrap samples) and averages their predictions. Random Forest is the "
        "most famous bagging ensemble. Reduces variance and helps prevent overfitting.\n\n"
        "**Boosting**: Trains models sequentially, where each new model focuses on correcting "
        "errors made by previous models. Examples include AdaBoost and Gradient Boosting. "
        "Reduces both bias and variance, often achieving state-of-the-art performance.\n\n"
        "**Stacking**: Trains multiple diverse base models, then uses a meta-learner to combine "
        "their predictions. The meta-learner learns the optimal way to weight each base model's "
        "predictions. Can capture complex model interactions.\n\n"
        "**Voting**: Combines predictions from multiple models using majority voting (hard) or "
        "averaged probabilities (soft). Simple but effective when base models are diverse and "
        "uncorrelated in their errors."
    ),
    pros=[
        "Significantly improves prediction accuracy over single models",
        "Reduces overfitting through model averaging",
        "More robust to noise and outliers",
        "Provides confidence estimates through model agreement",
        "Can combine different types of models for complementary strengths",
        "Often wins machine learning competitions"
    ],
    cons=[
        "More computationally expensive than single models",
        "Requires more memory to store multiple models",
        "Less interpretable than individual models",
        "Training time increases linearly with number of models",
        "May overfit if base models are too complex or similar"
    ],
    related_algorithms=["random-forest", "xgboost", "adaboost", "gradient-boosting"]
)

AlgorithmRegistry.register(ensemble_methods_metadata)


@router.post("/ensemble-methods/train", response_model=EnsembleMethodsResponse)
async def train_ensemble_methods(request: EnsembleMethodsRequest):
    """Train an Ensemble Methods classifier.

    This endpoint trains various ensemble learning models including Bagging,
    Boosting, Stacking, and Voting methods. It compares ensemble performance
    against individual base models and provides diversity metrics.

    Args:
        request: Training parameters including ensemble method, n_estimators,
                base_model, max_samples, and learning_rate

    Returns:
        Training results including metrics, predictions, performance comparison,
        diversity metrics, and voting patterns

    Raises:
        HTTPException: If training fails or invalid parameters provided
    """
    try:
        logger.info(f"Training Ensemble Methods with parameters: {request.model_dump()}")

        start_time = time.time()

        # Prepare data
        X_train, X_test, y_train, y_test, metadata = prepare_ensemble_data(
            dataset_name=request.dataset_name,
            normalize=request.normalize,
            test_size=request.test_size,
            random_state=request.random_state
        )

        # Create and train model
        model = EnsembleMethodsModel(
            method=request.method,
            n_estimators=request.n_estimators,
            base_model=request.base_model,
            max_samples=request.max_samples,
            learning_rate=request.learning_rate,
            random_state=request.random_state
        )

        # Train and get results
        results = model.train(X_train, y_train, X_test, y_test)

        # Prepare performance comparison
        performance_comparison = {
            'single_models': results['single_model_results'],
            'ensemble_metrics': {
                'accuracy': results['metrics']['accuracy'],
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'f1_score': results['metrics']['f1_score']
            }
        }

        # Prepare visualization data
        viz_data = prepare_ensemble_viz(
            performance_comparison=performance_comparison,
            feature_importance=results['feature_importance'],
            feature_names=metadata['feature_names'],
            confusion_matrix_data=results['confusion_matrix'],
            class_labels=metadata['target_names'],
            diversity_metrics=results['diversity_metrics'],
            voting_data=results['voting_data'],
            individual_predictions=results['individual_predictions']
        )

        # Get model info
        model_info = model.get_model_info()
        model_info['n_features'] = X_train.shape[1]
        model_info['n_classes'] = metadata['n_classes']
        model_info['dataset'] = request.dataset_name

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Ensemble Methods training completed in {execution_time_ms:.2f}ms "
            f"with {request.method} method, accuracy: {results['metrics']['accuracy']:.4f}"
        )

        # Build response
        return EnsembleMethodsResponse(
            success=True,
            metrics={
                'accuracy': results['metrics']['accuracy'],
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'f1_score': results['metrics']['f1_score'],
                'train_accuracy': results['metrics']['train_accuracy'],
                'test_accuracy': results['metrics']['test_accuracy']
            },
            predictions=results['predictions'],
            actual=y_test.tolist(),
            visualization_data=viz_data,
            execution_time_ms=execution_time_ms,
            parameters={
                'method': request.method,
                'n_estimators': request.n_estimators,
                'base_model': request.base_model,
                'max_samples': request.max_samples,
                'learning_rate': request.learning_rate,
                'test_size': request.test_size,
                'random_state': request.random_state,
                'dataset_name': request.dataset_name,
                'normalize': request.normalize
            },
            model_info=model_info,
            message=f"Successfully trained {request.method} ensemble with {request.n_estimators} estimators"
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/ensemble-methods/info")
async def get_ensemble_methods_info():
    """Get metadata information about the Ensemble Methods algorithm.

    Returns algorithm description, parameters, complexity, use cases,
    and other metadata for display in the frontend.

    Returns:
        Dictionary with algorithm metadata

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("ensemble-methods")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Ensemble Methods algorithm metadata not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["wine", "breast_cancer", "iris", "digits"]
    }


# Import and register Feature Importance
from algorithms.ml.feature_importance import (
    FeatureImportanceModel,
    FeatureImportanceRequest,
    FeatureImportanceResponse
)
from algorithms.ml.feature_importance.data import get_dataset_info as get_fi_dataset_info, get_available_datasets as get_fi_datasets

# Register Feature Importance metadata
feature_importance_metadata = AlgorithmMetadata(
    id="feature-importance",
    name="Feature Importance Analysis",
    slug="feature-importance",
    category=AlgorithmCategory.ML,
    description="Analyze and rank feature importance using multiple methods",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["ml", "feature-importance", "interpretability", "feature-selection", "explainability"],
    use_cases=[
        "Feature selection",
        "Model interpretability",
        "Dimensionality reduction",
        "Data understanding",
        "Model debugging",
        "Domain insight discovery"
    ],
    complexity=AlgorithmComplexity(
        time="O(n_features*n_estimators)",
        space="O(n_features)"
    ),
    parameters=[
        AlgorithmParameter(
            name="method",
            label="Importance Method",
            type="select",
            default="tree",
            options=[
                {"label": "Tree-based (MDI)", "value": "tree"},
                {"label": "Permutation Importance", "value": "permutation"},
                {"label": "All Methods", "value": "all"}
            ],
            description="Method for computing feature importance"
        ),
        AlgorithmParameter(
            name="model_type",
            label="Model Type",
            type="select",
            default="random_forest",
            options=[
                {"label": "Random Forest", "value": "random_forest"},
                {"label": "XGBoost", "value": "xgboost"},
                {"label": "Gradient Boosting", "value": "gradient_boosting"}
            ],
            description="Type of model to use for importance analysis"
        ),
        AlgorithmParameter(
            name="n_estimators",
            label="Number of Estimators",
            type="range",
            default=100,
            min=10,
            max=500,
            step=10,
            description="Number of trees in the ensemble"
        ),
        AlgorithmParameter(
            name="top_k",
            label="Top K Features",
            type="range",
            default=10,
            min=5,
            max=30,
            step=1,
            description="Number of top features to highlight"
        ),
        AlgorithmParameter(
            name="dataset",
            label="Dataset",
            type="select",
            default="housing",
            options=[
                {"label": "California Housing", "value": "housing"},
                {"label": "Diabetes", "value": "diabetes"},
                {"label": "Wine", "value": "wine"}
            ],
            description="Dataset to use for analysis"
        )
    ],
    dataset_name="housing",
    visualization_type="bar_chart_heatmap_cumulative",
    theory=(
        "Feature Importance Analysis helps identify which features (input variables) have the most "
        "impact on model predictions. This is crucial for model interpretability, feature selection, "
        "and understanding the underlying data relationships.\n\n"
        "**Methods:**\n"
        "1. **Tree-based Importance (MDI - Mean Decrease in Impurity)**: Measures how much each "
        "feature decreases the weighted impurity in tree-based models. Features that lead to larger "
        "decreases in impurity are more important. This method is fast and built into tree models.\n\n"
        "2. **Permutation Importance**: Model-agnostic method that measures the decrease in model "
        "performance when a feature's values are randomly shuffled. More important features cause "
        "larger performance drops when permuted. This method is more reliable but computationally expensive.\n\n"
        "3. **SHAP Values** (optional): Game theory-based approach that assigns importance values to "
        "features for individual predictions, providing local and global interpretability.\n\n"
        "The visualization includes:\n"
        "- Horizontal bar charts showing importance scores\n"
        "- Comparison charts when multiple methods are used\n"
        "- Feature correlation heatmap\n"
        "- Cumulative importance curve\n"
        "- Statistical summaries and rankings"
    ),
    pros=[
        "Helps understand which features drive predictions",
        "Enables effective feature selection and dimensionality reduction",
        "Identifies potentially redundant or irrelevant features",
        "Supports model debugging and validation",
        "Provides insights into domain relationships",
        "Multiple methods for robust analysis"
    ],
    cons=[
        "Tree-based importance can be biased toward high-cardinality features",
        "Permutation importance is computationally expensive",
        "Results may vary between methods",
        "Does not capture feature interactions directly",
        "Importance scores are relative, not absolute measures"
    ],
    related_algorithms=["random-forest", "xgboost", "gradient-boosting", "lasso-regression"]
)

AlgorithmRegistry.register(feature_importance_metadata)


@router.get("/feature-importance/info")
async def get_feature_importance_info() -> Dict[str, Any]:
    """Get Feature Importance algorithm information and metadata.

    Returns algorithm description, parameters, available methods, datasets,
    and educational content for display in the frontend.

    Returns:
        Dictionary containing algorithm metadata and available options

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("feature-importance")
    if not metadata:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": get_fi_datasets(),
        "available_methods": ["tree", "permutation", "all"],
        "available_models": ["random_forest", "xgboost", "gradient_boosting"]
    }


@router.post("/feature-importance/train", response_model=FeatureImportanceResponse)
async def train_feature_importance(request: FeatureImportanceRequest) -> FeatureImportanceResponse:
    """Analyze feature importance using specified methods.

    This endpoint trains a model on the selected dataset and computes feature
    importance using one or more methods (tree-based, permutation, or all).
    It returns comprehensive importance scores, rankings, correlations, and
    visualization data.

    Args:
        request: Feature importance analysis parameters including method,
                model_type, n_estimators, top_k, and dataset

    Returns:
        Feature importance results including:
        - Multiple importance scores by method
        - Feature rankings and statistics
        - Correlation matrix
        - Cumulative importance curves
        - Visualization-ready data

    Raises:
        HTTPException: If training or analysis fails

    Example:
        >>> request = FeatureImportanceRequest(
        ...     method="all",
        ...     model_type="random_forest",
        ...     dataset="housing",
        ...     top_k=10
        ... )
        >>> response = await train_feature_importance(request)
        >>> print(f"Top feature: {response.feature_rankings['tree'][0]['feature']}")
    """
    try:
        logger.info(
            f"Running Feature Importance Analysis: "
            f"method={request.method}, model={request.model_type}, "
            f"dataset={request.dataset}, top_k={request.top_k}"
        )

        # Create and train model
        model = FeatureImportanceModel()
        response = model.train(request)

        # Log results
        methods_used = list(response.feature_importance.keys())
        logger.info(
            f"Feature Importance analysis completed in {response.execution_time_ms:.2f}ms "
            f"using methods: {methods_used}. "
            f"Model performance: {response.metrics}"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error in Feature Importance: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Feature Importance analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feature importance analysis failed: {str(e)}"
        )


# Import and register Time Series Forecasting
from algorithms.ml.time_series_forecasting import (
    TimeSeriesForecastingModel,
    TimeSeriesForecastingRequest,
    TimeSeriesForecastingResponse,
    load_time_series_data,
    get_dataset_info as get_ts_dataset_info
)

# Register Time Series Forecasting metadata
time_series_forecasting_metadata = AlgorithmMetadata(
    id="time-series-forecasting",
    name="Time Series Forecasting",
    slug="time-series-forecasting",
    category=AlgorithmCategory.ML,
    description="Predict future values in temporal data using statistical and deep learning methods",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["ml", "time-series", "forecasting", "arima", "prophet", "lstm", "temporal"],
    use_cases=[
        "Demand forecasting",
        "Financial predictions",
        "Resource planning",
        "Anomaly detection",
        "Capacity planning",
        "Trend analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*p*q) for ARIMA, O(n*hidden) for LSTM",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="method",
            label="Forecast Method",
            type="select",
            default="arima",
            options=[
                {"label": "ARIMA (Statistical)", "value": "arima"},
                {"label": "Prophet (Facebook)", "value": "prophet"},
                {"label": "LSTM (Deep Learning)", "value": "lstm"},
                {"label": "Compare All Methods", "value": "compare"}
            ],
            description="Forecasting method to use"
        ),
        AlgorithmParameter(
            name="forecast_periods",
            label="Forecast Periods",
            type="range",
            default=30,
            min=5,
            max=100,
            step=5,
            description="Number of periods to forecast ahead"
        ),
        AlgorithmParameter(
            name="p",
            label="ARIMA AR Order (p)",
            type="range",
            default=5,
            min=0,
            max=10,
            step=1,
            description="Autoregressive order for ARIMA"
        ),
        AlgorithmParameter(
            name="d",
            label="ARIMA Differencing (d)",
            type="range",
            default=1,
            min=0,
            max=2,
            step=1,
            description="Differencing order for ARIMA"
        ),
        AlgorithmParameter(
            name="q",
            label="ARIMA MA Order (q)",
            type="range",
            default=0,
            min=0,
            max=10,
            step=1,
            description="Moving average order for ARIMA"
        ),
        AlgorithmParameter(
            name="series_index",
            label="Dataset",
            type="select",
            default=0,
            options=[
                {"label": "Stock Prices", "value": 0},
                {"label": "Temperature", "value": 1},
                {"label": "Sales", "value": 2},
                {"label": "Web Traffic", "value": 3},
                {"label": "Sensor Readings", "value": 4}
            ],
            description="Time series dataset to use"
        ),
        AlgorithmParameter(
            name="confidence_level",
            label="Confidence Level",
            type="range",
            default=0.95,
            min=0.80,
            max=0.99,
            step=0.01,
            description="Confidence level for prediction intervals"
        )
    ],
    dataset_name="time_series",
    visualization_type="line_chart_forecast_decomposition_acf_pacf",
    theory=(
        "Time series forecasting predicts future values based on previously observed values. "
        "This implementation includes three methods:\n\n"
        "**ARIMA (AutoRegressive Integrated Moving Average)**: A statistical method that models "
        "the time series as a combination of autoregressive (AR), differencing (I), and moving "
        "average (MA) components. The model is defined by three orders (p, d, q) where p is the "
        "AR order, d is the differencing order, and q is the MA order.\n\n"
        "**Prophet**: Facebook's time series forecasting algorithm designed for business time series. "
        "It decomposes the series into trend, seasonality, and holiday effects, and is robust to "
        "missing data and outliers.\n\n"
        "**LSTM (Long Short-Term Memory)**: A deep learning recurrent neural network that can learn "
        "long-term dependencies in sequential data. It uses memory cells and gates to control "
        "information flow, making it effective for complex patterns."
    ),
    pros=[
        "Multiple methods for different data characteristics",
        "Provides confidence intervals for uncertainty quantification",
        "Handles various time series patterns (trend, seasonality, cycles)",
        "Decomposition helps understand underlying patterns",
        "Comparison mode identifies best method automatically"
    ],
    cons=[
        "Requires sufficient historical data",
        "Assumes patterns continue into future",
        "Sensitive to outliers and structural breaks",
        "LSTM requires more data and computational resources",
        "Parameter tuning can be complex for ARIMA"
    ],
    related_algorithms=["linear-regression", "lstm-autoencoder", "xgboost"]
)

AlgorithmRegistry.register(time_series_forecasting_metadata)


@router.post("/time-series-forecasting/train", response_model=TimeSeriesForecastingResponse)
async def train_time_series_forecasting(request: TimeSeriesForecastingRequest) -> TimeSeriesForecastingResponse:
    """Train time series forecasting model and generate predictions.

    This endpoint supports multiple forecasting methods (ARIMA, Prophet, LSTM)
    on various time series datasets with trend, seasonality, and noise patterns.

    Args:
        request: Forecasting parameters including method, forecast periods,
                ARIMA orders, dataset selection, and confidence level

    Returns:
        Forecasting results including predictions, confidence intervals,
        metrics, decomposition, and diagnostic plots

    Raises:
        HTTPException: If forecasting fails or invalid parameters provided

    Example:
        >>> request = TimeSeriesForecastingRequest(
        ...     method="arima",
        ...     forecast_periods=30,
        ...     p=5, d=1, q=0,
        ...     series_index=0
        ... )
        >>> response = await train_time_series_forecasting(request)
        >>> print(f"MAE: {response.metrics['mae']:.2f}")
    """
    try:
        logger.info(
            f"Training Time Series Forecasting: "
            f"method={request.method}, periods={request.forecast_periods}, "
            f"order=({request.p},{request.d},{request.q}), series={request.series_index}"
        )

        # Load time series data
        ts_data = load_time_series_data(
            series_index=request.series_index,
            random_state=request.random_state
        )

        data = ts_data['data']
        dates = ts_data['dates']

        # Create and train model
        model = TimeSeriesForecastingModel(method=request.method)
        results = model.train(
            data=data,
            dates=dates,
            forecast_periods=request.forecast_periods,
            p=request.p,
            d=request.d,
            q=request.q,
            confidence_level=request.confidence_level
        )

        # Prepare visualization data
        historical_points = []
        for i, (date, value) in enumerate(zip(dates, data)):
            historical_points.append({
                'index': i,
                'date': date.isoformat(),
                'value': float(value)
            })

        # Forecast points
        forecast_start_index = len(data)
        last_date = dates[-1]
        freq = dates.freq
        forecast_dates = pd.date_range(start=last_date + freq, periods=request.forecast_periods, freq=freq)

        forecast_points = []
        for i, (date, value, lower, upper) in enumerate(zip(
            forecast_dates,
            results['forecast'],
            results['lower_bound'],
            results['upper_bound']
        )):
            forecast_points.append({
                'index': forecast_start_index + i,
                'date': date.isoformat(),
                'value': float(value),
                'lower': float(lower),
                'upper': float(upper)
            })

        # Prepare decomposition visualization if available
        decomposition_viz = None
        if results.get('decomposition'):
            decomp = results['decomposition']
            decomposition_viz = {
                'trend': [{'index': i, 'value': float(v)} for i, v in enumerate(decomp['trend'])],
                'seasonal': [{'index': i, 'value': float(v)} for i, v in enumerate(decomp['seasonal'])],
                'residual': [{'index': i, 'value': float(v)} for i, v in enumerate(decomp['residual'])]
            }

        visualization_data = {
            'historical': historical_points,
            'forecast': forecast_points,
            'decomposition': decomposition_viz,
            'acf_pacf': results.get('acf_data'),
            'dataset_name': ts_data['name'],
            'units': ts_data['units'],
            'frequency': ts_data['frequency']
        }

        logger.info(
            f"Time Series Forecasting completed in {results['execution_time_ms']:.2f}ms. "
            f"Metrics: MAE={results['metrics']['mae']:.2f}, "
            f"RMSE={results['metrics']['rmse']:.2f}, "
            f"MAPE={results['metrics']['mape']:.2f}%"
        )

        return TimeSeriesForecastingResponse(
            success=True,
            historical_data=data.tolist(),
            forecast=results['forecast'],
            lower_bound=results['lower_bound'],
            upper_bound=results['upper_bound'],
            metrics=results['metrics'],
            decomposition=results.get('decomposition'),
            acf_data=results.get('acf_data'),
            visualization_data=visualization_data,
            execution_time_ms=results['execution_time_ms'],
            model_info=results['model_info'],
            parameters_used={
                'method': request.method,
                'forecast_periods': request.forecast_periods,
                'p': request.p,
                'd': request.d,
                'q': request.q,
                'series_index': request.series_index,
                'confidence_level': request.confidence_level
            }
        )

    except ImportError as e:
        logger.error(f"Missing dependency for Time Series Forecasting: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Missing required library. Please install: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error in Time Series Forecasting: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Time Series Forecasting error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Time series forecasting failed: {str(e)}"
        )


@router.get("/time-series-forecasting/info")
async def get_time_series_forecasting_info() -> Dict[str, Any]:
    """Get Time Series Forecasting algorithm information and metadata.

    Returns algorithm description, parameters, complexity, theory, use cases,
    available methods, and supported datasets.

    Returns:
        Dictionary containing algorithm metadata and configuration

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("time-series-forecasting")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Time Series Forecasting algorithm not found"
        )

    dataset_info = get_ts_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "datasets": dataset_info,
        "methods": {
            "arima": {
                "name": "ARIMA",
                "description": "Statistical autoregressive integrated moving average",
                "best_for": "Stationary time series with linear patterns"
            },
            "prophet": {
                "name": "Prophet",
                "description": "Facebook's time series forecasting with seasonality",
                "best_for": "Business time series with strong seasonal patterns"
            },
            "lstm": {
                "name": "LSTM",
                "description": "Deep learning long short-term memory network",
                "best_for": "Complex non-linear patterns with long-term dependencies"
            },
            "compare": {
                "name": "Compare All",
                "description": "Automatically selects best method based on performance",
                "best_for": "When unsure which method to use"
            }
        }
    }


# Register Regularization Techniques metadata
regularization_metadata = AlgorithmMetadata(
    id="regularization",
    name="Regularization Techniques",
    slug="regularization",
    category=AlgorithmCategory.ML,
    description="Compare L1, L2, Elastic Net, and Early Stopping for preventing overfitting",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["ml", "regularization", "overfitting", "l1", "l2", "elastic-net"],
    use_cases=[
        "Preventing overfitting",
        "Feature selection",
        "Model simplification",
        "High-dimensional problems",
        "Small dataset training",
        "Production model robustness"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*p) to O(n*p²)",
        space="O(p)"
    ),
    parameters=[
        AlgorithmParameter(
            name="technique",
            label="Regularization Technique",
            type="select",
            default="l2",
            options=[
                {"label": "L1 (Lasso) - Sparse features", "value": "l1"},
                {"label": "L2 (Ridge) - Stable coefficients", "value": "l2"},
                {"label": "Elastic Net - L1 + L2 mix", "value": "elastic_net"},
                {"label": "Early Stopping - Stop before overfit", "value": "early_stopping"},
                {"label": "Compare All Techniques", "value": "compare"}
            ],
            description="Type of regularization to apply"
        ),
        AlgorithmParameter(
            name="alpha",
            label="Regularization Strength (Alpha)",
            type="range",
            default=1.0,
            min=0.001,
            max=100.0,
            step=0.1,
            description="Regularization strength (higher = more regularization)"
        ),
        AlgorithmParameter(
            name="l1_ratio",
            label="L1 Ratio (Elastic Net Mix)",
            type="range",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Mix between L1 and L2 (0.0 = Ridge, 1.0 = Lasso, 0.5 = balanced)"
        ),
        AlgorithmParameter(
            name="max_iterations",
            label="Maximum Iterations",
            type="number",
            default=1000,
            min=100,
            max=5000,
            step=100,
            description="Maximum training iterations"
        ),
        AlgorithmParameter(
            name="early_stopping_rounds",
            label="Early Stopping Patience",
            type="number",
            default=10,
            min=5,
            max=50,
            step=1,
            description="Number of rounds with no improvement before stopping"
        )
    ],
    dataset_name="synthetic",
    visualization_type="coefficient_paths_loss_curves_sparsity",
    theory=(
        "Regularization techniques prevent overfitting by adding penalties to the loss function:\n\n"
        "1. **L1 (Lasso)**: Adds |w| penalty, creates sparse models by driving some coefficients "
        "to exactly zero. Useful for feature selection.\n\n"
        "2. **L2 (Ridge)**: Adds w² penalty, shrinks all coefficients but keeps them non-zero. "
        "More stable than L1, handles multicollinearity well.\n\n"
        "3. **Elastic Net**: Combines L1 and L2 penalties with mixing parameter. Gets benefits "
        "of both approaches - sparsity and stability.\n\n"
        "4. **Early Stopping**: Monitors validation loss and stops training when it stops improving. "
        "Prevents overfitting without modifying the model structure.\n\n"
        "The optimization objective becomes: Loss(y, ŷ) + λ * Penalty(w) where λ (alpha) controls "
        "regularization strength."
    ),
    pros=[
        "Prevents overfitting on training data",
        "Improves generalization to unseen data",
        "L1 provides automatic feature selection",
        "L2 handles multicollinearity",
        "Elastic Net combines benefits of L1 and L2",
        "Early stopping is model-agnostic",
        "Reduces model variance"
    ],
    cons=[
        "Requires tuning regularization strength (alpha)",
        "L1 can be unstable with correlated features",
        "L2 doesn't perform feature selection",
        "May underfit if regularization too strong",
        "Early stopping needs validation set",
        "Computational overhead for parameter tuning"
    ],
    related_algorithms=["ridge-regression", "lasso-regression", "elastic-net"]
)

AlgorithmRegistry.register(regularization_metadata)


@router.post("/regularization/train", response_model=RegularizationResponse)
async def train_regularization(request: RegularizationRequest):
    """Train Regularization Techniques model.

    This endpoint demonstrates different regularization approaches for preventing
    overfitting, including L1 (Lasso), L2 (Ridge), Elastic Net, and Early Stopping.

    Args:
        request: Training parameters including technique, alpha, l1_ratio, etc.

    Returns:
        RegularizationResponse with training results, metrics, and visualization data

    Raises:
        HTTPException: If training fails or invalid parameters provided
    """
    try:
        start_time = time.time()

        # Create overfitting-prone dataset
        data = create_overfitting_prone_dataset(
            n_samples=request.n_samples,
            n_features=request.n_features,
            random_state=42
        )

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        feature_names = data['feature_names']

        # Normalize if requested
        if request.normalize:
            from utils.datasets import DatasetManager
            X_train, X_test = DatasetManager.normalize_data(X_train, X_test)

        # Initialize and train model
        model = RegularizationModel(
            technique=request.technique,
            alpha=request.alpha,
            l1_ratio=request.l1_ratio,
            max_iterations=request.max_iterations,
            early_stopping_rounds=request.early_stopping_rounds,
            random_state=42
        )

        # Train model
        train_results = model.train(X_train, y_train, X_test, y_test)

        # Prepare visualization data based on technique
        visualization_data = {}

        if request.technique == 'compare':
            # Comparison mode - prepare all visualizations
            comparison = train_results['comparison']

            # Coefficient paths for all techniques
            visualization_data['coefficient_paths'] = {
                'l1': comparison['l1']['coefficient_path'],
                'l2': comparison['l2']['coefficient_path'],
                'elastic_net': comparison['elastic_net']['coefficient_path']
            }

            # Sparsity comparison
            visualization_data['sparsity_comparison'] = train_results['sparsity_comparison']

            # Overfitting comparison (train vs test gap)
            visualization_data['overfitting_comparison'] = [
                {
                    'technique': 'L1 (Lasso)',
                    'train_r2': comparison['l1']['metrics']['train_r2'],
                    'test_r2': comparison['l1']['metrics']['test_r2'],
                    'gap': comparison['l1']['metrics']['overfitting_gap']
                },
                {
                    'technique': 'L2 (Ridge)',
                    'train_r2': comparison['l2']['metrics']['train_r2'],
                    'test_r2': comparison['l2']['metrics']['test_r2'],
                    'gap': comparison['l2']['metrics']['overfitting_gap']
                },
                {
                    'technique': 'Elastic Net',
                    'train_r2': comparison['elastic_net']['metrics']['train_r2'],
                    'test_r2': comparison['elastic_net']['metrics']['test_r2'],
                    'gap': comparison['elastic_net']['metrics']['overfitting_gap']
                },
                {
                    'technique': 'Early Stopping',
                    'train_r2': comparison['early_stopping']['metrics']['train_r2'],
                    'test_r2': comparison['early_stopping']['metrics']['test_r2'],
                    'gap': comparison['early_stopping']['metrics']['overfitting_gap']
                }
            ]

            # Early stopping loss curves
            if 'train_losses' in comparison['early_stopping']:
                visualization_data['loss_curves'] = {
                    'iterations': list(range(len(comparison['early_stopping']['train_losses']))),
                    'train_loss': comparison['early_stopping']['train_losses'],
                    'val_loss': comparison['early_stopping']['val_losses'],
                    'optimal_iteration': comparison['early_stopping']['optimal_iterations']
                }

            # Prepare metrics summary for comparison
            metrics = {
                'l1': comparison['l1']['metrics'],
                'l2': comparison['l2']['metrics'],
                'elastic_net': comparison['elastic_net']['metrics'],
                'early_stopping': comparison['early_stopping']['metrics']
            }

            predictions = train_results['predictions']

        else:
            # Single technique mode
            if 'coefficient_path' in train_results:
                visualization_data['coefficient_paths'] = {
                    request.technique: train_results['coefficient_path']
                }

            # Early stopping loss curves
            if 'train_losses' in train_results:
                visualization_data['loss_curves'] = {
                    'iterations': list(range(len(train_results['train_losses']))),
                    'train_loss': train_results['train_losses'],
                    'val_loss': train_results['val_losses'],
                    'optimal_iteration': train_results.get('optimal_iterations', 0)
                }

            # Sparsity info
            visualization_data['sparsity'] = train_results.get('sparsity', 0.0)

            metrics = train_results['metrics']
            predictions = train_results['predictions']

        # Predictions vs Actual chart data
        visualization_data['predictions_chart'] = [
            {
                "index": i,
                "predicted": float(pred),
                "actual": float(actual)
            }
            for i, (pred, actual) in enumerate(zip(predictions[:100], y_test[:100]))
        ]

        execution_time_ms = (time.time() - start_time) * 1000

        # Get model info
        model_info = model.get_model_info()
        model_info['n_samples'] = request.n_samples
        model_info['n_features'] = request.n_features

        return RegularizationResponse(
            success=True,
            metrics=metrics,
            predictions=predictions[:100] if isinstance(predictions, list) else predictions.tolist()[:100],
            actual=y_test[:100].tolist(),
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                "technique": request.technique,
                "alpha": request.alpha,
                "l1_ratio": request.l1_ratio,
                "max_iterations": request.max_iterations,
                "early_stopping_rounds": request.early_stopping_rounds,
                "n_samples": request.n_samples,
                "n_features": request.n_features,
                "normalize": request.normalize
            }
        )

    except ValueError as e:
        logger.error(f"Validation error in Regularization training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in Regularization training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/regularization/info")
async def get_regularization_info() -> Dict[str, Any]:
    """Get Regularization Techniques algorithm information.

    Returns metadata, parameters, and dataset information for Regularization Techniques.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("regularization")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Regularization Techniques metadata not found"
        )

    dataset_info = get_regularization_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "techniques": {
            "l1": {
                "name": "L1 (Lasso)",
                "description": "L1 regularization with sparse coefficients",
                "penalty": "Sum of absolute values of coefficients",
                "best_for": "Feature selection and sparse models"
            },
            "l2": {
                "name": "L2 (Ridge)",
                "description": "L2 regularization with stable coefficients",
                "penalty": "Sum of squared coefficients",
                "best_for": "Multicollinearity and stable predictions"
            },
            "elastic_net": {
                "name": "Elastic Net",
                "description": "Combination of L1 and L2 regularization",
                "penalty": "Weighted sum of L1 and L2 penalties",
                "best_for": "Correlated features with feature selection"
            },
            "early_stopping": {
                "name": "Early Stopping",
                "description": "Stop training when validation loss stops improving",
                "penalty": "None (prevents overfitting via iteration control)",
                "best_for": "Any model that can overfit during training"
            }
        }
    }


# Import and register Hyperparameter Tuning
from algorithms.ml.hyperparameter_tuning import (
    HyperparameterTuningModel,
    HyperparameterTuningRequest,
    HyperparameterTuningResponse,
    TrialResult
)
from algorithms.ml.hyperparameter_tuning.data import get_dataset_info as get_hp_dataset_info


# Register Hyperparameter Tuning metadata
hyperparameter_tuning_metadata = AlgorithmMetadata(
    id="hyperparameter-tuning",
    name="Hyperparameter Tuning",
    slug="hyperparameter-tuning",
    category=AlgorithmCategory.ML,
    description="Optimize model hyperparameters using multiple search strategies",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["ml", "hyperparameter-tuning", "optimization", "grid-search", "bayesian-optimization"],
    use_cases=[
        "Model optimization",
        "Performance improvement",
        "AutoML pipelines",
        "Competition preparation",
        "Production model tuning",
        "Baseline establishment"
    ],
    complexity=AlgorithmComplexity(
        time="O(n_trials*model_training)",
        space="O(n_trials)"
    ),
    parameters=[
        AlgorithmParameter(
            name="method",
            label="Search Method",
            type="select",
            default="grid",
            options=[
                {"label": "Grid Search (Exhaustive)", "value": "grid"},
                {"label": "Random Search (Sampling)", "value": "random"},
                {"label": "Bayesian Optimization (Smart)", "value": "bayesian"},
                {"label": "Compare All Methods", "value": "compare"}
            ],
            description="Search strategy for hyperparameter optimization"
        ),
        AlgorithmParameter(
            name="model_type",
            label="Model to Tune",
            type="select",
            default="random_forest",
            options=[
                {"label": "Random Forest", "value": "random_forest"},
                {"label": "Support Vector Machine", "value": "svm"},
                {"label": "XGBoost", "value": "xgboost"},
                {"label": "MLP Neural Network", "value": "mlp"}
            ],
            description="Type of model to optimize"
        ),
        AlgorithmParameter(
            name="n_trials",
            label="Number of Trials",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of parameter combinations to try (random/bayesian)"
        ),
        AlgorithmParameter(
            name="cv_folds",
            label="Cross-Validation Folds",
            type="range",
            default=5,
            min=3,
            max=10,
            step=1,
            description="Number of cross-validation folds for evaluation"
        ),
        AlgorithmParameter(
            name="scoring",
            label="Scoring Metric",
            type="select",
            default="accuracy",
            options=[
                {"label": "Accuracy", "value": "accuracy"},
                {"label": "F1 Score", "value": "f1"},
                {"label": "ROC AUC", "value": "roc_auc"},
                {"label": "Precision", "value": "precision"},
                {"label": "Recall", "value": "recall"}
            ],
            description="Metric for evaluating model performance"
        ),
        AlgorithmParameter(
            name="dataset",
            label="Dataset",
            type="select",
            default="iris",
            options=[
                {"label": "Iris (3 classes, 4 features)", "value": "iris"},
                {"label": "Wine (3 classes, 13 features)", "value": "wine"},
                {"label": "Breast Cancer (2 classes, 30 features)", "value": "breast_cancer"}
            ],
            description="Dataset to use for tuning"
        )
    ],
    dataset_name="iris",
    visualization_type="convergence_heatmap_comparison",
    theory=(
        "Hyperparameter Tuning is the process of finding optimal hyperparameters for machine "
        "learning models to maximize performance. Unlike model parameters (learned during training), "
        "hyperparameters are set before training and control the learning process.\n\n"
        "**Grid Search**: Exhaustively searches through a manually specified subset of the "
        "hyperparameter space. It evaluates all possible combinations using cross-validation. "
        "While thorough, it becomes computationally expensive with many parameters.\n\n"
        "**Random Search**: Randomly samples parameter combinations from specified distributions. "
        "Often more efficient than grid search, especially when only a few hyperparameters "
        "significantly affect performance. It can explore a wider range with fewer evaluations.\n\n"
        "**Bayesian Optimization**: Uses probabilistic models (Gaussian Processes or Tree-structured "
        "Parzen Estimators) to model the objective function and select promising parameter "
        "combinations. It learns from previous evaluations to intelligently explore the space, "
        "typically finding good solutions with fewer trials than random search.\n\n"
        "The process involves:\n"
        "1. Defining parameter search space\n"
        "2. Selecting search strategy\n"
        "3. Evaluating combinations via cross-validation\n"
        "4. Tracking best parameters and convergence\n"
        "5. Testing final model on held-out test set"
    ),
    pros=[
        "Significantly improves model performance over default parameters",
        "Automates the parameter selection process",
        "Multiple strategies for different scenarios (time vs. quality)",
        "Provides convergence analysis to track optimization progress",
        "Cross-validation ensures robust parameter selection",
        "Bayesian methods are sample-efficient for expensive models"
    ],
    cons=[
        "Computationally expensive, especially grid search",
        "Risk of overfitting to validation set with many trials",
        "Requires domain knowledge to define parameter spaces",
        "No guarantee of finding global optimum",
        "Results depend on cross-validation splits",
        "May require specialized libraries (Optuna for Bayesian)"
    ],
    related_algorithms=["random-forest", "xgboost", "svm", "neural-networks"]
)

AlgorithmRegistry.register(hyperparameter_tuning_metadata)


@router.post("/hyperparameter-tuning/train", response_model=HyperparameterTuningResponse)
async def train_hyperparameter_tuning(request: HyperparameterTuningRequest) -> HyperparameterTuningResponse:
    """Perform hyperparameter tuning using specified search method.

    This endpoint optimizes model hyperparameters using grid search, random search,
    or Bayesian optimization. It tracks all trials, convergence, and provides
    comprehensive visualization data for understanding the optimization process.

    Args:
        request: Tuning parameters including method, model_type, n_trials,
                cv_folds, scoring metric, and dataset

    Returns:
        Tuning results including:
        - Best hyperparameters found
        - Best cross-validation score
        - Test set performance
        - All trial results and rankings
        - Convergence data over trials
        - Parameter space heatmap
        - Parameter importance (Bayesian only)
        - Method comparison (compare mode)

    Raises:
        HTTPException: If tuning fails or invalid parameters provided

    Example:
        >>> request = HyperparameterTuningRequest(
        ...     method="bayesian",
        ...     model_type="random_forest",
        ...     n_trials=50,
        ...     cv_folds=5,
        ...     scoring="accuracy",
        ...     dataset="iris"
        ... )
        >>> response = await train_hyperparameter_tuning(request)
        >>> print(f"Best params: {response.best_params}")
        >>> print(f"Best CV score: {response.best_score:.4f}")
    """
    try:
        logger.info(
            f"Starting Hyperparameter Tuning: "
            f"method={request.method}, model={request.model_type}, "
            f"trials={request.n_trials}, cv={request.cv_folds}, "
            f"scoring={request.scoring}, dataset={request.dataset}"
        )

        # Create and run tuning model
        model = HyperparameterTuningModel()
        results = model.train(
            method=request.method,
            model_type=request.model_type,
            n_trials=request.n_trials,
            cv_folds=request.cv_folds,
            scoring=request.scoring,
            dataset=request.dataset,
            random_state=request.random_state
        )

        if not results['success']:
            logger.error(f"Hyperparameter tuning failed: {results.get('error')}")
            return HyperparameterTuningResponse(
                success=False,
                error=results.get('error', 'Unknown error'),
                execution_time_ms=results.get('execution_time_ms', 0.0),
                parameters_used=results.get('parameters_used', {})
            )

        # Handle comparison mode
        if request.method == 'compare':
            comparison_viz = []
            for method_name, method_data in results['comparison_data'].items():
                if 'error' not in method_data:
                    comparison_viz.append({
                        'method': method_name,
                        'best_score': method_data['best_score'],
                        'n_trials': method_data['n_trials']
                    })

            logger.info(
                f"Hyperparameter Tuning comparison completed in {results['execution_time_ms']:.2f}ms. "
                f"Best method: {results['best_method']} with score {results['best_score']:.4f}"
            )

            return HyperparameterTuningResponse(
                success=True,
                best_score=results['best_score'],
                best_params=results['best_params'],
                comparison_data=comparison_viz,
                visualization_data={
                    'comparison': comparison_viz,
                    'best_method': results['best_method']
                },
                execution_time_ms=results['execution_time_ms'],
                parameters_used=results['parameters_used']
            )

        # Handle single method results
        trials_formatted = [
            TrialResult(**trial)
            for trial in results['trials']
        ]

        # Prepare visualization data
        visualization_data = {
            'convergence': results['convergence_data'],
            'heatmap': results.get('heatmap_data'),
            'parameter_importance': results.get('parameter_importance'),
            'trial_count': len(results['trials'])
        }

        logger.info(
            f"Hyperparameter Tuning completed in {results['execution_time_ms']:.2f}ms. "
            f"Best CV score: {results['best_score']:.4f}, "
            f"Test score: {results.get('best_estimator_test_score', 0):.4f}, "
            f"Trials: {len(results['trials'])}"
        )

        return HyperparameterTuningResponse(
            success=True,
            best_score=results['best_score'],
            best_params=results['best_params'],
            best_estimator_test_score=results.get('best_estimator_test_score'),
            trials=trials_formatted,
            convergence_data=results['convergence_data'],
            heatmap_data=results.get('heatmap_data'),
            parameter_importance=results.get('parameter_importance'),
            visualization_data=visualization_data,
            execution_time_ms=results['execution_time_ms'],
            parameters_used=results['parameters_used']
        )

    except ValueError as e:
        logger.error(f"Validation error in Hyperparameter Tuning: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        logger.error(f"Missing dependency for Hyperparameter Tuning: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Missing required library. For Bayesian optimization, install: pip install optuna"
        )
    except Exception as e:
        logger.error(f"Hyperparameter Tuning error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Hyperparameter tuning failed: {str(e)}"
        )


@router.get("/hyperparameter-tuning/info")
async def get_hyperparameter_tuning_info() -> Dict[str, Any]:
    """Get Hyperparameter Tuning algorithm information and metadata.

    Returns algorithm description, parameters, complexity, theory, use cases,
    available search methods, supported models, and datasets.

    Returns:
        Dictionary containing algorithm metadata and configuration options

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("hyperparameter-tuning")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Hyperparameter Tuning algorithm not found"
        )

    return {
        "metadata": metadata.model_dump(),
        "available_methods": ["grid", "random", "bayesian", "compare"],
        "available_models": ["random_forest", "svm", "xgboost", "mlp"],
        "available_datasets": ["iris", "wine", "breast_cancer"],
        "datasets_info": {
            "iris": get_hp_dataset_info("iris"),
            "wine": get_hp_dataset_info("wine"),
            "breast_cancer": get_hp_dataset_info("breast_cancer")
        },
        "methods_info": {
            "grid": {
                "name": "Grid Search",
                "description": "Exhaustive search over all parameter combinations",
                "best_for": "Small parameter spaces, thorough exploration",
                "time_complexity": "O(n^k) where n=values per param, k=num params"
            },
            "random": {
                "name": "Random Search",
                "description": "Random sampling from parameter distributions",
                "best_for": "Large parameter spaces, faster results",
                "time_complexity": "O(n_trials * model_training)"
            },
            "bayesian": {
                "name": "Bayesian Optimization",
                "description": "Intelligent search using probabilistic model",
                "best_for": "Expensive models, sample efficiency",
                "time_complexity": "O(n_trials * model_training)",
                "requires": "optuna library"
            },
            "compare": {
                "name": "Compare All Methods",
                "description": "Runs all methods and compares results",
                "best_for": "Understanding trade-offs between methods"
            }
        },
        "model_parameter_spaces": {
            "random_forest": {
                "n_estimators": "Number of trees (50-300)",
                "max_depth": "Maximum tree depth (5-30 or None)",
                "min_samples_split": "Min samples to split node (2-10)",
                "min_samples_leaf": "Min samples at leaf (1-4)",
                "max_features": "Features per split (sqrt, log2)"
            },
            "svm": {
                "C": "Regularization parameter (0.1-100)",
                "kernel": "Kernel type (linear, rbf, poly)",
                "gamma": "Kernel coefficient (scale, auto, numeric)",
                "degree": "Polynomial degree (2-4, poly only)"
            },
            "xgboost": {
                "n_estimators": "Number of boosting rounds (50-200)",
                "max_depth": "Maximum tree depth (3-9)",
                "learning_rate": "Step size shrinkage (0.01-0.3)",
                "subsample": "Sample ratio (0.6-1.0)",
                "colsample_bytree": "Feature ratio per tree (0.6-1.0)"
            },
            "mlp": {
                "hidden_layer_sizes": "Layer architecture (e.g., (50,), (100,50))",
                "activation": "Activation function (relu, tanh)",
                "alpha": "L2 regularization (0.0001-0.01)",
                "learning_rate": "Learning rate schedule (constant, adaptive)"
            }
        }
    }


# Import and register Cross-Validation
from algorithms.ml.cross_validation import (
    train_cross_validation,
    CrossValidationRequest,
    CrossValidationResponse
)
from algorithms.ml.cross_validation.data import get_dataset_info as get_cv_dataset_info


# Register Cross-Validation metadata
cross_validation_metadata = AlgorithmMetadata(
    id="cross-validation",
    name="Cross-Validation",
    slug="cross-validation",
    category=AlgorithmCategory.ML,
    description="Evaluate model performance using k-fold, stratified, and other CV strategies",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["ml", "cross-validation", "model-evaluation", "k-fold", "validation"],
    use_cases=[
        "Model evaluation",
        "Performance estimation",
        "Model selection",
        "Preventing overfitting",
        "Small dataset handling",
        "Robust accuracy measurement"
    ],
    complexity=AlgorithmComplexity(
        time="O(k*model_training)",
        space="O(n/k)"
    ),
    parameters=[
        AlgorithmParameter(
            name="cv_method",
            label="CV Strategy",
            type="select",
            default="k_fold",
            options=[
                {"label": "K-Fold", "value": "k_fold"},
                {"label": "Stratified K-Fold", "value": "stratified"},
                {"label": "Shuffle Split", "value": "shuffle_split"},
                {"label": "Leave-One-Out", "value": "leave_one_out"},
                {"label": "Time Series Split", "value": "time_series"}
            ],
            description="Cross-validation strategy to use"
        ),
        AlgorithmParameter(
            name="n_splits",
            label="Number of Folds",
            type="range",
            default=5,
            min=2,
            max=10,
            step=1,
            description="Number of folds/splits for cross-validation"
        ),
        AlgorithmParameter(
            name="model_type",
            label="Model to Evaluate",
            type="select",
            default="random_forest",
            options=[
                {"label": "Random Forest", "value": "random_forest"},
                {"label": "Logistic Regression", "value": "logistic"},
                {"label": "SVM", "value": "svm"},
                {"label": "K-Nearest Neighbors", "value": "knn"}
            ],
            description="Type of model to evaluate"
        ),
        AlgorithmParameter(
            name="scoring",
            label="Scoring Metric",
            type="select",
            default="accuracy",
            options=[
                {"label": "Accuracy", "value": "accuracy"},
                {"label": "F1 Score", "value": "f1"},
                {"label": "Precision", "value": "precision"},
                {"label": "Recall", "value": "recall"},
                {"label": "ROC AUC", "value": "roc_auc"}
            ],
            description="Metric to use for model evaluation"
        ),
        AlgorithmParameter(
            name="shuffle",
            label="Shuffle Data",
            type="boolean",
            default=True,
            description="Whether to shuffle data before splitting"
        ),
        AlgorithmParameter(
            name="dataset",
            label="Dataset",
            type="select",
            default="iris",
            options=[
                {"label": "Iris", "value": "iris"},
                {"label": "Wine", "value": "wine"},
                {"label": "Breast Cancer", "value": "breast_cancer"}
            ],
            description="Dataset to use for evaluation"
        )
    ],
    dataset_name="iris",
    visualization_type="fold_performance_score_distribution_confusion_matrix",
    theory=(
        "Cross-Validation is a resampling technique used to evaluate machine learning models on "
        "a limited data sample. The technique involves partitioning the data into complementary "
        "subsets, training the model on one subset (training set) and validating on another (test set). "
        "Multiple rounds of cross-validation are performed using different partitions, and the results "
        "are averaged to produce a more reliable estimate of model performance.\n\n"
        "Common strategies include:\n"
        "• K-Fold: Data split into K equal folds, each fold used once as test set\n"
        "• Stratified K-Fold: Preserves class distribution in each fold\n"
        "• Shuffle Split: Random train/test splits, allows overlapping test sets\n"
        "• Leave-One-Out: Each sample used once as test set (K=n)\n"
        "• Time Series Split: Respects temporal order for time series data"
    ),
    pros=[
        "Provides robust estimate of model performance",
        "Uses all data for both training and validation",
        "Reduces overfitting in model selection",
        "Helps detect variance in model performance",
        "Essential for small datasets",
        "Enables model comparison"
    ],
    cons=[
        "Computationally expensive (trains K models)",
        "May be slow for large datasets",
        "Not suitable for time series (except Time Series Split)",
        "Results can vary with random seed",
        "Doesn't provide a final trained model"
    ],
    related_algorithms=["train-test-split", "bootstrap", "holdout-validation"]
)

AlgorithmRegistry.register(cross_validation_metadata)


@router.post("/cross-validation/train", response_model=CrossValidationResponse)
async def train_cross_validation_endpoint(request: CrossValidationRequest) -> CrossValidationResponse:
    """Perform cross-validation on a model.

    This endpoint evaluates a machine learning model using various cross-validation
    strategies, providing comprehensive metrics and visualization data for each fold.

    Args:
        request: CrossValidationRequest with CV parameters

    Returns:
        CrossValidationResponse with fold metrics and aggregate statistics

    Raises:
        HTTPException: If cross-validation fails
    """
    try:
        logger.info(f"Running cross-validation with parameters: {request.model_dump()}")

        # Perform cross-validation
        response = train_cross_validation(request)

        logger.info(
            f"Cross-validation completed in {response.execution_time_ms:.2f}ms. "
            f"Mean {request.scoring}: {response.metrics.mean_score:.4f} ± {response.metrics.std_score:.4f}"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Cross-validation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cross-validation failed: {str(e)}")


@router.get("/cross-validation/info")
async def get_cross_validation_info() -> Dict[str, Any]:
    """Get Cross-Validation algorithm information and metadata.

    Returns algorithm description, parameters, complexity, theory, use cases,
    available strategies, and supported datasets.

    Returns:
        Dictionary containing algorithm metadata and configuration

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("cross-validation")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Cross-Validation algorithm not found"
        )

    return {
        "metadata": metadata.model_dump(),
        "available_datasets": ["iris", "wine", "breast_cancer"],
        "cv_methods": {
            "k_fold": {
                "name": "K-Fold Cross-Validation",
                "description": "Standard k-fold cross-validation where data is split into K equal folds",
                "best_for": "General purpose model evaluation"
            },
            "stratified": {
                "name": "Stratified K-Fold",
                "description": "Preserves the percentage of samples for each class in each fold",
                "best_for": "Classification with imbalanced datasets"
            },
            "shuffle_split": {
                "name": "Shuffle Split",
                "description": "Random permutation cross-validator with independent train/test splits",
                "best_for": "When you want control over train/test size and number of splits"
            },
            "leave_one_out": {
                "name": "Leave-One-Out (LOO)",
                "description": "Each sample is used once as test set while remaining form training set",
                "best_for": "Very small datasets where every sample counts"
            },
            "time_series": {
                "name": "Time Series Split",
                "description": "Respects temporal ordering of data for time series validation",
                "best_for": "Time series data where order matters"
            }
        },
        "model_types": ["random_forest", "logistic", "svm", "knn"],
        "scoring_metrics": ["accuracy", "f1", "precision", "recall", "roc_auc"]
    }


# Import and register Imbalanced Classification
from algorithms.ml.imbalanced_classification import (
    ImbalancedClassificationModel,
    ImbalancedClassificationRequest,
    ImbalancedClassificationResponse
)
from algorithms.ml.imbalanced_classification.data import get_dataset_info as get_imbalanced_dataset_info


# Register Imbalanced Classification metadata
imbalanced_classification_metadata = AlgorithmMetadata(
    id="imbalanced-classification",
    name="Imbalanced Classification",
    slug="imbalanced-classification",
    category=AlgorithmCategory.ML,
    description="Handle class imbalance with SMOTE, class weights, and sampling strategies",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["ml", "imbalanced-data", "smote", "class-weights", "sampling"],
    use_cases=[
        "Fraud detection",
        "Medical diagnosis",
        "Anomaly detection",
        "Rare event prediction",
        "Quality control",
        "Churn prediction"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*k) for SMOTE",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="strategy",
            label="Balance Strategy",
            type="select",
            default="smote",
            options=[
                {"label": "SMOTE (Synthetic Minority Over-sampling)", "value": "smote"},
                {"label": "Random Under-sampling", "value": "undersample"},
                {"label": "Random Over-sampling", "value": "oversample"},
                {"label": "Class Weights", "value": "class_weights"},
                {"label": "Compare All Strategies", "value": "compare"}
            ],
            description="Strategy for handling class imbalance"
        ),
        AlgorithmParameter(
            name="target_ratio",
            label="Target Minority Ratio",
            type="range",
            default=0.5,
            min=0.1,
            max=1.0,
            step=0.1,
            description="Target ratio for minority class after resampling"
        ),
        AlgorithmParameter(
            name="k_neighbors",
            label="SMOTE Neighbors (k)",
            type="range",
            default=5,
            min=3,
            max=10,
            step=1,
            description="Number of nearest neighbors for SMOTE algorithm"
        ),
        AlgorithmParameter(
            name="model_type",
            label="Classifier Type",
            type="select",
            default="random_forest",
            options=[
                {"label": "Random Forest", "value": "random_forest"},
                {"label": "Logistic Regression", "value": "logistic"},
                {"label": "XGBoost", "value": "xgboost"}
            ],
            description="Type of classifier to use for evaluation"
        ),
        AlgorithmParameter(
            name="imbalance_ratio",
            label="Imbalance Ratio (Majority:Minority)",
            type="range",
            default=10,
            min=5,
            max=100,
            step=5,
            description="Ratio of majority to minority class (e.g., 10 means 10:1)"
        )
    ],
    dataset_name="imbalanced_synthetic",
    visualization_type="class_distribution_roc_pr_curves_confusion_matrices",
    theory=(
        "Imbalanced Classification addresses the challenge of learning from datasets where "
        "one class significantly outnumbers the other(s). Standard classifiers often fail on "
        "imbalanced data, achieving high accuracy while performing poorly on the minority class.\n\n"
        "**Techniques:**\n\n"
        "1. **SMOTE (Synthetic Minority Over-sampling Technique)**: Creates synthetic examples "
        "of the minority class by interpolating between existing minority samples and their "
        "k-nearest neighbors. This increases minority class representation without simple duplication.\n\n"
        "2. **Random Under-sampling**: Reduces the majority class by randomly removing samples "
        "until the desired balance is achieved. Fast but may lose important information.\n\n"
        "3. **Random Over-sampling**: Duplicates minority class samples randomly to increase "
        "their representation. Simple but can lead to overfitting.\n\n"
        "4. **Class Weights**: Assigns higher misclassification costs to minority class samples "
        "during training. Does not change the data distribution but adjusts the learning objective.\n\n"
        "The choice of strategy depends on:\n"
        "- Severity of imbalance (mild vs. extreme)\n"
        "- Dataset size (enough data for undersampling?)\n"
        "- Risk of overfitting (oversampling concerns)\n"
        "- Model type (some handle class weights better)\n\n"
        "**Evaluation Metrics:**\n"
        "With imbalanced data, accuracy is misleading. Instead, use:\n"
        "- Precision-Recall curves and PR-AUC\n"
        "- ROC curves and ROC-AUC\n"
        "- F1 score (harmonic mean of precision and recall)\n"
        "- Confusion matrix focusing on minority class metrics"
    ),
    pros=[
        "SMOTE creates synthetic diverse samples, not duplicates",
        "Class weights work without changing data distribution",
        "Undersampling is fast and reduces training time",
        "Multiple strategies for different scenarios",
        "Improves minority class recall significantly",
        "Essential for real-world imbalanced problems"
    ],
    cons=[
        "SMOTE can create unrealistic synthetic samples",
        "Undersampling may lose important information",
        "Oversampling can lead to overfitting",
        "Optimal strategy depends on specific dataset",
        "Requires careful metric selection (not accuracy)",
        "May need to tune target ratio parameter"
    ],
    related_algorithms=["smote", "random-forest", "xgboost", "anomaly-detection"]
)

AlgorithmRegistry.register(imbalanced_classification_metadata)


@router.post("/imbalanced-classification/train", response_model=ImbalancedClassificationResponse)
async def train_imbalanced_classification(request: ImbalancedClassificationRequest):
    """Train model with imbalanced data handling strategies.

    This endpoint demonstrates various techniques for handling class imbalance
    including SMOTE, undersampling, oversampling, and class weights. It can
    compare all strategies to find the best approach for a given imbalance ratio.

    Args:
        request: Training parameters including strategy, target_ratio,
                k_neighbors, model_type, and imbalance_ratio

    Returns:
        Training results including:
        - Performance metrics for each strategy (precision, recall, F1, ROC-AUC, PR-AUC)
        - Class distribution before/after resampling
        - ROC and Precision-Recall curves
        - Confusion matrices
        - Strategy comparison (if compare mode)

    Raises:
        HTTPException: If training fails or imbalanced-learn not installed

    Example:
        >>> request = ImbalancedClassificationRequest(
        ...     strategy="compare",
        ...     target_ratio=0.5,
        ...     k_neighbors=5,
        ...     model_type="random_forest",
        ...     imbalance_ratio=10
        ... )
        >>> response = await train_imbalanced_classification(request)
        >>> print(f"SMOTE F1: {response.metrics.smote.f1_score:.4f}")
        >>> print(f"SMOTE Minority Recall: {response.metrics.smote.minority_recall:.4f}")
    """
    try:
        logger.info(
            f"Training Imbalanced Classification: "
            f"strategy={request.strategy}, ratio={request.imbalance_ratio}:1, "
            f"target={request.target_ratio}, model={request.model_type}"
        )

        start_time = time.time()

        # Prepare imbalanced data
        from algorithms.ml.imbalanced_classification.data import prepare_imbalanced_data
        X_train, X_test, y_train, y_test, metadata = prepare_imbalanced_data(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            imbalance_ratio=request.imbalance_ratio,
            test_size=request.test_size,
            random_state=request.random_state
        )

        # Calculate original class distribution
        from algorithms.ml.imbalanced_classification.data import calculate_class_distribution
        original_dist = calculate_class_distribution(y_train)

        # Create and train model
        model = ImbalancedClassificationModel(
            model_type=request.model_type,
            random_state=request.random_state
        )

        # Train based on strategy
        if request.strategy == 'compare':
            # Compare all strategies
            results = model.train_compare_all(
                X_train, y_train, X_test, y_test,
                target_ratio=request.target_ratio,
                k_neighbors=request.k_neighbors
            )

            # Build comprehensive response
            from algorithms.ml.imbalanced_classification.schema import (
                ImbalancedClassificationMetrics,
                ClassDistribution,
                VisualizationData,
                ROCCurveData,
                PRCurveData
            )

            metrics_obj = ImbalancedClassificationMetrics(
                original=results['metrics']['original'],
                smote=results['metrics']['smote'],
                undersample=results['metrics']['undersample'],
                oversample=results['metrics']['oversample'],
                class_weights=results['metrics']['class_weights']
            )

            # Prepare ROC curves
            roc_curves = {}
            for strategy_name, roc_data in results['roc_curves'].items():
                roc_curves[strategy_name] = ROCCurveData(**roc_data)

            # Prepare PR curves
            pr_curves = {}
            for strategy_name, pr_data in results['pr_curves'].items():
                pr_curves[strategy_name] = PRCurveData(**pr_data)

            # Prepare metrics comparison
            metrics_comparison = {}
            for strategy_name, metrics in results['metrics'].items():
                metrics_comparison[strategy_name] = {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'roc_auc': metrics.roc_auc,
                    'pr_auc': metrics.pr_auc,
                    'minority_recall': metrics.minority_recall,
                    'minority_precision': metrics.minority_precision
                }

            visualization_data = VisualizationData(
                original_distribution=ClassDistribution(**original_dist),
                roc_curves=roc_curves,
                pr_curves=pr_curves,
                metrics_comparison=metrics_comparison,
                sample_counts=results['sample_counts']
            )

        else:
            # Single strategy
            metrics_single, roc_data, pr_data = model.train_single_strategy(
                X_train, y_train, X_test, y_test,
                request.strategy,
                target_ratio=request.target_ratio,
                k_neighbors=request.k_neighbors
            )

            # Calculate resampled distribution for visualization
            if request.strategy == 'smote':
                X_res, y_res = model._apply_smote(
                    X_train, y_train,
                    target_ratio=request.target_ratio,
                    k_neighbors=request.k_neighbors
                )
                resampled_dist = calculate_class_distribution(y_res)
            elif request.strategy == 'undersample':
                X_res, y_res = model._apply_undersampling(X_train, y_train, request.target_ratio)
                resampled_dist = calculate_class_distribution(y_res)
            elif request.strategy == 'oversample':
                X_res, y_res = model._apply_oversampling(X_train, y_train, request.target_ratio)
                resampled_dist = calculate_class_distribution(y_res)
            else:
                resampled_dist = original_dist

            from algorithms.ml.imbalanced_classification.schema import (
                ImbalancedClassificationMetrics,
                ClassDistribution,
                VisualizationData,
                ROCCurveData,
                PRCurveData
            )

            metrics_obj = ImbalancedClassificationMetrics(
                original=metrics_single,
                **{request.strategy: metrics_single}
            )

            visualization_data = VisualizationData(
                original_distribution=ClassDistribution(**original_dist),
                resampled_distribution=ClassDistribution(**resampled_dist),
                roc_curves={request.strategy: ROCCurveData(**roc_data)},
                pr_curves={request.strategy: PRCurveData(**pr_data)},
                metrics_comparison={request.strategy: {
                    'accuracy': metrics_single.accuracy,
                    'precision': metrics_single.precision,
                    'recall': metrics_single.recall,
                    'f1_score': metrics_single.f1_score,
                    'roc_auc': metrics_single.roc_auc,
                    'pr_auc': metrics_single.pr_auc,
                    'minority_recall': metrics_single.minority_recall,
                    'minority_precision': metrics_single.minority_precision
                }},
                sample_counts={request.strategy: resampled_dist}
            )

            results = {'predictions': {request.strategy: []}}

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Imbalanced Classification completed in {execution_time_ms:.2f}ms. "
            f"Original imbalance: {original_dist['ratio']:.1f}:1"
        )

        if request.strategy == 'compare':
            for strategy_name in ['smote', 'undersample', 'oversample', 'class_weights']:
                strategy_metrics = getattr(metrics_obj, strategy_name)
                if strategy_metrics:
                    logger.info(
                        f"  {strategy_name}: F1={strategy_metrics.f1_score:.4f}, "
                        f"Minority Recall={strategy_metrics.minority_recall:.4f}, "
                        f"ROC-AUC={strategy_metrics.roc_auc:.4f}"
                    )

        return ImbalancedClassificationResponse(
            success=True,
            metrics=metrics_obj,
            predictions=results.get('predictions', {}),
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            parameters_used={
                'strategy': request.strategy,
                'target_ratio': request.target_ratio,
                'k_neighbors': request.k_neighbors,
                'model_type': request.model_type,
                'imbalance_ratio': request.imbalance_ratio,
                'test_size': request.test_size,
                'random_state': request.random_state
            }
        )

    except ImportError as e:
        logger.error(f"Missing dependency for Imbalanced Classification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="imbalanced-learn is required. Install with: pip install imbalanced-learn"
        )
    except ValueError as e:
        logger.error(f"Validation error in Imbalanced Classification: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Imbalanced Classification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Imbalanced classification failed: {str(e)}"
        )


@router.get("/imbalanced-classification/info")
async def get_imbalanced_classification_info() -> Dict[str, Any]:
    """Get Imbalanced Classification algorithm information and metadata.

    Returns algorithm description, parameters, complexity, theory, use cases,
    available strategies, and visualization examples.

    Returns:
        Dictionary containing algorithm metadata and configuration

    Raises:
        HTTPException: If metadata not found
    """
    metadata = AlgorithmRegistry.get("imbalanced-classification")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Imbalanced Classification algorithm not found"
        )

    dataset_info = get_imbalanced_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "strategies": {
            "smote": {
                "name": "SMOTE",
                "full_name": "Synthetic Minority Over-sampling Technique",
                "description": "Creates synthetic minority samples by interpolation",
                "pros": ["Diverse synthetic samples", "Reduces overfitting vs duplication"],
                "cons": ["May create unrealistic samples", "Slower than simple oversampling"],
                "best_for": "Moderate to severe imbalance with sufficient minority samples"
            },
            "undersample": {
                "name": "Random Under-sampling",
                "description": "Randomly removes majority class samples",
                "pros": ["Fast", "Reduces training time", "Balances classes directly"],
                "cons": ["Loses information", "May remove important samples"],
                "best_for": "Large datasets where data loss is acceptable"
            },
            "oversample": {
                "name": "Random Over-sampling",
                "description": "Duplicates minority class samples randomly",
                "pros": ["Simple", "Fast", "No information loss"],
                "cons": ["Exact duplicates", "Overfitting risk"],
                "best_for": "Quick baseline or very small minority class"
            },
            "class_weights": {
                "name": "Class Weights",
                "description": "Adjusts misclassification costs during training",
                "pros": ["No data modification", "Works with any model", "Fast"],
                "cons": ["Less direct control", "Effectiveness varies by model"],
                "best_for": "When you want to preserve original data distribution"
            }
        },
        "metrics_guidance": {
            "accuracy": "Misleading with imbalanced data - avoid as primary metric",
            "precision": "How many predicted positives are actually positive",
            "recall": "How many actual positives were caught (crucial for minority class)",
            "f1_score": "Harmonic mean balancing precision and recall",
            "roc_auc": "Area under ROC curve - overall discriminative ability",
            "pr_auc": "Area under PR curve - better for severe imbalance"
        },
        "recommended_workflow": [
            "1. Start with 'compare' strategy to evaluate all approaches",
            "2. Focus on minority class recall and PR-AUC for severe imbalance",
            "3. Use confusion matrix to understand false positive/negative trade-offs",
            "4. Consider domain costs (e.g., missing fraud vs false alarm)",
            "5. Fine-tune target_ratio based on requirements"
        ]
    }
