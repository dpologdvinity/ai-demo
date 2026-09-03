from fastapi import APIRouter, HTTPException
import time
from typing import Dict, Any
import logging

from algorithms.deep_learning.cnn import CNNModel, CNNRequest, CNNResponse
from algorithms.deep_learning.cnn.data import (
    load_digits_data,
    get_dataset_info,
    prepare_visualization_data
)
from algorithms.deep_learning.rnn import (
    RNNModel,
    RNNRequest,
    RNNResponse,
    generate_sine_wave_data,
    TimeSeriesDataset,
    get_dataset_info as get_rnn_dataset_info
)
from algorithms.deep_learning.lstm import LSTMModel, LSTMRequest, LSTMResponse
from algorithms.deep_learning.lstm.data import get_dataset_info as get_lstm_dataset_info
from algorithms.deep_learning.feedforward_nn import MLPModel, MLPRequest, MLPResponse
from algorithms.deep_learning.feedforward_nn.data import (
    load_iris_data,
    get_dataset_info as get_mlp_dataset_info,
    prepare_visualization_data as prepare_mlp_visualization_data
)
from algorithms.deep_learning.gan import (
    GANModel,
    GANRequest,
    GANResponse,
    load_digits_data as load_gan_digits_data,
    get_dataset_info as get_gan_dataset_info,
    prepare_data_for_gan
)
from algorithms.deep_learning.vae import VAEModel, VAERequest, VAEResponse
from algorithms.deep_learning.vae.data import load_vae_data, get_dataset_info as get_vae_dataset_info
from algorithms.deep_learning.autoencoder import (
    AutoencoderModel,
    AutoencoderRequest,
    AutoencoderResponse,
    load_mnist_data,
    get_dataset_info as get_autoencoder_dataset_info,
    compute_latent_visualization,
    prepare_sample_comparison
)
from algorithms.deep_learning.autoencoder_variants import (
    AutoencoderVariantsModel,
    AutoencoderVariantsRequest,
    AutoencoderVariantsResponse,
    load_mnist_data as load_autoencoder_variants_data,
    get_dataset_info as get_autoencoder_variants_dataset_info,
    compute_latent_visualization as compute_autoencoder_variants_latent_viz,
    prepare_variant_comparison,
    add_noise
)
from algorithms.deep_learning.gru import GRUModel, GRURequest, GRUResponse
from algorithms.deep_learning.gru.data import get_dataset_info as get_gru_dataset_info
from algorithms.deep_learning.transformer import (
    TransformerModel,
    TransformerRequest,
    TransformerResponse,
    generate_reversal_data,
    SequenceDataset,
    get_dataset_info as get_transformer_dataset_info
)
from algorithms.deep_learning.resnet import ResNetModel, ResNetRequest, ResNetResponse
from algorithms.deep_learning.resnet.data import get_dataset_info as get_resnet_dataset_info
from algorithms.deep_learning.dropout import DropoutModel, DropoutRequest, DropoutResponse
from algorithms.deep_learning.dropout.data import (
    generate_overfitting_prone_dataset,
    get_dataset_info as get_dropout_dataset_info
)
from algorithms.deep_learning.vgg import VGGModel, VGGRequest, VGGResponse
from algorithms.deep_learning.vgg.data import get_dataset_info as get_vgg_dataset_info
from algorithms.deep_learning.activation_functions import (
    ActivationFunctionsModel,
    ActivationFunctionsRequest,
    ActivationFunctionsResponse
)
from algorithms.deep_learning.activation_functions.data import get_dataset_info as get_activation_functions_dataset_info
from algorithms.deep_learning.activation_functions.model import compute_activation_functions
from algorithms.deep_learning.batch_normalization import (
    BatchNormModel,
    BatchNormRequest,
    BatchNormResponse,
    get_dataset_info as get_batch_norm_dataset_info
)
from algorithms.deep_learning.convolutional_layers import (
    ConvolutionalLayersModel,
    ConvolutionalLayersRequest,
    ConvolutionalLayersResponse
)
from algorithms.deep_learning.convolutional_layers.data import get_dataset_info as get_conv_layers_dataset_info
from algorithms.deep_learning.pooling_layers import (
    PoolingLayersModel,
    PoolingLayersRequest,
    PoolingLayersResponse
)
from algorithms.deep_learning.pooling_layers.data import get_dataset_info as get_pooling_dataset_info
from algorithms.deep_learning.pooling_layers.model import compute_pooling
from algorithms.deep_learning.adam_optimizer import (
    AdamOptimizerModel,
    AdamOptimizerRequest,
    AdamOptimizerResponse,
    get_dataset_info as get_adam_optimizer_dataset_info
)
from algorithms.deep_learning.gradient_descent import (
    GradientDescentModel,
    GradientDescentRequest,
    GradientDescentResponse
)
from algorithms.deep_learning.gradient_descent.data import get_dataset_info as get_gradient_descent_dataset_info
from algorithms.deep_learning.data_augmentation import (
    DataAugmentationModel,
    DataAugmentationRequest,
    DataAugmentationResponse,
    get_dataset_info as get_data_augmentation_dataset_info
)
from algorithms.deep_learning.learning_rate_scheduling import (
    LearningRateSchedulingModel,
    LearningRateSchedulingRequest,
    LearningRateSchedulingResponse,
    get_dataset_info as get_lr_scheduling_dataset_info,
    get_model_info as get_lr_scheduling_model_info
)
from algorithms.deep_learning.transfer_learning import (
    run_transfer_learning,
    TransferLearningRequest,
    TransferLearningResponse,
    get_dataset_info as get_transfer_learning_dataset_info
)
import torch
from torch.utils.data import DataLoader
from utils.algorithm_metadata import (
    AlgorithmMetadata,
    AlgorithmParameter,
    AlgorithmComplexity,
    AlgorithmCategory,
    DifficultyLevel,
    AlgorithmRegistry,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/deep-learning", tags=["Deep Learning"])


# Register CNN metadata
cnn_metadata = AlgorithmMetadata(
    id="cnn",
    name="Convolutional Neural Network",
    slug="cnn",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Deep learning architecture with convolutional layers for image processing",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "cnn", "computer-vision", "convolutional"],
    use_cases=[
        "Image classification",
        "Object recognition",
        "Feature extraction",
        "Pattern detection"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*f*k²*c*h*w*epochs)",
        space="O(layers*filters*h*w)"
    ),
    parameters=[
        AlgorithmParameter(
            name="conv_filters",
            label="Convolutional Filters",
            type="text",
            default=[16, 32],
            description="Number of filters per convolutional layer (comma-separated, e.g., '16,32')"
        ),
        AlgorithmParameter(
            name="kernel_size",
            label="Kernel Size",
            type="range",
            default=3,
            min=3,
            max=7,
            step=2,
            description="Size of convolution kernel (odd numbers only)"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for Adam optimizer"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=15,
            min=5,
            max=50,
            step=5,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="dropout",
            label="Dropout Rate",
            type="range",
            default=0.5,
            min=0.0,
            max=0.8,
            step=0.1,
            description="Dropout rate for regularization"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=32,
            options=[
                {"label": "8", "value": 8},
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128}
            ],
            description="Batch size for training"
        )
    ],
    dataset_name="digits",
    visualization_type="network_architecture,feature_maps,training_curves,confusion_matrix",
    theory=(
        "Convolutional Neural Networks (CNNs) are specialized deep learning architectures "
        "designed for processing grid-like data such as images. CNNs use convolutional layers "
        "that apply learnable filters across the input to detect local patterns and features. "
        "Key components include:\n\n"
        "1. Convolutional Layers: Apply filters to extract features like edges, textures, and patterns\n"
        "2. Pooling Layers: Downsample feature maps to reduce dimensionality and computation\n"
        "3. Batch Normalization: Normalizes activations to improve training stability\n"
        "4. Fully Connected Layers: Combine extracted features for final classification\n\n"
        "CNNs achieve translation invariance through weight sharing and hierarchical feature learning, "
        "making them highly effective for computer vision tasks. The network learns increasingly "
        "complex features from low-level edges to high-level semantic patterns through multiple layers."
    ),
    pros=[
        "Automatic feature extraction without manual engineering",
        "Translation invariant through weight sharing",
        "Hierarchical learning of increasingly complex features",
        "Reduced parameters compared to fully connected networks",
        "State-of-the-art performance on image tasks"
    ],
    cons=[
        "Requires large amounts of training data",
        "Computationally expensive to train",
        "Requires GPU for efficient training",
        "Less interpretable than traditional methods",
        "Sensitive to hyperparameter choices"
    ],
    related_algorithms=["mlp", "resnet", "vgg", "alexnet"]
)

AlgorithmRegistry.register(cnn_metadata)


# Register RNN metadata
rnn_metadata = AlgorithmMetadata(
    id="rnn",
    name="Recurrent Neural Network",
    slug="rnn",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Neural network with recurrent connections for sequence processing and time series prediction",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "rnn", "sequence-modeling", "time-series"],
    use_cases=[
        "Time series forecasting",
        "Sequence generation",
        "Pattern prediction",
        "Signal processing"
    ],
    complexity=AlgorithmComplexity(
        time="O(seq_len × batch × hidden² × epochs)",
        space="O(seq_len × batch × hidden)"
    ),
    parameters=[
        AlgorithmParameter(
            name="hidden_size",
            label="Hidden Size",
            type="range",
            default=64,
            min=32,
            max=256,
            step=16,
            description="Dimension of the hidden state in the RNN"
        ),
        AlgorithmParameter(
            name="num_layers",
            label="Number of Layers",
            type="range",
            default=2,
            min=1,
            max=5,
            step=1,
            description="Number of stacked RNN layers"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for the optimizer"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="sequence_length",
            label="Sequence Length",
            type="range",
            default=20,
            min=5,
            max=50,
            step=5,
            description="Length of input sequences for training"
        ),
        AlgorithmParameter(
            name="prediction_length",
            label="Prediction Length",
            type="range",
            default=10,
            min=1,
            max=20,
            step=1,
            description="Number of future steps to predict"
        ),
    ],
    dataset_name="sine_wave",
    visualization_type="training_loss,prediction_chart,hidden_states",
    theory=(
        "Recurrent Neural Networks (RNNs) are a class of neural networks designed for "
        "sequential data processing. Unlike feedforward networks, RNNs have recurrent "
        "connections that allow information to persist across time steps. Key concepts:\n\n"
        "1. Hidden State: Maintains a memory of previous inputs\n"
        "2. Temporal Dependencies: Captures patterns across time\n"
        "3. Weight Sharing: Same parameters applied at each time step\n"
        "4. Backpropagation Through Time (BPTT): Training algorithm that unfolds the network\n\n"
        "RNNs process sequences by maintaining a hidden state that is updated at each time "
        "step based on the current input and previous hidden state: h_t = f(W_h * h_{t-1} + W_x * x_t + b). "
        "This architecture makes RNNs suitable for time series prediction, sequence modeling, "
        "and tasks requiring temporal context."
    ),
    pros=[
        "Can process variable-length sequences",
        "Maintains temporal context through hidden state",
        "Parameter sharing across time steps",
        "Suitable for time series and sequential data",
        "Foundation for more advanced architectures (LSTM, GRU)"
    ],
    cons=[
        "Prone to vanishing/exploding gradients",
        "Difficult to capture long-term dependencies",
        "Sequential processing limits parallelization",
        "Computationally expensive for long sequences",
        "Often outperformed by LSTM/GRU for complex tasks"
    ],
    related_algorithms=["lstm", "gru", "transformer", "seq2seq"]
)

AlgorithmRegistry.register(rnn_metadata)


# Register LSTM metadata
lstm_metadata = AlgorithmMetadata(
    id="lstm",
    name="LSTM",
    slug="lstm",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="RNN variant with gates to handle long-term dependencies",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "lstm", "sequence-modeling", "time-series", "gated-networks"],
    use_cases=[
        "Language modeling",
        "Speech recognition",
        "Long sequence prediction",
        "Stock price forecasting"
    ],
    complexity=AlgorithmComplexity(
        time="O(seq_len*batch*4*hidden²*epochs)",
        space="O(seq_len*batch*hidden)"
    ),
    parameters=[
        AlgorithmParameter(
            name="hidden_size",
            label="Hidden/Cell State Size",
            type="range",
            default=128,
            min=64,
            max=512,
            step=32,
            description="Size of hidden state and cell state vectors"
        ),
        AlgorithmParameter(
            name="num_layers",
            label="Number of LSTM Layers",
            type="number",
            default=2,
            min=1,
            max=4,
            step=1,
            description="Number of stacked LSTM layers"
        ),
        AlgorithmParameter(
            name="dropout",
            label="Dropout Rate",
            type="range",
            default=0.2,
            min=0.0,
            max=0.5,
            step=0.1,
            description="Dropout probability between LSTM layers"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for Adam optimizer"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="sequence_length",
            label="Sequence Length",
            type="range",
            default=30,
            min=10,
            max=100,
            step=5,
            description="Length of input sequences"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=32,
            options=[
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128}
            ],
            description="Training batch size"
        )
    ],
    dataset_name="synthetic_time_series",
    visualization_type="time_series_gates",
    theory=(
        "Long Short-Term Memory (LSTM) networks are a special kind of Recurrent Neural Network (RNN) "
        "capable of learning long-term dependencies. Unlike standard RNNs that struggle with vanishing "
        "gradients, LSTMs use a sophisticated gating mechanism to regulate information flow. "
        "\n\n"
        "The LSTM cell contains three gates:\n"
        "1. Forget Gate: Decides what information to discard from the cell state\n"
        "2. Input Gate: Decides which new information to store in the cell state\n"
        "3. Output Gate: Decides what information to output based on the cell state\n"
        "\n\n"
        "Additionally, the LSTM maintains two states:\n"
        "- Cell State: The long-term memory that flows through the network\n"
        "- Hidden State: The short-term memory passed to the next time step\n"
        "\n\n"
        "This architecture allows LSTMs to learn which information is important to keep or forget, "
        "making them highly effective for sequence modeling tasks where long-term context matters."
    ),
    pros=[
        "Handles long-term dependencies effectively",
        "Avoids vanishing gradient problem through gating",
        "Can learn to remember or forget information selectively",
        "Excellent for variable-length sequences",
        "Works well for time series and sequential data"
    ],
    cons=[
        "Computationally expensive (4x more parameters than simple RNN)",
        "Slower to train than simpler models",
        "Requires more data to train effectively",
        "Can still struggle with very long sequences (100+ steps)",
        "More complex architecture leads to harder interpretability"
    ],
    related_algorithms=["gru", "rnn", "transformer", "attention"]
)

AlgorithmRegistry.register(lstm_metadata)


# Register GRU metadata
gru_metadata = AlgorithmMetadata(
    id="gru",
    name="GRU (Gated Recurrent Unit)",
    slug="gru",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Recurrent neural network with gating mechanisms for sequence modeling",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["deep-learning", "rnn", "sequence-modeling", "gating"],
    use_cases=[
        "Time series forecasting",
        "Natural language processing",
        "Speech recognition",
        "Machine translation",
        "Video analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(T*n*h²) where T=sequence length, n=samples, h=hidden size",
        space="O(h²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="hidden_size",
            label="Hidden Layer Size",
            type="range",
            default=64,
            min=32,
            max=256,
            step=16,
            description="Size of the hidden state in the GRU"
        ),
        AlgorithmParameter(
            name="num_layers",
            label="Number of Layers",
            type="range",
            default=1,
            min=1,
            max=3,
            step=1,
            description="Number of stacked GRU layers"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for the optimizer"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="sequence_length",
            label="Sequence Length",
            type="range",
            default=20,
            min=5,
            max=50,
            step=5,
            description="Length of input sequences for training"
        ),
        AlgorithmParameter(
            name="dropout",
            label="Dropout Rate",
            type="range",
            default=0.0,
            min=0.0,
            max=0.5,
            step=0.1,
            description="Dropout probability between layers"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=32,
            options=[
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128}
            ],
            description="Training batch size"
        )
    ],
    dataset_name="sine_wave",
    visualization_type="training_loss,prediction_chart,gate_activations,sequence_comparison",
    theory=(
        "Gated Recurrent Units (GRUs) are a variant of recurrent neural networks that use "
        "gating mechanisms to control the flow of information. GRUs address the vanishing "
        "gradient problem while being simpler than LSTMs. The GRU architecture contains:\n\n"
        "1. Update Gate (z): Controls how much of the previous hidden state to retain\n"
        "2. Reset Gate (r): Decides how much of the past information to forget\n"
        "3. Candidate Hidden State: Computed using the reset gate\n"
        "4. Final Hidden State: Combines previous and candidate states using the update gate\n\n"
        "The key equations are:\n"
        "- z_t = σ(W_z · [h_{t-1}, x_t])  (update gate)\n"
        "- r_t = σ(W_r · [h_{t-1}, x_t])  (reset gate)\n"
        "- h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])  (candidate)\n"
        "- h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  (final state)\n\n"
        "GRUs have fewer parameters than LSTMs (no separate cell state) while maintaining "
        "similar performance on many tasks. They are particularly effective for sequence "
        "modeling tasks where computational efficiency is important."
    ),
    pros=[
        "Fewer parameters than LSTM, faster to train",
        "Handles vanishing gradient problem effectively",
        "Simpler architecture than LSTM, easier to implement",
        "Good performance on many sequence tasks",
        "Works well with limited training data"
    ],
    cons=[
        "May underperform LSTM on tasks requiring precise timing",
        "Less control over memory compared to LSTM",
        "Still sequential, cannot be fully parallelized",
        "May struggle with very long sequences (100+ steps)",
        "Gate activations can be difficult to interpret"
    ],
    related_algorithms=["lstm", "rnn", "transformer", "attention"]
)

AlgorithmRegistry.register(gru_metadata)


@router.get("/")
async def deep_learning_root():
    return {"message": "Deep Learning algorithms endpoint"}


@router.get("/algorithms")
async def list_deep_learning_algorithms():
    """Get all registered Deep Learning algorithms.

    Returns:
        List of algorithm metadata for all registered Deep Learning algorithms
    """
    algorithms = AlgorithmRegistry.get_by_category(AlgorithmCategory.DEEP_LEARNING)
    return [algo.model_dump() for algo in algorithms]


@router.post("/cnn/train", response_model=CNNResponse)
async def train_cnn(request: CNNRequest) -> CNNResponse:
    """Train a Convolutional Neural Network on the digits dataset.

    This endpoint trains a CNN with configurable architecture on handwritten
    digit images, returning metrics, predictions, feature maps, and training curves.

    Args:
        request: CNN training parameters including conv_filters, kernel_size,
                learning_rate, epochs, dropout, and batch_size

    Returns:
        CNNResponse with training results, metrics, feature maps, and visualization data

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training CNN with parameters: {request.model_dump()}")

        # Load data
        data = load_digits_data(
            test_size=0.2,
            random_state=request.random_state,
            normalize=True
        )

        # Initialize and train model
        model = CNNModel(
            conv_filters=request.conv_filters,
            kernel_size=request.kernel_size,
            learning_rate=request.learning_rate,
            dropout=request.dropout
        )

        training_info = model.train(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            epochs=request.epochs,
            batch_size=request.batch_size
        )

        # Make predictions
        predictions = model.predict(data['X_test'])

        # Evaluate model
        metrics = model.evaluate(data['X_test'], data['y_test'])

        # Extract feature maps for visualization
        feature_maps = model.extract_feature_maps(data['X_test'], num_samples=3)

        # Prepare visualization data
        viz_data = prepare_visualization_data(
            X_sample=data['X_test'][:5],
            y_sample=data['y_test'][:5],
            feature_maps=feature_maps
        )

        # Add training curves to visualization
        viz_data['training_curves'] = {
            'epochs': list(range(1, request.epochs + 1)),
            'train_loss': model.training_history['train_loss'],
            'train_accuracy': model.training_history['train_accuracy'],
            'val_loss': model.training_history['val_loss'],
            'val_accuracy': model.training_history['val_accuracy']
        }

        # Add confusion matrix to visualization
        viz_data['confusion_matrix'] = metrics['confusion_matrix']

        # Add network architecture info
        model_info = model.get_model_info()
        viz_data['network_architecture'] = {
            'layers': [
                {'type': 'Input', 'shape': '(1, 8, 8)'},
            ],
            'total_parameters': model_info['total_parameters']
        }

        # Add conv layers
        for i, filters in enumerate(request.conv_filters):
            viz_data['network_architecture']['layers'].extend([
                {'type': f'Conv2d_{i+1}', 'filters': filters, 'kernel': request.kernel_size},
                {'type': f'BatchNorm_{i+1}', 'filters': filters},
                {'type': f'ReLU_{i+1}', 'activation': 'relu'},
                {'type': f'MaxPool_{i+1}', 'pool_size': 2}
            ])

        # Add FC layers
        viz_data['network_architecture']['layers'].extend([
            {'type': 'Flatten'},
            {'type': 'Linear_1', 'units': 128},
            {'type': 'ReLU', 'activation': 'relu'},
            {'type': 'Dropout_1', 'rate': request.dropout},
            {'type': 'Linear_2', 'units': 64},
            {'type': 'ReLU', 'activation': 'relu'},
            {'type': 'Dropout_2', 'rate': request.dropout},
            {'type': 'Output', 'units': 10}
        ])

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"CNN training completed in {execution_time_ms:.2f}ms "
            f"with accuracy: {metrics['accuracy']:.4f}"
        )

        return CNNResponse(
            success=True,
            metrics=metrics,
            predictions=predictions.tolist(),
            actual=data['y_test'].tolist(),
            training_history=model.training_history,
            feature_maps=feature_maps[:, :8].tolist() if feature_maps is not None else None,
            visualization_data=viz_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'conv_filters': request.conv_filters,
                'kernel_size': request.kernel_size,
                'learning_rate': request.learning_rate,
                'epochs': request.epochs,
                'dropout': request.dropout,
                'batch_size': request.batch_size,
                'dataset_name': request.dataset_name
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/cnn/info")
async def get_cnn_info() -> Dict[str, Any]:
    """Get CNN algorithm information and metadata.

    Returns metadata, parameters, and dataset information for CNN.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("cnn")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="CNN metadata not found"
        )

    dataset_info = get_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/rnn/train", response_model=RNNResponse)
async def train_rnn(request: RNNRequest) -> RNNResponse:
    """Train RNN model on synthetic sine wave data.

    This endpoint trains a recurrent neural network to predict future values
    in a time series. It uses synthetic sine wave data with noise for demonstration.

    Args:
        request: RNN training configuration

    Returns:
        RNNResponse containing training results, predictions, and visualizations

    Raises:
        HTTPException: If training fails
    """
    start_time = time.time()

    try:
        logger.info(f"Starting RNN training with parameters: {request.model_dump()}")

        # Generate synthetic data
        data_dict = generate_sine_wave_data(
            n_samples=1000,
            sequence_length=request.sequence_length,
            prediction_length=request.prediction_length,
            noise_level=0.1,
            random_state=42
        )

        sequences = data_dict["sequences"]
        targets = data_dict["targets"]
        train_indices = data_dict["train_indices"]
        test_indices = data_dict["test_indices"]

        # Create datasets
        train_dataset = TimeSeriesDataset(
            sequences[train_indices],
            targets[train_indices]
        )
        test_dataset = TimeSeriesDataset(
            sequences[test_indices],
            targets[test_indices]
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model
        model = RNNModel(
            input_size=1,
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            output_size=1,
            learning_rate=request.learning_rate
        )

        # Train model
        training_results = model.train(
            train_loader=train_loader,
            epochs=request.epochs,
            prediction_length=request.prediction_length
        )

        # Evaluate on test set
        test_metrics = model.evaluate(
            test_loader=test_loader,
            prediction_length=request.prediction_length
        )

        # Get predictions for visualization (first 5 test samples)
        test_sequences = test_dataset.sequences[:5]
        test_targets = test_dataset.targets[:5]

        predictions, hidden_states = model.predict(
            test_sequences,
            prediction_length=request.prediction_length
        )

        # Convert to numpy for response
        predictions_np = predictions.cpu().numpy()
        test_targets_np = test_targets.cpu().numpy()
        test_sequences_np = test_sequences.cpu().numpy()

        # Prepare visualization data
        visualization_data = {
            "training_loss": [
                {"epoch": i + 1, "loss": loss}
                for i, loss in enumerate(training_results["training_history"])
            ],
            "predictions_chart": []
        }

        # Format predictions chart data (first sample)
        for i in range(request.prediction_length):
            visualization_data["predictions_chart"].append({
                "step": i,
                "predicted": float(predictions_np[0, i, 0]),
                "actual": float(test_targets_np[0, i, 0])
            })

        # Add input sequence to visualization
        input_sequence_chart = [
            {"step": i, "value": float(test_sequences_np[0, i, 0])}
            for i in range(request.sequence_length)
        ]
        visualization_data["input_sequence"] = input_sequence_chart

        # Format hidden states for visualization (last layer, first sample)
        if hidden_states:
            last_hidden = hidden_states[-1][0].numpy()  # Last prediction, first sample
            visualization_data["hidden_states"] = {
                "data": last_hidden.tolist(),
                "shape": list(last_hidden.shape)
            }

        # Calculate final metrics
        metrics = {
            "final_loss": training_results["final_loss"],
            "test_loss": test_metrics["loss"],
            "mse": test_metrics["mse"],
            "mae": test_metrics["mae"]
        }

        # Get model info
        model_info = model.get_model_info()

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(f"RNN training completed successfully in {execution_time_ms:.2f}ms")

        return RNNResponse(
            success=True,
            metrics=metrics,
            training_history=training_results["training_history"],
            predictions=predictions_np[:, :, 0].tolist(),
            actual=test_targets_np[:, :, 0].tolist(),
            input_sequences=test_sequences_np[:, :, 0].tolist(),
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used=request.model_dump()
        )

    except Exception as e:
        logger.error(f"Error training RNN: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to train RNN: {str(e)}"
        )


@router.get("/rnn/info")
async def get_rnn_info() -> Dict[str, Any]:
    """Get information about the RNN algorithm and dataset.

    Returns:
        Dictionary containing algorithm metadata and dataset information
    """
    metadata = AlgorithmRegistry.get("rnn")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="RNN metadata not found"
        )

    dataset_info = get_rnn_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register MLP (Feedforward Neural Network) metadata
mlp_metadata = AlgorithmMetadata(
    id="feedforward-nn",
    name="Feedforward Neural Network (MLP)",
    slug="feedforward-nn",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Basic neural network with fully connected layers for classification",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "neural-networks", "supervised", "classification", "mlp"],
    use_cases=[
        "Classification tasks",
        "Pattern recognition",
        "Function approximation",
        "Feature learning"
    ],
    complexity=AlgorithmComplexity(
        time="O(L*n*m²)",
        space="O(L*m²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="hidden_layers",
            label="Hidden Layer Sizes",
            type="text",
            default=[64, 32],
            description="Sizes of hidden layers (comma-separated, e.g., '64,32' for two layers)"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.1,
            step=0.0001,
            description="Learning rate for optimizer"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=100,
            min=10,
            max=500,
            step=10,
            description="Maximum number of training iterations"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=32,
            options=[
                {"label": "8", "value": 8},
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128}
            ],
            description="Size of minibatches for training"
        ),
        AlgorithmParameter(
            name="activation",
            label="Activation Function",
            type="select",
            default="relu",
            options=[
                {"label": "ReLU", "value": "relu"},
                {"label": "Tanh", "value": "tanh"},
                {"label": "Sigmoid", "value": "sigmoid"}
            ],
            description="Activation function for hidden layers"
        )
    ],
    dataset_name="iris",
    visualization_type="network_architecture,training_curves,confusion_matrix",
    theory=(
        "Feedforward Neural Networks, also known as Multilayer Perceptrons (MLPs), are "
        "the foundation of deep learning. These networks consist of layers of neurons where "
        "information flows in one direction: from input through hidden layers to output. "
        "Key components:\n\n"
        "1. Input Layer: Receives the input features\n"
        "2. Hidden Layers: Process information through weighted connections and activation functions\n"
        "3. Output Layer: Produces final predictions (classification probabilities or regression values)\n"
        "4. Activation Functions: Introduce non-linearity (ReLU, tanh, sigmoid)\n"
        "5. Backpropagation: Algorithm for computing gradients and updating weights\n\n"
        "Training uses gradient descent to minimize a loss function by adjusting weights. "
        "The universal approximation theorem states that MLPs with sufficient neurons can "
        "approximate any continuous function, making them powerful for complex pattern recognition tasks."
    ),
    pros=[
        "Universal function approximators with sufficient layers",
        "Can learn complex non-linear patterns",
        "Works well with structured/tabular data",
        "Flexible architecture design",
        "Well-established training algorithms"
    ],
    cons=[
        "Requires careful hyperparameter tuning",
        "Can overfit with insufficient regularization",
        "Black box nature - less interpretable",
        "Requires sufficient training data",
        "Sensitive to feature scaling"
    ],
    related_algorithms=["cnn", "rnn", "autoencoder", "rbf-network"]
)

AlgorithmRegistry.register(mlp_metadata)


@router.post("/feedforward-nn/train", response_model=MLPResponse)
async def train_mlp(request: MLPRequest) -> MLPResponse:
    """Train a Feedforward Neural Network (MLP) on the Iris dataset.

    This endpoint trains an MLP classifier with configurable architecture on the
    classic Iris flower classification dataset, returning metrics, predictions,
    training curves, and network architecture visualization.

    Args:
        request: MLP training parameters including hidden_layers, learning_rate,
                epochs, batch_size, and activation function

    Returns:
        MLPResponse with training results, metrics, training curves, confusion matrix,
        and network architecture visualization data

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training MLP with parameters: {request.model_dump()}")

        # Load data
        data = load_iris_data(
            test_size=0.2,
            random_state=request.random_state,
            normalize=True
        )

        # Initialize and train model
        model = MLPModel(
            hidden_layers=request.hidden_layers,
            learning_rate=request.learning_rate,
            activation=request.activation,
            batch_size=request.batch_size,
            random_state=request.random_state
        )

        training_info = model.train(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            max_iter=request.epochs
        )

        # Make predictions
        predictions = model.predict(data['X_test'])

        # Evaluate model
        metrics = model.evaluate(data['X_test'], data['y_test'])

        # Get model info
        model_info = model.get_model_info()
        network_architecture = model.get_network_architecture()

        # Prepare visualization data
        viz_data = prepare_mlp_visualization_data(
            X_sample=data['X_test'][:10],
            y_sample=data['y_test'][:10],
            y_pred=predictions[:10],
            feature_names=data['feature_names']
        )

        # Add training curves to visualization
        viz_data['training_curves'] = {
            'iterations': model.training_history['iterations'],
            'loss': model.training_history['loss'],
            'accuracy': model.training_history['accuracy']
        }

        # Add confusion matrix to visualization
        viz_data['confusion_matrix'] = metrics['confusion_matrix']

        # Add network architecture info
        viz_data['network_architecture'] = network_architecture

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"MLP training completed in {execution_time_ms:.2f}ms "
            f"with accuracy: {metrics['accuracy']:.4f}, "
            f"iterations: {training_info['iterations_trained']}"
        )

        return MLPResponse(
            success=True,
            metrics=metrics,
            predictions=predictions.tolist(),
            actual=data['y_test'].tolist(),
            training_history=model.training_history,
            visualization_data=viz_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'hidden_layers': request.hidden_layers,
                'learning_rate': request.learning_rate,
                'epochs': request.epochs,
                'batch_size': request.batch_size,
                'activation': request.activation,
                'dataset_name': request.dataset_name
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/feedforward-nn/info")
async def get_mlp_info() -> Dict[str, Any]:
    """Get MLP algorithm information and metadata.

    Returns metadata, parameters, and dataset information for MLP.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("feedforward-nn")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="MLP metadata not found"
        )

    dataset_info = get_mlp_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register GAN metadata
gan_metadata = AlgorithmMetadata(
    id="gan",
    name="GAN (Generative Adversarial Network)",
    slug="gan",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Train generator and discriminator networks to create synthetic images",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["deep-learning", "gan", "generative", "adversarial", "synthesis"],
    use_cases=[
        "Image synthesis",
        "Data augmentation",
        "Style transfer",
        "Super resolution",
        "Art generation",
        "Missing data imputation"
    ],
    complexity=AlgorithmComplexity(
        time="O(epochs*batch_size*2)",
        space="O(generator_params + discriminator_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="latent_dim",
            label="Latent Space Size",
            type="range",
            default=100,
            min=32,
            max=256,
            step=16,
            description="Latent space dimension for noise vector input"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.0002,
            min=0.00001,
            max=0.001,
            step=0.00001,
            description="Learning rate for both networks"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=64,
            options=[
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128},
                {"label": "256", "value": 256}
            ],
            description="Batch size for training"
        ),
        AlgorithmParameter(
            name="g_hidden",
            label="Generator Hidden Size",
            type="range",
            default=256,
            min=128,
            max=512,
            step=64,
            description="Size of generator's hidden layers"
        ),
        AlgorithmParameter(
            name="d_hidden",
            label="Discriminator Hidden Size",
            type="range",
            default=256,
            min=128,
            max=512,
            step=64,
            description="Size of discriminator's hidden layers"
        )
    ],
    dataset_name="mnist_digits",
    visualization_type="generated_samples,training_curves,latent_interpolation,decision_boundary",
    theory=(
        "Generative Adversarial Networks (GANs) consist of two neural networks that compete "
        "in a zero-sum game. The Generator (G) learns to create realistic fake samples from "
        "random noise, while the Discriminator (D) learns to distinguish real from fake samples.\n\n"
        "Training Process:\n"
        "1. Generator creates fake samples from random noise vectors\n"
        "2. Discriminator evaluates both real and fake samples\n"
        "3. Both networks update their weights based on their performance\n"
        "4. Generator improves at fooling the discriminator\n"
        "5. Discriminator improves at detecting fakes\n\n"
        "The training reaches equilibrium when the discriminator can no longer distinguish "
        "real from generated samples. GANs use binary cross-entropy loss and are trained "
        "with alternating gradient descent. The generator never sees real data directly, "
        "learning only through the discriminator's feedback."
    ),
    pros=[
        "Can generate highly realistic samples",
        "Learns complex data distributions",
        "No explicit density estimation needed",
        "Flexible architecture for various data types",
        "Can generate novel, diverse samples"
    ],
    cons=[
        "Training can be unstable and difficult",
        "Prone to mode collapse (limited diversity)",
        "Requires careful hyperparameter tuning",
        "Difficult to evaluate quality objectively",
        "May require large datasets for good results"
    ],
    related_algorithms=["vae", "diffusion", "autoencoder", "cnn"]
)

AlgorithmRegistry.register(gan_metadata)


# Register Autoencoder metadata
autoencoder_metadata = AlgorithmMetadata(
    id="autoencoder",
    name="Autoencoder",
    slug="autoencoder",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Neural network for unsupervised feature learning and dimensionality reduction",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "unsupervised", "dimensionality-reduction", "generative"],
    use_cases=[
        "Data compression",
        "Denoising",
        "Anomaly detection",
        "Feature extraction",
        "Image generation"
    ],
    complexity=AlgorithmComplexity(
        time="O(epochs*n*d²)",
        space="O(d²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="latent_dim",
            label="Latent Dimension",
            type="range",
            default=32,
            min=2,
            max=128,
            step=2,
            description="Dimension of the latent space for dimensionality reduction"
        ),
        AlgorithmParameter(
            name="hidden_dim",
            label="Hidden Layer Size",
            type="range",
            default=128,
            min=64,
            max=512,
            step=32,
            description="Size of hidden layers in encoder and decoder"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for Adam optimizer"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=64,
            options=[
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128},
                {"label": "256", "value": 256}
            ],
            description="Batch size for training"
        )
    ],
    dataset_name="mnist_digits",
    visualization_type="reconstructions,latent_space,loss_curves",
    theory=(
        "Autoencoders are neural networks designed for unsupervised feature learning. "
        "They consist of two main components:\n\n"
        "1. Encoder: Compresses input data into a low-dimensional latent representation\n"
        "2. Decoder: Reconstructs the original data from the latent representation\n\n"
        "The network is trained to minimize reconstruction error, forcing it to learn "
        "the most important features needed to recreate the input. The bottleneck "
        "(latent space) acts as a compressed representation of the data.\n\n"
        "Training Process:\n"
        "- Input data is passed through the encoder to get latent representation\n"
        "- Latent representation is passed through decoder to reconstruct input\n"
        "- Mean Squared Error (MSE) between input and reconstruction is minimized\n"
        "- Network learns to extract essential features for reconstruction\n\n"
        "The learned latent space often captures meaningful structure, with similar "
        "inputs mapping to nearby points in latent space. This makes autoencoders "
        "useful for dimensionality reduction, denoising, and anomaly detection."
    ),
    pros=[
        "Learns meaningful features without labels (unsupervised)",
        "Effective dimensionality reduction for visualization",
        "Can denoise data by learning robust representations",
        "Useful for anomaly detection (high reconstruction error)",
        "Simple architecture, easy to understand and implement"
    ],
    cons=[
        "May not capture complex data distributions as well as VAE",
        "Latent space may not be smooth or continuous",
        "Cannot generate new samples as easily as GANs/VAEs",
        "Requires careful tuning of latent dimension size",
        "May overfit on small datasets"
    ],
    related_algorithms=["vae", "pca", "gan", "denoising-autoencoder"]
)

AlgorithmRegistry.register(autoencoder_metadata)


# Register ResNet metadata
resnet_metadata = AlgorithmMetadata(
    id="resnet",
    name="ResNet (Residual Network)",
    slug="resnet",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Deep CNN with skip connections for image classification",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["deep-learning", "cnn", "image-classification", "residual-learning"],
    use_cases=[
        "Image classification",
        "Feature extraction",
        "Transfer learning",
        "Medical imaging",
        "Object recognition"
    ],
    complexity=AlgorithmComplexity(
        time="O(image_size*depth)",
        space="O(depth*filters²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="model_variant",
            label="Model Variant",
            type="select",
            default="resnet18",
            options=[
                {"label": "ResNet-18", "value": "resnet18"},
                {"label": "ResNet-34", "value": "resnet34"},
                {"label": "ResNet-50", "value": "resnet50"},
                {"label": "ResNet-101", "value": "resnet101"}
            ],
            description="ResNet architecture variant"
        ),
        AlgorithmParameter(
            name="top_k",
            label="Top K Predictions",
            type="range",
            default=5,
            min=1,
            max=10,
            step=1,
            description="Number of top predictions to return"
        ),
        AlgorithmParameter(
            name="use_pretrained",
            label="Use Pre-trained Weights",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Whether to use pre-trained ImageNet weights"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="range",
            default=0,
            min=0,
            max=9,
            step=1,
            description="Select sample image from gallery (0-9)"
        )
    ],
    dataset_name="imagenet",
    visualization_type="predictions,confidence_bars,residual_blocks,architecture",
    theory=(
        "Residual Networks (ResNet) revolutionized deep learning by introducing skip connections "
        "that allow training of very deep networks (100+ layers). The key innovation is the "
        "residual block, which learns a residual function F(x) = H(x) - x rather than directly "
        "learning the desired mapping H(x). This is achieved through skip connections that add "
        "the input x to the output of stacked layers.\n\n"
        "Key Components:\n"
        "1. Residual Block: Conv layers + skip connection that bypasses them\n"
        "2. Skip Connection: Identity mapping x that is added to block output\n"
        "3. Batch Normalization: Normalizes activations for stable training\n"
        "4. Global Average Pooling: Replaces fully connected layers\n\n"
        "The residual formulation makes optimization easier because:\n"
        "- If identity mapping is optimal, layers can easily learn to zero out their weights\n"
        "- Gradients flow directly through skip connections, avoiding vanishing gradients\n"
        "- Network can choose optimal depth by learning which layers to use\n\n"
        "ResNet variants differ in depth:\n"
        "- ResNet-18/34: Use basic residual blocks (2 conv layers per block)\n"
        "- ResNet-50/101/152: Use bottleneck blocks (3 conv layers: 1x1, 3x3, 1x1)\n\n"
        "Pre-trained on ImageNet (1000 classes), ResNet achieves state-of-the-art accuracy "
        "and serves as an excellent feature extractor for transfer learning."
    ),
    pros=[
        "Can train very deep networks (100+ layers) effectively",
        "Skip connections prevent vanishing gradient problem",
        "State-of-the-art accuracy on ImageNet",
        "Excellent for transfer learning and feature extraction",
        "Identity mappings improve gradient flow"
    ],
    cons=[
        "Larger models (ResNet-50+) require significant memory",
        "More computationally expensive than simpler CNNs",
        "Pre-trained weights are large (50-100MB+)",
        "Inference can be slow on CPU for deeper variants",
        "Requires careful hyperparameter tuning when training from scratch"
    ],
    related_algorithms=["cnn", "vgg", "inception", "densenet"]
)

AlgorithmRegistry.register(resnet_metadata)


@router.post("/gan/train", response_model=GANResponse)
async def train_gan(request: GANRequest) -> GANResponse:
    """Train a Generative Adversarial Network to generate digit images.

    This endpoint trains a GAN on the digits dataset, with a Generator learning
    to create digit-like images from noise and a Discriminator learning to
    distinguish real from generated images.

    Args:
        request: GAN training parameters including latent_dim, g_hidden, d_hidden,
                learning_rate, epochs, batch_size, and random_state

    Returns:
        GANResponse with training results, loss history, generated samples,
        and visualization data

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training GAN with parameters: {request.model_dump()}")

        # Load digits data
        data = load_gan_digits_data(random_state=request.random_state)
        X = data['X']

        # Prepare data for GAN
        X = prepare_data_for_gan(X)

        # Initialize GAN model
        model = GANModel(
            latent_dim=request.latent_dim,
            g_hidden=request.g_hidden,
            d_hidden=request.d_hidden,
            learning_rate=request.learning_rate,
            random_state=request.random_state
        )

        # Train GAN
        training_results = model.train(
            X=X,
            epochs=request.epochs,
            batch_size=request.batch_size,
            sample_interval=max(10, request.epochs // 10)  # Save ~10 samples during training
        )

        # Generate final samples
        final_samples = model.generate_samples(n_samples=16)

        # Generate latent space interpolations
        interpolated_samples = model.interpolate_latent_space(
            n_steps=10,
            n_interpolations=3
        )

        # Compute discriminator decision boundary
        decision_boundary = model.compute_decision_boundary(n_samples=200)

        # Get model info
        model_info = model.get_model_info()
        g_params, d_params = model.count_parameters()
        model_info['generator_params'] = g_params
        model_info['discriminator_params'] = d_params
        model_info['g_hidden'] = request.g_hidden
        model_info['d_hidden'] = request.d_hidden
        model_info['total_epochs'] = request.epochs

        # Prepare visualization data
        sample_epochs = [s['epoch'] for s in training_results['generated_samples']]
        loss_epochs = [h['epoch'] for h in training_results['loss_history']]

        visualization_data = {
            'sample_epochs': sample_epochs,
            'loss_epochs': loss_epochs,
            'n_samples': data['n_samples'],
            'image_shape': [8, 8]
        }

        execution_time_ms = (time.time() - start_time) * 1000

        # Get final metrics
        final_metrics = training_results['loss_history'][-1]

        logger.info(
            f"GAN training completed in {execution_time_ms:.2f}ms. "
            f"Final G loss: {final_metrics['g_loss']:.4f}, "
            f"Final D loss: {final_metrics['d_loss']:.4f}, "
            f"D real acc: {final_metrics.get('d_real_accuracy', 0):.4f}, "
            f"D fake acc: {final_metrics.get('d_fake_accuracy', 0):.4f}"
        )

        return GANResponse(
            loss_history=training_results['loss_history'],
            generated_samples=training_results['generated_samples'],
            final_samples=final_samples.tolist(),
            interpolated_samples=interpolated_samples.tolist(),
            decision_boundary=decision_boundary,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/gan/info")
async def get_gan_info() -> Dict[str, Any]:
    """Get GAN algorithm information and metadata.

    Returns metadata, parameters, and dataset information for GAN.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("gan")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="GAN metadata not found"
        )

    dataset_info = get_gan_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register VAE metadata
vae_metadata = AlgorithmMetadata(
    id="vae",
    name="Variational Autoencoder",
    slug="vae",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Probabilistic autoencoder that learns latent space distributions",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["deep-learning", "generative", "unsupervised", "vae", "probabilistic", "autoencoder"],
    use_cases=[
        "Generative modeling",
        "Data augmentation",
        "Anomaly detection",
        "Interpolation between data points",
        "Controllable generation"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*encoder_params*epochs)",
        space="O(latent_dim)"
    ),
    parameters=[
        AlgorithmParameter(
            name="latent_dim",
            label="Latent Space Dimension",
            type="range",
            default=20,
            min=2,
            max=64,
            step=1,
            description="Dimension of the latent space representation"
        ),
        AlgorithmParameter(
            name="encoder_hidden",
            label="Encoder Hidden Layers",
            type="text",
            default="[128, 64]",
            description="List of hidden layer sizes for encoder (e.g., [128, 64])"
        ),
        AlgorithmParameter(
            name="decoder_hidden",
            label="Decoder Hidden Layers",
            type="text",
            default="[64, 128]",
            description="List of hidden layer sizes for decoder (e.g., [64, 128])"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for the optimizer"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=30,
            min=10,
            max=100,
            step=5,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="beta",
            label="Beta (KL Weight)",
            type="range",
            default=1.0,
            min=0.1,
            max=10.0,
            step=0.1,
            description="Weight for KL divergence term (beta-VAE)"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=128,
            options=[
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128},
                {"label": "256", "value": 256}
            ],
            description="Batch size for training"
        )
    ],
    dataset_name="digits",
    visualization_type="vae_visualization",
    theory=(
        "Variational Autoencoder (VAE) is a probabilistic generative model that learns "
        "to encode data into a latent distribution and decode samples back to the original "
        "data space. Unlike standard autoencoders, VAEs learn a distribution over the latent "
        "space rather than point estimates. The model uses the reparameterization trick "
        "(z = μ + σ * ε, where ε ~ N(0,1)) to enable backpropagation through random sampling. "
        "The loss function consists of two terms: (1) Reconstruction loss - measures how well "
        "the decoder reconstructs the input, and (2) KL divergence - regularizes the latent "
        "distribution to be close to a standard normal distribution. This enables smooth "
        "interpolation in latent space and generation of new samples."
    ),
    pros=[
        "Learns smooth, continuous latent space representations",
        "Can generate new samples by sampling from latent distribution",
        "Provides principled probabilistic framework",
        "Enables interpolation between data points",
        "Useful for semi-supervised learning and data augmentation"
    ],
    cons=[
        "Generated samples can be blurry compared to GANs",
        "Training can be sensitive to hyperparameters (especially beta)",
        "Posterior collapse: latent space may be underutilized",
        "Requires careful balancing of reconstruction and KL terms",
        "Computationally intensive for high-dimensional data"
    ],
    related_algorithms=["autoencoder", "gan", "beta-vae", "cvae"]
)

AlgorithmRegistry.register(vae_metadata)


@router.post("/vae/train", response_model=VAEResponse)
async def train_vae(request: VAERequest):
    """Train a Variational Autoencoder (VAE) on the digits dataset.

    This endpoint trains a VAE with the specified architecture and hyperparameters,
    returning reconstructions, generated samples, latent space visualization,
    and training metrics.

    Args:
        request: VAE training parameters

    Returns:
        VAEResponse with training results, visualizations, and metrics

    Raises:
        HTTPException: If training fails
    """
    try:
        start_time = time.time()

        logger.info(f"Training VAE with parameters: {request.model_dump()}")

        # Load data
        data = load_vae_data(random_state=request.random_state, normalize=True)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        input_dim = data['input_dim']

        # Initialize model
        model = VAEModel(
            input_dim=input_dim,
            latent_dim=request.latent_dim,
            encoder_hidden=request.encoder_hidden,
            decoder_hidden=request.decoder_hidden,
            learning_rate=request.learning_rate,
            beta=request.beta,
            random_state=request.random_state
        )

        # Train model
        train_results = model.train(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            epochs=request.epochs,
            batch_size=request.batch_size
        )

        # Get reconstructions (sample from test set)
        n_samples = min(20, len(X_test))
        sample_indices = list(range(n_samples))
        X_sample = X_test[sample_indices]
        y_sample = y_test[sample_indices]
        reconstructions = model.reconstruct(X_sample)

        # Generate new samples
        n_generated = 16
        generated_samples = model.generate(n_generated)

        # Get latent space representation (2D projection)
        latent_2d = model.get_latent_2d_projection(X_test, method='pca')

        # Prepare visualization data
        visualization_data = {
            "original_images": [
                {
                    "image": X_sample[i].tolist(),
                    "label": int(y_sample[i]),
                    "index": i
                }
                for i in range(n_samples)
            ],
            "reconstructed_images": [
                {
                    "image": reconstructions[i].tolist(),
                    "label": int(y_sample[i]),
                    "index": i
                }
                for i in range(n_samples)
            ],
            "generated_images": [
                {
                    "image": generated_samples[i].tolist(),
                    "index": i
                }
                for i in range(n_generated)
            ],
            "latent_space": {
                "points": [
                    {
                        "x": float(latent_2d[i, 0]),
                        "y": float(latent_2d[i, 1]),
                        "label": int(y_test[i])
                    }
                    for i in range(len(latent_2d))
                ],
                "n_components": 2
            },
            "loss_curves": {
                "epochs": list(range(1, request.epochs + 1)),
                "total_loss": model.training_history['total_loss'],
                "reconstruction_loss": model.training_history['recon_loss'],
                "kl_divergence": model.training_history['kl_div']
            },
            "image_shape": data['image_shape']
        }

        # Prepare metrics
        metrics = {
            "final_train_loss": train_results['final_train_loss'],
            "final_reconstruction_loss": train_results['final_train_recon_loss'],
            "final_kl_divergence": train_results['final_train_kl_div'],
            "test_loss": train_results['test_loss'],
            "test_reconstruction_loss": train_results['test_recon_loss'],
            "test_kl_divergence": train_results['test_kl_div']
        }

        # Get model info
        model_info = model.get_model_info()

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"VAE training completed in {execution_time_ms:.2f}ms. "
            f"Final loss: {metrics['final_train_loss']:.4f}"
        )

        return VAEResponse(
            success=True,
            metrics=metrics,
            original_images=[img["image"] for img in visualization_data["original_images"]],
            reconstructed_images=[img["image"] for img in visualization_data["reconstructed_images"]],
            generated_images=[img["image"] for img in visualization_data["generated_images"]],
            latent_space=[
                [p["x"], p["y"]] for p in visualization_data["latent_space"]["points"]
            ],
            latent_labels=[p["label"] for p in visualization_data["latent_space"]["points"]],
            loss_history={
                "epochs": visualization_data["loss_curves"]["epochs"],
                "total_loss": visualization_data["loss_curves"]["total_loss"],
                "reconstruction_loss": visualization_data["loss_curves"]["reconstruction_loss"],
                "kl_divergence": visualization_data["loss_curves"]["kl_divergence"]
            },
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                "latent_dim": request.latent_dim,
                "encoder_hidden": request.encoder_hidden,
                "decoder_hidden": request.decoder_hidden,
                "learning_rate": request.learning_rate,
                "epochs": request.epochs,
                "beta": request.beta,
                "batch_size": request.batch_size
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/vae/info")
async def get_vae_info() -> Dict[str, Any]:
    """Get VAE algorithm information and metadata.

    Returns metadata, parameters, and dataset information for VAE.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("vae")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="VAE metadata not found"
        )

    dataset_info = get_vae_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/lstm/train", response_model=LSTMResponse)
async def train_lstm(request: LSTMRequest) -> LSTMResponse:
    """Train an LSTM network on synthetic time series data.

    This endpoint trains an LSTM model with the specified parameters,
    demonstrating the network's ability to learn long-term dependencies
    in sequential data. Returns comprehensive metrics, predictions,
    and gate activation visualizations.

    Args:
        request: LSTM training parameters including hidden_size, num_layers,
                dropout, learning_rate, epochs, sequence_length, and batch_size

    Returns:
        LSTMResponse containing:
            - Training and test metrics (MSE, MAE, R²)
            - Predictions vs actual values
            - Training curves (loss over epochs)
            - Gate activations (forget, input, output gates)
            - Cell state and hidden state evolution
            - Model information and execution time

    Raises:
        HTTPException: If training fails or invalid parameters provided
    """
    try:
        logger.info(f"Training LSTM with parameters: {request.model_dump()}")

        # Create and train model
        model = LSTMModel(request)
        response = model.train()

        if not response.success:
            logger.error(f"LSTM training failed: {response.error}")
            raise HTTPException(status_code=400, detail=response.error)

        logger.info(
            f"LSTM training completed in {response.execution_time_ms:.2f}ms "
            f"with test MSE: {response.metrics.test_mse:.4f}, "
            f"test R²: {response.metrics.test_r2:.4f}"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/lstm/info")
async def get_lstm_info() -> Dict[str, Any]:
    """Get LSTM algorithm metadata and information.

    Returns detailed information about the LSTM algorithm including
    parameters, complexity, theory, use cases, and dataset information.

    Returns:
        Dictionary containing:
            - metadata: Algorithm metadata (parameters, theory, pros/cons)
            - dataset: Information about the synthetic time series dataset

    Raises:
        HTTPException: If algorithm metadata not found
    """
    metadata = AlgorithmRegistry.get("lstm")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="LSTM algorithm metadata not found"
        )

    dataset_info = get_lstm_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/autoencoder/train", response_model=AutoencoderResponse)
async def train_autoencoder(request: AutoencoderRequest) -> AutoencoderResponse:
    """Train an Autoencoder for unsupervised feature learning and dimensionality reduction.

    This endpoint trains an autoencoder on MNIST digits dataset. The autoencoder
    learns to compress images into a low-dimensional latent space and reconstruct them,
    discovering important features without supervision.

    Args:
        request: Autoencoder training parameters including latent_dim, hidden_dim,
                epochs, learning_rate, batch_size, and random_state

    Returns:
        AutoencoderResponse with training results, reconstructions, latent space
        visualization, and loss curves

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training Autoencoder with parameters: {request.model_dump()}")

        # Load MNIST digits data (subset for speed)
        data = load_mnist_data(
            n_samples=1000,
            test_size=0.2,
            random_state=request.random_state
        )

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        # Initialize autoencoder model
        model = AutoencoderModel(
            latent_dim=request.latent_dim,
            hidden_dim=request.hidden_dim,
            learning_rate=request.learning_rate,
            random_state=request.random_state
        )

        # Train autoencoder
        training_results = model.train(
            X_train=X_train,
            X_test=X_test,
            epochs=request.epochs,
            batch_size=request.batch_size
        )

        # Evaluate reconstruction quality
        metrics = model.evaluate(X_test)
        reconstruction_loss = metrics['reconstruction_loss']
        avg_pixel_error = metrics['avg_pixel_error']

        # Get reconstructions for visualization
        reconstructed = model.reconstruct(X_test)
        original_samples, reconstructed_samples = prepare_sample_comparison(
            X_test, reconstructed, n_samples=10
        )

        # Get latent representations for visualization
        latent_vectors = model.encode(X_test)
        coords_2d, projection_method = compute_latent_visualization(
            latent_vectors,
            y_test,
            request.latent_dim,
            random_state=request.random_state
        )

        # Prepare latent space visualization data
        latent_representations = [
            {
                'x': float(coords_2d[i, 0]),
                'y': float(coords_2d[i, 1]),
                'label': int(y_test[i])
            }
            for i in range(len(coords_2d))
        ]

        # Prepare loss history
        loss_history = [
            {
                'epoch': i,
                'train_loss': float(model.training_history['train_loss'][i]),
                'val_loss': float(model.training_history['val_loss'][i])
            }
            for i in range(len(model.training_history['train_loss']))
        ]

        # Prepare visualization data
        visualization_data = {
            'n_samples': len(X_test),
            'image_shape': [8, 8],
            'latent_dim': request.latent_dim,
            'projection_method': projection_method
        }

        # Get model info
        model_info = model.get_model_info()
        encoder_params, decoder_params = model.count_parameters()
        model_info['encoder_params'] = encoder_params
        model_info['decoder_params'] = decoder_params
        model_info['total_epochs'] = request.epochs

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Autoencoder training completed in {execution_time_ms:.2f}ms. "
            f"Final reconstruction loss: {reconstruction_loss:.4f}, "
            f"Average pixel error: {avg_pixel_error:.4f}"
        )

        return AutoencoderResponse(
            reconstruction_loss=reconstruction_loss,
            avg_pixel_error=avg_pixel_error,
            loss_history=loss_history,
            original_images=original_samples.tolist(),
            reconstructed_images=reconstructed_samples.tolist(),
            latent_representations=latent_representations,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/autoencoder/info")
async def get_autoencoder_info() -> Dict[str, Any]:
    """Get Autoencoder algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Autoencoder.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("autoencoder")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Autoencoder metadata not found"
        )

    dataset_info = get_autoencoder_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/gru/train", response_model=GRUResponse)
async def train_gru(request: GRURequest) -> GRUResponse:
    """Train a GRU network on synthetic time series data.

    This endpoint trains a GRU model with the specified parameters,
    demonstrating the network's gating mechanisms for sequence learning.
    Returns comprehensive metrics, predictions, and gate activation visualizations.

    Args:
        request: GRU training parameters including hidden_size, num_layers,
                learning_rate, epochs, sequence_length, dropout, and batch_size

    Returns:
        GRUResponse containing:
            - Training and test metrics (MSE, MAE, R²)
            - Predictions vs actual values
            - Training curves (loss over epochs)
            - Gate activations (update gate, reset gate)
            - Hidden state evolution
            - Model information and execution time

    Raises:
        HTTPException: If training fails or invalid parameters provided
    """
    try:
        logger.info(f"Training GRU with parameters: {request.model_dump()}")

        # Create and train model
        model = GRUModel(request)
        response = model.train()

        if not response.success:
            logger.error(f"GRU training failed: {response.error}")
            raise HTTPException(status_code=400, detail=response.error)

        logger.info(
            f"GRU training completed in {response.execution_time_ms:.2f}ms "
            f"with test MSE: {response.metrics.test_mse:.4f}, "
            f"test R²: {response.metrics.test_r2:.4f}"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/gru/info")
async def get_gru_info() -> Dict[str, Any]:
    """Get GRU algorithm metadata and information.

    Returns detailed information about the GRU algorithm including
    parameters, complexity, theory, use cases, and dataset information.

    Returns:
        Dictionary containing:
            - metadata: Algorithm metadata (parameters, theory, pros/cons)
            - dataset: Information about the synthetic time series dataset

    Raises:
        HTTPException: If algorithm metadata not found
    """
    metadata = AlgorithmRegistry.get("gru")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="GRU algorithm metadata not found"
        )

    dataset_info = get_gru_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register VGG metadata
vgg_metadata = AlgorithmMetadata(
    id="vgg",
    name="VGG Network",
    slug="vgg",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Deep CNN with small 3x3 filters for image classification",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["deep-learning", "cnn", "image-classification", "transfer-learning"],
    use_cases=[
        "Image classification",
        "Feature extraction",
        "Transfer learning base",
        "Object recognition",
        "Style transfer (VGG features)"
    ],
    complexity=AlgorithmComplexity(
        time="O(image_size*depth)",
        space="O(138M params for VGG16)"
    ),
    parameters=[
        AlgorithmParameter(
            name="model_variant",
            label="Model Variant",
            type="select",
            default="vgg16",
            options=[
                {"label": "VGG-16", "value": "vgg16"},
                {"label": "VGG-19", "value": "vgg19"}
            ],
            description="VGG architecture variant"
        ),
        AlgorithmParameter(
            name="top_k",
            label="Top K Predictions",
            type="range",
            default=5,
            min=1,
            max=10,
            step=1,
            description="Number of top predictions to return"
        ),
        AlgorithmParameter(
            name="use_pretrained",
            label="Use Pre-trained Weights",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Whether to use pre-trained ImageNet weights"
        ),
        AlgorithmParameter(
            name="batch_norm",
            label="Batch Normalization",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Use batch normalization version"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="range",
            default=0,
            min=0,
            max=9,
            step=1,
            description="Select sample image from gallery (0-9)"
        )
    ],
    dataset_name="imagenet",
    visualization_type="predictions,confidence_bars,feature_maps,architecture",
    theory=(
        "VGG Networks (Visual Geometry Group) are deep convolutional neural networks "
        "characterized by their use of very small (3x3) convolution filters throughout "
        "the entire architecture. This design choice was revolutionary, demonstrating that "
        "network depth (number of layers) is a critical component for good performance.\n\n"
        "Key Architecture Features:\n"
        "1. Small Filters: All convolutional layers use 3x3 filters (smallest size to capture "
        "spatial patterns like left/right, up/down, center)\n"
        "2. Deep Networks: VGG-16 has 16 layers, VGG-19 has 19 layers\n"
        "3. Uniform Structure: 5 convolutional blocks, each containing multiple conv layers "
        "followed by max pooling (2x2, stride 2)\n"
        "4. Channel Progression: Filters double after each pooling: 64 -> 128 -> 256 -> 512 -> 512\n"
        "5. Fully Connected Layers: 3 FC layers (4096 -> 4096 -> 1000) at the end\n\n"
        "VGG-16 Structure:\n"
        "- Block 1: 2 conv layers (64 filters) + max pool\n"
        "- Block 2: 2 conv layers (128 filters) + max pool\n"
        "- Block 3: 3 conv layers (256 filters) + max pool\n"
        "- Block 4: 3 conv layers (512 filters) + max pool\n"
        "- Block 5: 3 conv layers (512 filters) + max pool\n"
        "- FC layers: 4096 -> 4096 -> 1000\n\n"
        "VGG-19 is similar but has 4 conv layers in blocks 3, 4, and 5 instead of 3.\n\n"
        "The key insight is that stacking multiple 3x3 conv layers has the same effective "
        "receptive field as a single larger filter, but with fewer parameters and more non-linearity. "
        "Two 3x3 convs = 5x5 receptive field, three 3x3 convs = 7x7 receptive field.\n\n"
        "VGG networks are particularly popular for transfer learning due to their simple, "
        "uniform architecture and strong feature extraction capabilities. The deep features "
        "learned by VGG are also widely used in style transfer applications."
    ),
    pros=[
        "Simple, uniform architecture that's easy to understand and implement",
        "Strong performance on ImageNet (92.7% top-5 accuracy)",
        "Excellent feature extractor for transfer learning",
        "Small 3x3 filters reduce parameters while increasing depth",
        "Deep features widely used in computer vision applications",
        "Well-suited for style transfer and feature visualization"
    ],
    cons=[
        "Very large model size (VGG-16: 138M parameters, mostly in FC layers)",
        "Slow to train and inference compared to modern architectures",
        "High memory consumption during training",
        "Outperformed by more recent architectures (ResNet, EfficientNet)",
        "No skip connections means vanishing gradient problems for very deep variants",
        "Fully connected layers contain most parameters but add limited capacity"
    ],
    related_algorithms=["resnet", "cnn", "alexnet", "inception"]
)

AlgorithmRegistry.register(vgg_metadata)


@router.post("/resnet/predict", response_model=ResNetResponse)
async def predict_resnet(request: ResNetRequest) -> ResNetResponse:
    """Run ResNet inference for image classification.

    This endpoint uses a pre-trained ResNet model to classify images from the
    ImageNet dataset. Returns top-K predictions with confidence scores,
    feature maps, and residual block architecture information.

    Args:
        request: ResNet inference parameters including model_variant, top_k,
                use_pretrained, and image selection

    Returns:
        ResNetResponse with predictions, confidence scores, feature maps,
        residual block information, and architecture visualization

    Raises:
        HTTPException: If inference fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Running ResNet inference with parameters: {request.model_dump()}")

        # Initialize ResNet model
        model = ResNetModel(
            model_variant=request.model_variant,
            use_pretrained=request.use_pretrained
        )

        # Run inference
        response = model.run_inference(request)

        logger.info(
            f"ResNet inference completed in {response.execution_time_ms:.2f}ms. "
            f"Top prediction: {response.predictions[0].class_name} "
            f"(confidence: {response.predictions[0].confidence:.4f})"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/resnet/info")
async def get_resnet_info() -> Dict[str, Any]:
    """Get ResNet algorithm information and metadata.

    Returns metadata, parameters, and dataset information for ResNet.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("resnet")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="ResNet metadata not found"
        )

    dataset_info = get_resnet_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Transformer metadata
transformer_metadata = AlgorithmMetadata(
    id="transformer",
    name="Transformer",
    slug="transformer",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Attention-based sequence-to-sequence model without recurrence",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["deep-learning", "attention", "sequence-to-sequence", "nlp", "transformer"],
    use_cases=[
        "Machine translation",
        "Text summarization",
        "Question answering",
        "Language modeling",
        "Image captioning"
    ],
    complexity=AlgorithmComplexity(
        time="O(T²*d)",
        space="O(T²)"
    ),
    parameters=[
        AlgorithmParameter(
            name="d_model",
            label="Model Dimension",
            type="range",
            default=128,
            min=64,
            max=512,
            step=64,
            description="Dimension of model embeddings and hidden states"
        ),
        AlgorithmParameter(
            name="nhead",
            label="Number of Attention Heads",
            type="select",
            default=8,
            options=[
                {"label": "2", "value": 2},
                {"label": "4", "value": 4},
                {"label": "8", "value": 8},
                {"label": "16", "value": 16}
            ],
            description="Number of parallel attention heads"
        ),
        AlgorithmParameter(
            name="num_layers",
            label="Number of Layers",
            type="range",
            default=2,
            min=1,
            max=6,
            step=1,
            description="Number of encoder and decoder layers"
        ),
        AlgorithmParameter(
            name="dim_feedforward",
            label="Feedforward Dimension",
            type="range",
            default=512,
            min=256,
            max=2048,
            step=256,
            description="Dimension of feedforward network in each layer"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for optimizer"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="dropout",
            label="Dropout Rate",
            type="range",
            default=0.1,
            min=0.0,
            max=0.5,
            step=0.05,
            description="Dropout rate for regularization"
        )
    ],
    dataset_name="sequence_reversal",
    visualization_type="attention_heatmap,training_curves,sequence_prediction,architecture",
    theory=(
        "The Transformer is a sequence-to-sequence model that relies entirely on attention mechanisms, "
        "eliminating the need for recurrence. It was introduced in the 'Attention is All You Need' paper "
        "and has become the foundation for modern NLP models like BERT and GPT.\n\n"
        "Key components:\n"
        "1. Multi-Head Self-Attention: Allows the model to attend to different parts of the input sequence\n"
        "2. Positional Encoding: Adds position information since there's no recurrence\n"
        "3. Encoder-Decoder Architecture: Encoder processes input, decoder generates output\n"
        "4. Feed-Forward Networks: Applied to each position separately and identically\n"
        "5. Layer Normalization and Residual Connections: Stabilize training\n\n"
        "Attention Mechanism: For each position, the model computes attention weights over all positions:\n"
        "Attention(Q, K, V) = softmax(QK^T / √d_k)V\n\n"
        "where Q (query), K (key), V (value) are learned linear projections of the input. "
        "Multi-head attention runs multiple attention operations in parallel, allowing the model "
        "to attend to information from different representation subspaces."
    ),
    pros=[
        "Highly parallelizable - faster training than RNNs",
        "Can capture long-range dependencies effectively",
        "State-of-the-art performance on many NLP tasks",
        "Self-attention provides interpretable attention weights",
        "Scalable to very large models and datasets"
    ],
    cons=[
        "Quadratic memory and computation complexity in sequence length",
        "Requires large amounts of data to train effectively",
        "More parameters than RNNs for similar capacity",
        "Positional encoding can limit extrapolation to longer sequences",
        "High computational cost for very long sequences"
    ],
    related_algorithms=["bert", "gpt", "attention", "seq2seq", "lstm"]
)

AlgorithmRegistry.register(transformer_metadata)


# Register Adam Optimizer metadata
adam_optimizer_metadata = AlgorithmMetadata(
    id="adam-optimizer",
    name="Adam Optimizer",
    slug="adam-optimizer",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Adaptive learning rate optimizer combining momentum and RMSprop",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "optimization", "training", "adaptive-learning-rate"],
    use_cases=[
        "Deep network training",
        "Adaptive learning rates",
        "Sparse gradient handling",
        "Default optimizer choice",
        "Non-convex optimization"
    ],
    complexity=AlgorithmComplexity(
        time="O(parameters)",
        space="O(parameters) for momentum buffers"
    ),
    parameters=[
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate (α)",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.1,
            step=0.0001,
            description="Base learning rate (step size)"
        ),
        AlgorithmParameter(
            name="beta1",
            label="Beta1 (β₁)",
            type="range",
            default=0.9,
            min=0.5,
            max=0.99,
            step=0.01,
            description="Exponential decay rate for first moment estimates (momentum)"
        ),
        AlgorithmParameter(
            name="beta2",
            label="Beta2 (β₂)",
            type="range",
            default=0.999,
            min=0.9,
            max=0.9999,
            step=0.0001,
            description="Exponential decay rate for second moment estimates (RMSprop)"
        ),
        AlgorithmParameter(
            name="epsilon",
            label="Epsilon (ε)",
            type="range",
            default=1e-8,
            min=1e-10,
            max=1e-6,
            step=1e-9,
            description="Small constant for numerical stability"
        ),
        AlgorithmParameter(
            name="compare_optimizers",
            label="Compare with Other Optimizers",
            type="select",
            default=True,
            options=[
                {"label": "Yes - Show All", "value": True},
                {"label": "No - Adam Only", "value": False}
            ],
            description="Compare Adam vs SGD vs Momentum vs RMSprop"
        ),
        AlgorithmParameter(
            name="max_iterations",
            label="Max Iterations",
            type="range",
            default=100,
            min=50,
            max=500,
            step=50,
            description="Maximum optimization steps"
        ),
        AlgorithmParameter(
            name="function_type",
            label="Optimization Function",
            type="select",
            default="rosenbrock",
            options=[
                {"label": "Rosenbrock (Banana Valley)", "value": "rosenbrock"},
                {"label": "Beale Function", "value": "beale"},
                {"label": "Himmelblau's Function", "value": "himmelblau"}
            ],
            description="Test function for optimization"
        )
    ],
    dataset_name="synthetic_optimization_landscape",
    visualization_type="trajectories,loss_surface,convergence_comparison,momentum_visualization",
    theory=(
        "Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm "
        "designed for training deep neural networks. It combines ideas from two popular optimizers:\n\n"
        "1. Momentum: Maintains exponentially decaying average of past gradients\n"
        "2. RMSprop: Adapts learning rates based on squared gradient magnitudes\n\n"
        "Adam Update Rules:\n"
        "For parameter θ, gradient g, and time step t:\n\n"
        "m_t = β₁ · m_{t-1} + (1 - β₁) · g_t  (first moment, momentum)\n"
        "v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²  (second moment, squared gradient)\n\n"
        "Bias Correction (important for early iterations):\n"
        "m̂_t = m_t / (1 - β₁^t)\n"
        "v̂_t = v_t / (1 - β₂^t)\n\n"
        "Parameter Update:\n"
        "θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)\n\n"
        "Key Features:\n"
        "- Adaptive per-parameter learning rates (dividing by √v̂_t)\n"
        "- Momentum-like behavior through first moment (m_t)\n"
        "- Bias correction prevents initial estimates from being biased toward zero\n"
        "- Works well with sparse gradients and non-stationary objectives\n"
        "- Combines benefits of AdaGrad and RMSprop\n\n"
        "Default hyperparameters (α=0.001, β₁=0.9, β₂=0.999, ε=10⁻⁸) work well for most problems, "
        "making Adam a popular default choice for deep learning."
    ),
    pros=[
        "Adaptive per-parameter learning rates - handles different scales automatically",
        "Fast convergence on non-convex optimization problems",
        "Works well with sparse gradients (NLP, recommender systems)",
        "Requires little hyperparameter tuning - good default values",
        "Computationally efficient and memory-friendly",
        "Handles noisy gradients and non-stationary objectives well"
    ],
    cons=[
        "Can fail to converge to optimal solution in some cases",
        "May generalize worse than SGD with momentum on some tasks",
        "Bias correction adds slight computational overhead",
        "Not guaranteed to converge in convex settings without modifications",
        "Can exhibit unstable behavior with very small β₂ values"
    ],
    related_algorithms=["sgd", "momentum", "rmsprop", "adagrad", "gradient-descent"]
)

AlgorithmRegistry.register(adam_optimizer_metadata)


# Register Activation Functions metadata
activation_functions_metadata = AlgorithmMetadata(
    id="activation-functions",
    name="Activation Functions",
    slug="activation-functions",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Compare different activation functions and their properties",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["deep-learning", "neural-networks", "activation-functions", "fundamentals"],
    use_cases=[
        "Neural network design",
        "Gradient flow optimization",
        "Avoiding vanishing gradients",
        "Dead neuron prevention",
        "Deep network training"
    ],
    complexity=AlgorithmComplexity(
        time="O(n)",
        space="O(n)"
    ),
    parameters=[
        AlgorithmParameter(
            name="function_type",
            label="Activation Function",
            type="select",
            default="leaky_relu",
            options=[
                {"label": "ReLU", "value": "relu"},
                {"label": "Leaky ReLU", "value": "leaky_relu"},
                {"label": "Sigmoid", "value": "sigmoid"},
                {"label": "Tanh", "value": "tanh"},
                {"label": "ELU", "value": "elu"},
                {"label": "Swish", "value": "swish"}
            ],
            description="Primary activation function to demonstrate"
        ),
        AlgorithmParameter(
            name="alpha",
            label="Leaky ReLU Alpha",
            type="range",
            default=0.01,
            min=0.0,
            max=0.3,
            step=0.01,
            description="Negative slope for Leaky ReLU (0 = ReLU, >0 = Leaky ReLU)"
        ),
        AlgorithmParameter(
            name="input_range",
            label="Input Range",
            type="text",
            default=[-10.0, 10.0],
            description="Range for x-axis visualization [min, max]"
        ),
        AlgorithmParameter(
            name="compare_all",
            label="Compare All Functions",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Show all activation functions together"
        ),
        AlgorithmParameter(
            name="num_points",
            label="Number of Points",
            type="range",
            default=200,
            min=50,
            max=1000,
            step=50,
            description="Number of points for visualization smoothness"
        )
    ],
    dataset_name="synthetic",
    visualization_type="function_plots,derivative_plots,comparison_table,dead_neuron_demo",
    theory=(
        "Activation functions introduce non-linearity into neural networks, enabling them to "
        "learn complex patterns. Without activation functions, a neural network would be equivalent "
        "to a linear regression model regardless of depth. Key activation functions include:\n\n"
        "1. ReLU (Rectified Linear Unit): f(x) = max(0, x)\n"
        "   - Most widely used in deep learning\n"
        "   - Fast computation and helps with vanishing gradients\n"
        "   - Suffers from 'dead neurons' (zero gradient for negative inputs)\n\n"
        "2. Leaky ReLU: f(x) = x if x > 0 else αx (typically α = 0.01)\n"
        "   - Fixes the dead neuron problem by allowing small negative values\n"
        "   - Maintains computational efficiency of ReLU\n\n"
        "3. Sigmoid: f(x) = 1 / (1 + e^(-x))\n"
        "   - Squashes output to (0, 1) range\n"
        "   - Useful for binary classification output layers\n"
        "   - Suffers from vanishing gradients for large |x|\n\n"
        "4. Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))\n"
        "   - Squashes output to (-1, 1) range\n"
        "   - Zero-centered (mean output closer to 0)\n"
        "   - Better than sigmoid but still has vanishing gradient issue\n\n"
        "5. ELU (Exponential Linear Unit): f(x) = x if x > 0 else α(e^x - 1)\n"
        "   - Smooth function with negative values\n"
        "   - Can produce negative outputs for faster learning\n\n"
        "6. Swish: f(x) = x * sigmoid(x)\n"
        "   - Self-gated activation function\n"
        "   - Smooth and non-monotonic\n"
        "   - Can outperform ReLU in deeper networks\n\n"
        "The choice of activation function significantly impacts training dynamics, "
        "convergence speed, and final model performance. Modern architectures often use "
        "ReLU variants for hidden layers and sigmoid/softmax for output layers."
    ),
    pros=[
        "Essential for non-linear learning in neural networks",
        "Different functions suit different network architectures",
        "Simple mathematical operations (efficient computation)",
        "Well-studied properties and derivatives for backpropagation",
        "ReLU variants help avoid vanishing gradient problem"
    ],
    cons=[
        "ReLU can cause dead neurons (zero gradient)",
        "Sigmoid/Tanh suffer from vanishing gradients",
        "No single best activation function for all tasks",
        "Choice requires experimentation and domain knowledge",
        "Some functions (ELU, Swish) more computationally expensive"
    ],
    related_algorithms=["feedforward-nn", "cnn", "backpropagation", "gradient-descent"]
)

AlgorithmRegistry.register(activation_functions_metadata)


# Register Gradient Descent Variants metadata
gradient_descent_metadata = AlgorithmMetadata(
    id="gradient-descent",
    name="Gradient Descent Variants",
    slug="gradient-descent",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Compare different gradient descent optimization algorithms",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "optimization", "gradient-descent", "training"],
    use_cases=[
        "Neural network training",
        "Parameter optimization",
        "Convergence analysis",
        "Optimizer selection",
        "Hyperparameter tuning"
    ],
    complexity=AlgorithmComplexity(
        time="O(iterations*parameters)",
        space="O(parameters)"
    ),
    parameters=[
        AlgorithmParameter(
            name="optimizer_type",
            label="Optimizer Algorithm",
            type="select",
            default="adam",
            options=[
                {"label": "SGD", "value": "sgd"},
                {"label": "Momentum", "value": "momentum"},
                {"label": "RMSprop", "value": "rmsprop"},
                {"label": "Adam", "value": "adam"},
                {"label": "Adagrad", "value": "adagrad"}
            ],
            description="Optimization algorithm to use"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.01,
            min=0.001,
            max=0.5,
            step=0.001,
            description="Learning rate for parameter updates"
        ),
        AlgorithmParameter(
            name="momentum",
            label="Momentum Coefficient",
            type="range",
            default=0.9,
            min=0.0,
            max=0.99,
            step=0.01,
            description="Momentum coefficient (for momentum-based optimizers)"
        ),
        AlgorithmParameter(
            name="iterations",
            label="Optimization Steps",
            type="range",
            default=100,
            min=20,
            max=500,
            step=10,
            description="Number of optimization iterations"
        ),
        AlgorithmParameter(
            name="compare_all",
            label="Compare All Optimizers",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Run all optimizers for comparison"
        ),
        AlgorithmParameter(
            name="test_function",
            label="Test Function",
            type="select",
            default="rosenbrock",
            options=[
                {"label": "Rosenbrock", "value": "rosenbrock"},
                {"label": "Beale", "value": "beale"},
                {"label": "Ackley", "value": "ackley"},
                {"label": "Sphere", "value": "sphere"}
            ],
            description="2D test function for optimization"
        )
    ],
    dataset_name="optimization_landscapes",
    visualization_type="contour_trajectories,convergence_curves,statistics_table",
    theory=(
        "Gradient Descent is the fundamental optimization algorithm for training neural networks. "
        "It iteratively updates parameters by moving in the direction of steepest descent: θ = θ - η∇f(θ). "
        "Modern variants improve upon basic SGD:\n\n"
        "1. SGD (Stochastic Gradient Descent): Basic gradient descent with constant learning rate. "
        "Simple but can be slow and get stuck in saddle points.\n\n"
        "2. Momentum: Adds velocity term that accumulates gradients: v = βv + η∇f(θ), θ = θ - v. "
        "Helps accelerate in relevant directions and dampen oscillations. Typical β = 0.9.\n\n"
        "3. RMSprop: Adapts learning rate per parameter using moving average of squared gradients: "
        "E[g²] = ρE[g²] + (1-ρ)g², θ = θ - η·g/√(E[g²]+ε). Maintains per-parameter learning rates.\n\n"
        "4. Adam (Adaptive Moment Estimation): Combines momentum and RMSprop. Maintains first (m) and "
        "second (v) moment estimates with bias correction. Most popular optimizer for deep learning. "
        "Formula: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², θ = θ - η·m̂/(√v̂+ε). "
        "Typical values: β₁=0.9, β₂=0.999, ε=1e-8.\n\n"
        "5. Adagrad: Accumulates squared gradients: G = G + g², θ = θ - η·g/√(G+ε). "
        "Adapts learning rate but can decay too aggressively for deep networks.\n\n"
        "Key concepts:\n"
        "- Learning Rate (η): Step size for updates. Too large causes divergence, too small is slow.\n"
        "- Momentum (β): Smooths updates and accelerates convergence\n"
        "- Adaptive Rates: Per-parameter learning rates improve convergence\n"
        "- Bias Correction: Compensates for initialization bias in moment estimates\n\n"
        "Choosing an optimizer: Adam is often the best default. Use SGD+Momentum for fine-tuning. "
        "RMSprop works well for RNNs. Consider the landscape: convex (SGD OK), non-convex (adaptive methods better)."
    ),
    pros=[
        "Adam: Fast convergence, works well with sparse gradients, adaptive per-parameter learning rates",
        "Momentum: Accelerates convergence, reduces oscillations, overcomes local minima",
        "RMSprop: Handles non-stationary objectives, good for RNNs",
        "Adagrad: No manual learning rate tuning needed, good for sparse data",
        "All variants scale to large networks and datasets"
    ],
    cons=[
        "Adam: Can generalize worse than SGD+Momentum in some cases, requires more memory",
        "SGD: Requires careful learning rate tuning, slow convergence",
        "Momentum: Additional hyperparameter (momentum coefficient) to tune",
        "Adagrad: Learning rate can decay too aggressively, not suitable for deep networks",
        "All adaptive methods: Higher memory footprint (store per-parameter statistics)"
    ],
    related_algorithms=["adam-optimizer", "learning-rate-scheduling", "feedforward-nn", "backpropagation"]
)

AlgorithmRegistry.register(gradient_descent_metadata)


@router.post("/transformer/train", response_model=TransformerResponse)
async def train_transformer(request: TransformerRequest) -> TransformerResponse:
    """Train a Transformer model on sequence-to-sequence task.

    This endpoint trains a Transformer model on a simple sequence reversal task,
    demonstrating the power of attention mechanisms. Returns predictions,
    attention weights, and training metrics.

    Args:
        request: Transformer training parameters including d_model, nhead,
                num_layers, dim_feedforward, learning_rate, epochs, dropout

    Returns:
        TransformerResponse with training results, predictions, attention weights,
        and visualization data

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training Transformer with parameters: {request.model_dump()}")

        # Validate d_model is divisible by nhead
        if request.d_model % request.nhead != 0:
            raise ValueError(f"d_model ({request.d_model}) must be divisible by nhead ({request.nhead})")

        # Generate sequence reversal data
        data = generate_reversal_data(
            n_samples=1000,
            seq_length=10,
            vocab_size=20,
            train_split=0.8,
            random_state=request.random_state
        )

        src_sequences = torch.from_numpy(data['src_sequences']).long()
        tgt_sequences = torch.from_numpy(data['tgt_sequences']).long()

        # Create datasets
        train_dataset = SequenceDataset(
            src_sequences[data['train_indices']],
            tgt_sequences[data['train_indices']]
        )
        test_dataset = SequenceDataset(
            src_sequences[data['test_indices']],
            tgt_sequences[data['test_indices']]
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize Transformer model
        model = TransformerModel(
            vocab_size=data['vocab_size'],
            d_model=request.d_model,
            nhead=request.nhead,
            num_layers=request.num_layers,
            dim_feedforward=request.dim_feedforward,
            learning_rate=request.learning_rate,
            dropout=request.dropout,
            random_state=request.random_state
        )

        # Train model
        training_results = model.train(
            train_loader=train_loader,
            epochs=request.epochs,
            test_loader=test_loader
        )

        # Get predictions on test samples
        test_samples = test_dataset.src_sequences[:5]
        test_targets = test_dataset.tgt_sequences[:5]
        predictions, attention_weights = model.predict(
            test_samples,
            max_length=data['seq_length'],
            return_attention=True
        )

        # Convert to numpy for response
        predictions_np = predictions.cpu().numpy()
        test_samples_np = test_samples.cpu().numpy()
        test_targets_np = test_targets.cpu().numpy()

        # Compute accuracy on test set
        test_metrics = training_results.get('test_metrics', {})
        accuracy = test_metrics.get('accuracy', 0.0)
        test_loss = test_metrics.get('loss', 0.0)
        perplexity = test_metrics.get('perplexity', 0.0)

        # Prepare visualization data
        visualization_data = {
            "training_loss": [
                {"epoch": i + 1, "loss": loss}
                for i, loss in enumerate(training_results['training_history'])
            ],
            "sequence_predictions": [
                {
                    "input": test_samples_np[i].tolist(),
                    "predicted": predictions_np[i, 1:].tolist(),  # Skip start token
                    "actual": test_targets_np[i].tolist(),
                    "correct": (predictions_np[i, 1:] == test_targets_np[i]).all()
                }
                for i in range(min(5, len(test_samples_np)))
            ],
            "attention_heatmap": {
                "weights": attention_weights[0].cpu().numpy().tolist() if attention_weights is not None else None,
                "input_tokens": test_samples_np[0].tolist(),
                "output_tokens": predictions_np[0, 1:].tolist()
            },
            "architecture": {
                "encoder_layers": request.num_layers,
                "decoder_layers": request.num_layers,
                "attention_heads": request.nhead,
                "d_model": request.d_model,
                "dim_feedforward": request.dim_feedforward
            }
        }

        # Prepare metrics
        metrics = {
            "accuracy": accuracy,
            "test_loss": test_loss,
            "perplexity": perplexity,
            "final_train_loss": training_results['final_loss']
        }

        # Get model info
        model_info = model.get_model_info()
        model_info['num_layers'] = request.num_layers
        model_info['nhead'] = request.nhead
        model_info['dim_feedforward'] = request.dim_feedforward

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Transformer training completed in {execution_time_ms:.2f}ms. "
            f"Test accuracy: {accuracy:.4f}, Test loss: {test_loss:.4f}"
        )

        return TransformerResponse(
            success=True,
            metrics=metrics,
            training_history=training_results['training_history'],
            predictions=predictions_np[:, 1:].tolist(),  # Skip start token
            actual=test_targets_np.tolist(),
            input_sequences=test_samples_np.tolist(),
            attention_weights=attention_weights.cpu().numpy().tolist() if attention_weights is not None else None,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                "d_model": request.d_model,
                "nhead": request.nhead,
                "num_layers": request.num_layers,
                "dim_feedforward": request.dim_feedforward,
                "learning_rate": request.learning_rate,
                "epochs": request.epochs,
                "dropout": request.dropout
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/transformer/info")
async def get_transformer_info() -> Dict[str, Any]:
    """Get Transformer algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Transformer.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("transformer")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Transformer metadata not found"
        )

    dataset_info = get_transformer_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Batch Normalization metadata
batch_norm_metadata = AlgorithmMetadata(
    id="batch-normalization",
    name="Batch Normalization",
    slug="batch-normalization",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Normalize layer inputs to stabilize and accelerate training",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "normalization", "training", "optimization"],
    use_cases=[
        "Deep network training",
        "Faster convergence",
        "Higher learning rates",
        "Covariate shift reduction",
        "Gradient flow improvement"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*d) per batch",
        space="O(d) for running stats"
    ),
    parameters=[
        AlgorithmParameter(
            name="momentum",
            label="Running Stats Momentum",
            type="range",
            default=0.1,
            min=0.01,
            max=0.5,
            step=0.01,
            description="Momentum for updating running mean and variance"
        ),
        AlgorithmParameter(
            name="eps",
            label="Epsilon (Numerical Stability)",
            type="range",
            default=1e-5,
            min=1e-8,
            max=1e-3,
            step=1e-6,
            description="Small constant added to variance for numerical stability"
        ),
        AlgorithmParameter(
            name="affine",
            label="Learnable Scale/Shift",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Whether to learn affine parameters (gamma and beta)"
        ),
        AlgorithmParameter(
            name="track_running_stats",
            label="Track Running Statistics",
            type="select",
            default=True,
            options=[
                {"label": "Yes", "value": True},
                {"label": "No", "value": False}
            ],
            description="Whether to track running mean and variance for inference"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=50,
            min=10,
            max=200,
            step=10,
            description="Number of training epochs for comparison"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.01,
            min=0.0001,
            max=0.1,
            step=0.001,
            description="Learning rate for SGD optimizer"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=32,
            options=[
                {"label": "8", "value": 8},
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128}
            ],
            description="Training batch size"
        ),
        AlgorithmParameter(
            name="hidden_size",
            label="Hidden Layer Size",
            type="range",
            default=64,
            min=32,
            max=256,
            step=16,
            description="Size of hidden layers in the network"
        )
    ],
    dataset_name="synthetic_classification",
    visualization_type="training_curves,activation_distributions,gradient_flow,convergence_comparison",
    theory=(
        "Batch Normalization is a technique that normalizes layer inputs by re-centering and "
        "re-scaling based on mini-batch statistics. For a layer with d-dimensional input x = (x₁...xd), "
        "BN normalizes each dimension:\n\n"
        "x̂ᵢ = (xᵢ - E[xᵢ]) / √(Var[xᵢ] + ε)\n\n"
        "where the mean E[xᵢ] and variance Var[xᵢ] are computed over the mini-batch. "
        "The ε term ensures numerical stability.\n\n"
        "Key Components:\n"
        "1. Normalization: Standardizes inputs to zero mean and unit variance\n"
        "2. Scale and Shift: Learnable parameters γ (scale) and β (shift) allow the network to "
        "undo the normalization if needed: yᵢ = γx̂ᵢ + β\n"
        "3. Running Statistics: During training, exponential moving averages of mean and variance "
        "are maintained for use during inference\n\n"
        "Benefits:\n"
        "- Reduces internal covariate shift (change in input distributions to layers)\n"
        "- Allows higher learning rates without divergence\n"
        "- Acts as a regularizer, reducing the need for dropout\n"
        "- Accelerates training and improves gradient flow\n"
        "- Makes networks less sensitive to initialization\n\n"
        "During training, BN uses batch statistics. During inference, it uses the running "
        "statistics computed during training to ensure consistent behavior."
    ),
    pros=[
        "Significantly faster convergence (often 2-3x speedup)",
        "Enables higher learning rates without instability",
        "Improves gradient flow through deep networks",
        "Reduces sensitivity to weight initialization",
        "Acts as implicit regularization",
        "Reduces internal covariate shift"
    ],
    cons=[
        "Adds computational overhead (normalizing each batch)",
        "Behavior differs between training and inference modes",
        "Less effective with small batch sizes",
        "Can be problematic with recurrent architectures",
        "Adds hyperparameters (momentum, epsilon) to tune"
    ],
    related_algorithms=["layer-normalization", "instance-normalization", "group-normalization", "dropout"]
)

AlgorithmRegistry.register(batch_norm_metadata)


@router.post("/batch-normalization/train", response_model=BatchNormResponse)
async def train_batch_normalization(request: BatchNormRequest) -> BatchNormResponse:
    """Train and compare networks with and without Batch Normalization.

    This endpoint trains two identical neural networks on synthetic data:
    one with batch normalization and one without. It demonstrates the benefits
    of BN through direct comparison of training curves, convergence speed,
    activation distributions, and gradient flow.

    Args:
        request: Batch Normalization parameters including momentum, eps,
                affine, track_running_stats, epochs, learning_rate, batch_size,
                and hidden_size

    Returns:
        BatchNormResponse with comparison metrics, training curves, activation
        statistics, gradient flow analysis, and convergence comparison

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training Batch Normalization comparison with parameters: {request.model_dump()}")

        # Create and train models
        model = BatchNormModel(request)
        result = model.train()

        logger.info(
            f"Batch Normalization training completed in {result['execution_time_ms']:.2f}ms. "
            f"With BN final loss: {result['metrics']['with_bn_final_loss']:.4f}, "
            f"Without BN final loss: {result['metrics']['without_bn_final_loss']:.4f}, "
            f"Improvement: {result['metrics']['improvement_percent']:.2f}%"
        )

        return BatchNormResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/batch-normalization/info")
async def get_batch_normalization_info() -> Dict[str, Any]:
    """Get Batch Normalization algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Batch Normalization.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("batch-normalization")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Batch Normalization metadata not found"
        )

    dataset_info = get_batch_norm_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/vgg/predict", response_model=VGGResponse)
async def predict_vgg(request: VGGRequest) -> VGGResponse:
    """Run VGG inference for image classification.

    This endpoint uses a pre-trained VGG model to classify images from the
    ImageNet dataset. VGG networks are characterized by their use of small
    3x3 convolution filters throughout the architecture. Returns top-K
    predictions with confidence scores, feature maps from key layers,
    and detailed architecture information.

    Args:
        request: VGG inference parameters including model_variant (vgg16/vgg19),
                top_k, use_pretrained, batch_norm, and image selection

    Returns:
        VGGResponse with predictions, confidence scores, feature maps,
        convolutional block information, and architecture visualization

    Raises:
        HTTPException: If inference fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Running VGG inference with parameters: {request.model_dump()}")

        # Initialize VGG model
        model = VGGModel(
            model_variant=request.model_variant,
            use_pretrained=request.use_pretrained,
            batch_norm=request.batch_norm
        )

        # Run inference
        response = model.run_inference(request)

        logger.info(
            f"VGG inference completed in {response.execution_time_ms:.2f}ms. "
            f"Top prediction: {response.predictions[0].class_name} "
            f"(confidence: {response.predictions[0].confidence:.4f})"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/vgg/info")
async def get_vgg_info() -> Dict[str, Any]:
    """Get VGG algorithm information and metadata.

    Returns metadata, parameters, and dataset information for VGG Network.
    Includes details about the VGG-16 and VGG-19 architectures, their use
    of small 3x3 filters, and applications in transfer learning and style transfer.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("vgg")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="VGG metadata not found"
        )

    dataset_info = get_vgg_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/activation-functions/compute", response_model=ActivationFunctionsResponse)
async def compute_activation_functions_endpoint(
    request: ActivationFunctionsRequest
) -> ActivationFunctionsResponse:
    """Compute and visualize activation functions.

    This endpoint computes various activation functions (ReLU, Leaky ReLU, Sigmoid,
    Tanh, ELU, Swish) and their derivatives over a specified range. It demonstrates
    the properties of each function, including the dead neuron problem with ReLU.

    Args:
        request: Activation functions parameters including function_type, alpha,
                input_range, compare_all, and num_points

    Returns:
        ActivationFunctionsResponse with function data, derivatives, comparison table,
        dead neuron demonstration, and visualization data

    Raises:
        HTTPException: If computation fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Computing activation functions with parameters: {request.model_dump()}")

        # Compute activation functions
        result = compute_activation_functions(
            function_type=request.function_type,
            alpha=request.alpha,
            input_range=request.input_range,
            compare_all=request.compare_all,
            num_points=request.num_points
        )

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Activation functions computation completed in {execution_time_ms:.2f}ms. "
            f"Generated {request.num_points} points for "
            f"{len(result['function_data'])} functions"
        )

        return ActivationFunctionsResponse(
            success=result['success'],
            function_data=result['function_data'],
            comparison_table=result['comparison_table'],
            dead_neuron_demo=result['dead_neuron_demo'],
            visualization_data=result['visualization_data'],
            execution_time_ms=execution_time_ms,
            parameters_used=result['parameters_used']
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Computation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@router.get("/activation-functions/info")
async def get_activation_functions_info() -> Dict[str, Any]:
    """Get Activation Functions algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Activation Functions.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("activation-functions")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Activation Functions metadata not found"
        )

    dataset_info = get_activation_functions_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Convolutional Layers metadata
convolutional_layers_metadata = AlgorithmMetadata(
    id="convolutional-layers",
    name="Convolutional Layers",
    slug="convolutional-layers",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Demonstrate convolution operation and filters for feature extraction",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["deep-learning", "cnn", "convolution", "feature-extraction"],
    use_cases=[
        "Image feature extraction",
        "Edge detection",
        "Pattern recognition",
        "Object detection preprocessing",
        "CNNs building block"
    ],
    complexity=AlgorithmComplexity(
        time="O(H*W*K²*F)",
        space="O(H*W*F)"
    ),
    parameters=[
        AlgorithmParameter(
            name="num_filters",
            label="Number of Filters",
            type="range",
            default=32,
            min=8,
            max=128,
            step=8,
            description="Number of convolutional filters to learn"
        ),
        AlgorithmParameter(
            name="kernel_size",
            label="Kernel Size",
            type="select",
            default=3,
            options=[
                {"label": "3x3", "value": 3},
                {"label": "5x5", "value": 5},
                {"label": "7x7", "value": 7}
            ],
            description="Size of the convolution kernel"
        ),
        AlgorithmParameter(
            name="stride",
            label="Stride",
            type="select",
            default=1,
            options=[
                {"label": "1", "value": 1},
                {"label": "2", "value": 2},
                {"label": "3", "value": 3}
            ],
            description="Stride for convolution operation"
        ),
        AlgorithmParameter(
            name="padding",
            label="Padding",
            type="select",
            default="same",
            options=[
                {"label": "Same (preserve size)", "value": "same"},
                {"label": "Valid (no padding)", "value": "valid"}
            ],
            description="Padding type for convolution"
        ),
        AlgorithmParameter(
            name="activation",
            label="Activation Function",
            type="select",
            default="relu",
            options=[
                {"label": "ReLU", "value": "relu"},
                {"label": "Tanh", "value": "tanh"},
                {"label": "None", "value": "none"}
            ],
            description="Activation function to apply"
        )
    ],
    dataset_name="sample_image",
    visualization_type="input_image,filter_kernels,feature_maps,activation_patterns,common_filters",
    theory=(
        "Convolutional layers are the fundamental building blocks of Convolutional Neural Networks (CNNs). "
        "They perform a mathematical operation called convolution, which applies a small filter (kernel) "
        "across the input image to extract features.\n\n"
        "How Convolution Works:\n"
        "1. A filter (e.g., 3x3 matrix) slides across the input image\n"
        "2. At each position, element-wise multiplication is performed\n"
        "3. Results are summed to produce a single output value\n"
        "4. This creates a feature map highlighting detected patterns\n\n"
        "Key Concepts:\n"
        "- Filters/Kernels: Small matrices that detect specific features (edges, textures, patterns)\n"
        "- Stride: How many pixels the filter moves at each step\n"
        "- Padding: Adding zeros around the image border to control output size\n"
        "- Feature Maps: Output activations showing where features were detected\n\n"
        "Common Filter Types:\n"
        "- Sobel: Detects edges in horizontal/vertical directions\n"
        "- Gaussian: Smooths and reduces noise\n"
        "- Sharpen: Enhances edges and details\n"
        "- Laplacian: Detects edges in all directions\n\n"
        "Output Size Calculation:\n"
        "output_size = (input_size + 2*padding - kernel_size) / stride + 1\n\n"
        "CNNs learn optimal filter weights through backpropagation, automatically discovering "
        "which features are most useful for the task. Early layers typically learn low-level "
        "features (edges, corners), while deeper layers learn high-level features (shapes, objects)."
    ),
    pros=[
        "Automatic feature extraction without manual engineering",
        "Translation invariance through weight sharing",
        "Fewer parameters than fully connected layers",
        "Captures spatial relationships in images",
        "Foundation for modern computer vision"
    ],
    cons=[
        "Requires understanding of hyperparameters (kernel size, stride, padding)",
        "Output size depends on architecture choices",
        "May need many filters for complex tasks",
        "Computationally intensive for large images",
        "Filter interpretation can be challenging"
    ],
    related_algorithms=["cnn", "pooling", "batch-normalization", "resnet"]
)

AlgorithmRegistry.register(convolutional_layers_metadata)


@router.post("/convolutional-layers/demo", response_model=ConvolutionalLayersResponse)
async def demonstrate_convolutional_layers(request: ConvolutionalLayersRequest) -> ConvolutionalLayersResponse:
    """Demonstrate convolutional layer operations and filters.

    This endpoint shows how convolution operations work by:
    - Applying learnable filters to an image
    - Visualizing filter kernels (weights)
    - Showing resulting feature maps
    - Demonstrating common predefined filters (Sobel, Gaussian, etc.)
    - Computing output dimensions based on stride and padding

    Args:
        request: Convolution parameters including num_filters, kernel_size,
                stride, padding, and activation function

    Returns:
        ConvolutionalLayersResponse with input image, filter visualizations,
        feature maps, common filter results, and dimension calculations

    Raises:
        HTTPException: If demonstration fails due to invalid parameters or errors
    """
    try:
        logger.info(f"Running Convolutional Layers demo with parameters: {request.model_dump()}")

        # Initialize model
        model = ConvolutionalLayersModel(
            num_filters=request.num_filters,
            kernel_size=request.kernel_size,
            stride=request.stride,
            padding=request.padding,
            activation=request.activation,
            random_state=request.random_state
        )

        # Run demonstration
        response = model.run_demonstration(request)

        logger.info(
            f"Convolutional Layers demo completed in {response.execution_time_ms:.2f}ms. "
            f"Output dimensions: {response.output_dimensions['height']}x"
            f"{response.output_dimensions['width']}x{response.output_dimensions['channels']}"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Demonstration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Demonstration failed: {str(e)}")


@router.get("/convolutional-layers/info")
async def get_convolutional_layers_info() -> Dict[str, Any]:
    """Get Convolutional Layers algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Convolutional Layers.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("convolutional-layers")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Convolutional Layers metadata not found"
        )

    dataset_info = get_conv_layers_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Dropout metadata
dropout_metadata = AlgorithmMetadata(
    id="dropout",
    name="Dropout Regularization",
    slug="dropout",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Randomly drop neurons during training to prevent overfitting",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["deep-learning", "regularization", "overfitting", "training"],
    use_cases=[
        "Overfitting prevention",
        "Model ensemble (implicit)",
        "Robust feature learning",
        "Regularization alternative to weight decay",
        "Deep network training"
    ],
    complexity=AlgorithmComplexity(
        time="O(1) per neuron",
        space="O(1)"
    ),
    parameters=[
        AlgorithmParameter(
            name="dropout_rate",
            label="Dropout Rate",
            type="range",
            default=0.5,
            min=0.0,
            max=0.9,
            step=0.1,
            description="Probability of dropping neurons during training"
        ),
        AlgorithmParameter(
            name="apply_to_layers",
            label="Apply to Layers",
            type="text",
            default=["hidden1", "hidden2"],
            description="Which layers get dropout (e.g., hidden1,hidden2)"
        ),
        AlgorithmParameter(
            name="training_epochs",
            label="Training Epochs",
            type="range",
            default=100,
            min=20,
            max=300,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="hidden_layers",
            label="Hidden Layer Sizes",
            type="text",
            default=[128, 64],
            description="Sizes of hidden layers (e.g., 128,64)"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.1,
            step=0.0001,
            description="Learning rate for optimizer"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=32,
            options=[
                {"label": "8", "value": 8},
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128}
            ],
            description="Batch size for training"
        )
    ],
    dataset_name="overfitting_demo",
    visualization_type="training_curves,dropout_comparison,overfitting_metrics,dropout_masks",
    theory=(
        "Dropout is a powerful regularization technique that prevents neural networks from "
        "overfitting by randomly dropping (setting to zero) a subset of neurons during training. "
        "During each training iteration, each neuron has a probability p (dropout rate) of being "
        "temporarily removed from the network.\n\n"
        "Key Concepts:\n"
        "1. During Training: At each iteration, randomly drop neurons with probability p\n"
        "2. During Inference: Use all neurons but scale their outputs by (1-p)\n"
        "3. Inverted Dropout: Scale activations by 1/(1-p) during training instead\n"
        "4. Implicit Ensemble: Dropout trains an ensemble of 2^n networks (where n = number of neurons)\n\n"
        "How Dropout Prevents Overfitting:\n"
        "- Forces neurons to not rely on specific other neurons being present\n"
        "- Prevents co-adaptation of neurons (complex dependencies)\n"
        "- Creates more robust and generalizable features\n"
        "- Acts as model averaging over exponentially many networks\n\n"
        "Mathematical Formulation:\n"
        "For a layer with activation h, during training:\n"
        "- Sample mask: m ~ Bernoulli(1-p)\n"
        "- Apply dropout: h_drop = m ⊙ h / (1-p)  (inverted dropout)\n"
        "During inference: h_inference = h (no dropout, already scaled)\n\n"
        "Common dropout rates: 0.2-0.5 for hidden layers, 0.5-0.8 for input layers.\n"
        "Dropout is most effective when training data is limited and the network is large enough "
        "to memorize training examples."
    ),
    pros=[
        "Simple and effective regularization technique",
        "Prevents overfitting without requiring more data",
        "Computationally efficient (only random sampling)",
        "Works as implicit ensemble of many networks",
        "Reduces co-adaptation of neurons (more robust features)",
        "Often eliminates need for other regularization methods"
    ],
    cons=[
        "Increases training time (typically 2-3x more epochs needed)",
        "Introduces hyperparameter (dropout rate) to tune",
        "Can hurt performance if applied incorrectly",
        "Less effective with batch normalization",
        "Not suitable for very small networks or datasets",
        "Different behavior between training and inference modes"
    ],
    related_algorithms=["batch-normalization", "weight-decay", "early-stopping", "data-augmentation"]
)

AlgorithmRegistry.register(dropout_metadata)


@router.post("/dropout/train", response_model=DropoutResponse)
async def train_dropout(request: DropoutRequest) -> DropoutResponse:
    """Train networks with different dropout rates to demonstrate regularization effects.

    This endpoint trains multiple neural networks on a small dataset that's prone to
    overfitting. It compares networks with no dropout, light dropout (0.2), standard
    dropout (0.5), and heavy dropout (0.8) to show how dropout prevents overfitting
    and improves generalization.

    Args:
        request: Dropout training parameters including dropout_rate, apply_to_layers,
                training_epochs, hidden_layers, learning_rate, batch_size, and random_state

    Returns:
        DropoutResponse with training curves, dropout comparison metrics, overfitting
        gap analysis, dropout mask visualizations, and comprehensive visualization data

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training Dropout demonstration with parameters: {request.model_dump()}")

        # Generate overfitting-prone dataset
        data = generate_overfitting_prone_dataset(
            n_samples=200,
            n_features=20,
            n_informative=10,
            n_classes=2,
            test_size=0.3,
            random_state=request.random_state
        )

        # Initialize dropout model
        model = DropoutModel(
            dropout_rate=request.dropout_rate,
            apply_to_layers=request.apply_to_layers,
            hidden_layers=request.hidden_layers,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            random_state=request.random_state
        )

        # Train models with different dropout rates
        training_results = model.train(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            epochs=request.training_epochs
        )

        # Get comparison metrics
        dropout_comparison = model.get_comparison_metrics()

        # Get dropout masks for visualization
        dropout_masks = model.get_dropout_masks(n_samples=3)

        # Prepare training curves
        epochs_list = list(range(1, request.training_epochs + 1))
        training_curves = {
            'epochs': epochs_list
        }

        # Add curves for each configuration
        for config_name, history in model.training_histories.items():
            training_curves[f'{config_name}_train_loss'] = history['train_loss']
            training_curves[f'{config_name}_val_loss'] = history['val_loss']
            training_curves[f'{config_name}_train_acc'] = history['train_accuracy']
            training_curves[f'{config_name}_val_acc'] = history['val_accuracy']

        # Calculate overfitting metrics
        overfitting_metrics = {}
        for config_name, history in model.training_histories.items():
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            overfitting_gap = final_train_loss - final_val_loss

            overfitting_metrics[config_name] = {
                'train_loss': final_train_loss,
                'val_loss': final_val_loss,
                'overfitting_gap': overfitting_gap,
                'train_accuracy': history['train_accuracy'][-1],
                'val_accuracy': history['val_accuracy'][-1],
                'generalization_gap': history['train_accuracy'][-1] - history['val_accuracy'][-1]
            }

        # Prepare comprehensive metrics
        metrics = {
            config_name: {
                'train_accuracy': overfitting_metrics[config_name]['train_accuracy'],
                'val_accuracy': overfitting_metrics[config_name]['val_accuracy'],
                'overfitting_gap': overfitting_metrics[config_name]['overfitting_gap'],
                'generalization_gap': overfitting_metrics[config_name]['generalization_gap']
            }
            for config_name in overfitting_metrics.keys()
        }

        # Prepare visualization data
        visualization_data = {
            'n_samples_train': len(data['X_train']),
            'n_samples_val': len(data['X_val']),
            'n_samples_test': len(data['X_test']),
            'n_features': data['n_features'],
            'n_classes': data['n_classes'],
            'configurations_compared': len(model.training_histories),
            'dropout_rates_tested': [0.0, 0.2, 0.5, 0.8]
        }

        # Get model info
        model_info = model.get_model_info()

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Dropout demonstration completed in {execution_time_ms:.2f}ms. "
            f"No dropout overfitting gap: {overfitting_metrics['no_dropout']['overfitting_gap']:.4f}, "
            f"With dropout (0.5) overfitting gap: {overfitting_metrics['dropout_0.5']['overfitting_gap']:.4f}"
        )

        return DropoutResponse(
            success=True,
            metrics=metrics,
            training_curves=training_curves,
            dropout_comparison=dropout_comparison,
            overfitting_metrics=overfitting_metrics,
            dropout_masks=dropout_masks,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'dropout_rate': request.dropout_rate,
                'apply_to_layers': request.apply_to_layers,
                'training_epochs': request.training_epochs,
                'hidden_layers': request.hidden_layers,
                'learning_rate': request.learning_rate,
                'batch_size': request.batch_size,
                'dataset_name': request.dataset_name
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/dropout/info")
async def get_dropout_info() -> Dict[str, Any]:
    """Get Dropout algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Dropout Regularization.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("dropout")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Dropout metadata not found"
        )

    dataset_info = get_dropout_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


@router.post("/adam-optimizer/optimize", response_model=AdamOptimizerResponse)
async def optimize_with_adam(request: AdamOptimizerRequest) -> AdamOptimizerResponse:
    """Demonstrate Adam Optimizer on synthetic optimization landscapes.

    This endpoint runs Adam optimization on 2D non-convex test functions
    (Rosenbrock, Beale, or Himmelblau) and compares its performance with
    other popular optimizers (SGD, Momentum, RMSprop). Returns optimization
    trajectories, loss convergence curves, momentum visualizations, and
    detailed comparison metrics.

    Args:
        request: Adam Optimizer parameters including learning_rate, beta1, beta2,
                epsilon, compare_optimizers, max_iterations, and function_type

    Returns:
        AdamOptimizerResponse with optimization trajectories, convergence data,
        momentum visualizations, and performance comparison metrics

    Raises:
        HTTPException: If optimization fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Running Adam Optimizer with parameters: {request.model_dump()}")

        # Create and run optimization
        model = AdamOptimizerModel(request)
        response = model.train()

        if not response.success:
            logger.error(f"Adam optimization failed: {response.error}")
            raise HTTPException(status_code=400, detail=response.error)

        logger.info(
            f"Adam optimization completed in {response.execution_time_ms:.2f}ms. "
            f"Adam final loss: {response.metrics.adam_final_loss:.6f}, "
            f"Iterations: {response.metrics.adam_iterations}"
        )

        if response.metrics.convergence_improvement is not None:
            logger.info(
                f"Adam improvement over SGD: {response.metrics.convergence_improvement:.2f}%"
            )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/adam-optimizer/info")
async def get_adam_optimizer_info() -> Dict[str, Any]:
    """Get Adam Optimizer algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Adam Optimizer.
    Includes detailed theory about adaptive learning rates, momentum terms,
    and bias correction.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("adam-optimizer")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Adam Optimizer metadata not found"
        )

    dataset_info = get_adam_optimizer_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Data Augmentation metadata
data_augmentation_metadata = AlgorithmMetadata(
    id="data-augmentation",
    name="Data Augmentation",
    slug="data-augmentation",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Generate training data variations to improve model generalization",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["deep-learning", "preprocessing", "augmentation", "generalization"],
    use_cases=[
        "Training data expansion",
        "Overfitting prevention",
        "Model robustness improvement",
        "Limited dataset handling",
        "Real-world variation simulation"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*transforms)",
        space="O(n*augmented_copies)"
    ),
    parameters=[
        AlgorithmParameter(
            name="augmentation_types",
            label="Augmentation Types",
            type="text",
            default=["rotation", "flip_horizontal", "brightness"],
            description="Augmentation types to apply (rotation, flip_horizontal, flip_vertical, brightness, contrast, blur, crop, scale, noise, color_jitter)"
        ),
        AlgorithmParameter(
            name="rotation_range",
            label="Rotation Range (degrees)",
            type="range",
            default=30.0,
            min=0.0,
            max=180.0,
            step=5.0,
            description="Maximum rotation angle in degrees"
        ),
        AlgorithmParameter(
            name="brightness_factor",
            label="Brightness Factor",
            type="range",
            default=0.3,
            min=0.0,
            max=1.0,
            step=0.1,
            description="Brightness adjustment factor (0.0 to 1.0)"
        ),
        AlgorithmParameter(
            name="num_augmented",
            label="Number of Augmented Copies",
            type="range",
            default=9,
            min=1,
            max=16,
            step=1,
            description="Number of augmented copies to generate"
        )
    ],
    dataset_name="sample_image",
    visualization_type="original_image,augmented_gallery,augmentation_comparison,before_after",
    theory=(
        "Data Augmentation is a technique used to artificially expand training datasets by creating "
        "modified versions of existing images. This helps neural networks learn more robust features "
        "and generalize better to unseen data, especially when training data is limited.\n\n"
        "Why Data Augmentation Works:\n"
        "1. Increases Effective Dataset Size: Creates many variations from limited data\n"
        "2. Improves Generalization: Model learns invariance to transformations\n"
        "3. Reduces Overfitting: Prevents memorization of training examples\n"
        "4. Simulates Real-World Variations: Rotation, lighting, position changes\n\n"
        "Common Augmentation Techniques:\n\n"
        "Geometric Transformations:\n"
        "- Rotation: Rotate images by random angles (-30° to +30°)\n"
        "- Flipping: Horizontal/vertical flips for mirror images\n"
        "- Scaling: Zoom in/out to simulate distance changes\n"
        "- Cropping: Random crops to simulate different viewpoints\n\n"
        "Color Transformations:\n"
        "- Brightness: Adjust overall image brightness\n"
        "- Contrast: Modify difference between light and dark areas\n"
        "- Color Jitter: Randomly shift RGB channels\n"
        "- Saturation: Change color intensity\n\n"
        "Noise and Blur:\n"
        "- Gaussian Noise: Add random noise to simulate sensor noise\n"
        "- Gaussian Blur: Blur images to simulate motion or focus issues\n\n"
        "Implementation Considerations:\n"
        "- Apply augmentations during training only (not test time)\n"
        "- Use random parameters for each image\n"
        "- Combine multiple augmentations for diversity\n"
        "- Ensure augmentations preserve label validity\n"
        "- Balance augmentation intensity (too much can harm learning)\n\n"
        "Modern frameworks like PyTorch (torchvision.transforms) and TensorFlow (tf.image) "
        "provide efficient implementations that can be applied on-the-fly during training."
    ),
    pros=[
        "Increases effective training dataset size without collecting new data",
        "Reduces overfitting and improves generalization",
        "Makes models robust to transformations and variations",
        "Simple to implement with existing libraries",
        "Can be applied on-the-fly during training (no storage overhead)",
        "Domain-specific augmentations can encode prior knowledge"
    ],
    cons=[
        "Increases training time (applying transformations)",
        "Must choose appropriate augmentations for the task",
        "Excessive augmentation can introduce unrealistic examples",
        "Some augmentations may not preserve labels (e.g., vertical flip for text)",
        "Hyperparameter tuning needed for augmentation parameters",
        "May not help if model capacity is the bottleneck"
    ],
    related_algorithms=["dropout", "batch-normalization", "cnn", "transfer-learning"]
)

AlgorithmRegistry.register(data_augmentation_metadata)


@router.post("/data-augmentation/augment", response_model=DataAugmentationResponse)
async def augment_data(request: DataAugmentationRequest) -> DataAugmentationResponse:
    """Generate augmented image variations for data augmentation demonstration.

    This endpoint demonstrates data augmentation by applying various image transformations
    to a sample image. It shows how augmentation creates diverse training examples from
    a single image, helping models learn robust features and generalize better.

    Supported augmentation techniques:
    - Geometric: rotation, flip_horizontal, flip_vertical, crop, scale
    - Color: brightness, contrast, color_jitter
    - Noise: noise, blur

    Args:
        request: Data augmentation parameters including augmentation_types,
                rotation_range, brightness_factor, num_augmented, and random_state

    Returns:
        DataAugmentationResponse with original image, augmented variations,
        augmentation descriptions, and visualization data

    Raises:
        HTTPException: If augmentation fails due to invalid parameters or errors
    """
    try:
        logger.info(f"Running Data Augmentation with parameters: {request.model_dump()}")

        # Initialize data augmentation model
        model = DataAugmentationModel(request)

        # Generate augmented images
        response = model.augment()

        logger.info(
            f"Data Augmentation completed in {response.execution_time_ms:.2f}ms. "
            f"Generated {len(response.augmented_images)} augmented images using "
            f"{len(request.augmentation_types)} augmentation techniques"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Augmentation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Augmentation failed: {str(e)}")


@router.get("/data-augmentation/info")
async def get_data_augmentation_info() -> Dict[str, Any]:
    """Get Data Augmentation algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Data Augmentation.
    Includes details about supported augmentation techniques, their effects on
    training, and best practices for using augmentation in deep learning.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("data-augmentation")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Data Augmentation metadata not found"
        )

    dataset_info = get_data_augmentation_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Pooling Layers metadata
pooling_layers_metadata = AlgorithmMetadata(
    id="pooling-layers",
    name="Pooling Layers (Max/Average)",
    slug="pooling-layers",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Downsample feature maps to reduce spatial dimensions",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["deep-learning", "cnn", "pooling", "downsampling"],
    use_cases=[
        "Spatial dimension reduction",
        "Translation invariance",
        "Computational efficiency",
        "Feature hierarchy",
        "Overfitting prevention"
    ],
    complexity=AlgorithmComplexity(
        time="O(H*W*K²) where H,W=input size, K=pool_size",
        space="O(H/K * W/K)"
    ),
    parameters=[
        AlgorithmParameter(
            name="pool_type",
            label="Pooling Method",
            type="select",
            default="max",
            options=[
                {"label": "Max Pooling", "value": "max"},
                {"label": "Average Pooling", "value": "average"},
                {"label": "Global Max Pooling", "value": "global_max"},
                {"label": "Global Average Pooling", "value": "global_average"}
            ],
            description="Type of pooling operation to apply"
        ),
        AlgorithmParameter(
            name="pool_size",
            label="Pool Size",
            type="select",
            default=2,
            options=[
                {"label": "2×2", "value": 2},
                {"label": "3×3", "value": 3},
                {"label": "4×4", "value": 4}
            ],
            description="Size of pooling window (pool_size × pool_size)"
        ),
        AlgorithmParameter(
            name="stride",
            label="Stride",
            type="select",
            default=2,
            options=[
                {"label": "1", "value": 1},
                {"label": "2", "value": 2},
                {"label": "3", "value": 3}
            ],
            description="Stride for pooling operation"
        ),
        AlgorithmParameter(
            name="padding",
            label="Padding",
            type="range",
            default=0,
            min=0,
            max=2,
            step=1,
            description="Padding to add around input feature map"
        ),
        AlgorithmParameter(
            name="input_size",
            label="Input Size",
            type="range",
            default=8,
            min=4,
            max=32,
            step=4,
            description="Size of square input feature map (input_size × input_size)"
        ),
        AlgorithmParameter(
            name="num_channels",
            label="Number of Channels",
            type="select",
            default=1,
            options=[
                {"label": "1 (Grayscale)", "value": 1},
                {"label": "3 (RGB)", "value": 3}
            ],
            description="Number of input channels"
        )
    ],
    dataset_name="sample_feature_maps",
    visualization_type="input_heatmap,output_heatmap,dimension_calculator,pooling_windows,comparison",
    theory=(
        "Pooling layers are a fundamental component of Convolutional Neural Networks (CNNs) "
        "that downsample feature maps to reduce their spatial dimensions. Pooling provides "
        "several benefits:\n\n"
        "1. Dimensionality Reduction: Reduces the spatial size of feature maps, decreasing "
        "the number of parameters and computation in the network\n"
        "2. Translation Invariance: Makes the representation approximately invariant to small "
        "translations of the input\n"
        "3. Overfitting Prevention: Acts as a form of regularization by abstracting features\n"
        "4. Feature Hierarchy: Progressively builds higher-level representations\n\n"
        "Common Pooling Operations:\n\n"
        "Max Pooling: Selects the maximum value from each pooling window\n"
        "- Most commonly used in CNNs\n"
        "- Preserves the strongest activations\n"
        "- Gradient flows only to the maximum position during backpropagation\n"
        "- Formula: output[i,j] = max(window)\n\n"
        "Average Pooling: Computes the average of all values in the pooling window\n"
        "- Provides a smoother downsampling\n"
        "- All positions contribute equally\n"
        "- Gradient is distributed equally during backpropagation\n"
        "- Formula: output[i,j] = mean(window)\n\n"
        "Global Pooling: Reduces entire feature map to a single value per channel\n"
        "- Global Max: Maximum value across entire feature map\n"
        "- Global Average: Average value across entire feature map\n"
        "- Often used before fully connected layers to eliminate spatial dimensions\n"
        "- Particularly popular in modern architectures (e.g., ResNet, Inception)\n\n"
        "Output Size Calculation:\n"
        "output_size = ((input_size + 2*padding - pool_size) / stride) + 1\n\n"
        "For example, with input_size=8, pool_size=2, stride=2, padding=0:\n"
        "output_size = ((8 + 0 - 2) / 2) + 1 = 4\n\n"
        "Pooling layers have no learnable parameters, making them computationally efficient. "
        "They are typically placed after convolutional layers to progressively reduce spatial "
        "dimensions while increasing the depth (number of feature maps)."
    ),
    pros=[
        "Reduces computational cost and memory usage",
        "Provides local translation invariance",
        "No learnable parameters (parameter-free operation)",
        "Helps prevent overfitting through feature abstraction",
        "Increases receptive field of subsequent layers",
        "Fast computation (simple operations like max or mean)"
    ],
    cons=[
        "Loses spatial information and fine details",
        "Max pooling can lose important non-maximal features",
        "Not differentiable at max positions (requires special handling)",
        "Fixed operation (cannot adapt to different feature types)",
        "Can be too aggressive for some tasks requiring precise localization",
        "Some modern architectures replace pooling with strided convolutions"
    ],
    related_algorithms=["cnn", "convolutional-layers", "stride", "downsampling"]
)

AlgorithmRegistry.register(pooling_layers_metadata)


@router.post("/pooling-layers/compute", response_model=PoolingLayersResponse)
async def compute_pooling_layers_endpoint(
    request: PoolingLayersRequest
) -> PoolingLayersResponse:
    """Compute and visualize pooling operations.

    This endpoint demonstrates various pooling operations (max, average, global) on
    feature maps. It shows the dimension reduction, pooling windows in action,
    and comparison of different pooling types on the same input.

    Args:
        request: Pooling parameters including pool_type, pool_size, stride,
                padding, input_size, num_channels, and random_state

    Returns:
        PoolingLayersResponse with input feature map, pooled output, max positions,
        dimension information, pooling windows, comparison data, and visualization data

    Raises:
        HTTPException: If computation fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Computing pooling layers with parameters: {request.model_dump()}")

        # Compute pooling
        result = compute_pooling(
            pool_type=request.pool_type,
            pool_size=request.pool_size,
            stride=request.stride,
            padding=request.padding,
            input_size=request.input_size,
            num_channels=request.num_channels,
            random_state=request.random_state
        )

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Pooling computation completed in {execution_time_ms:.2f}ms. "
            f"Input shape: {result['dimension_info']['input_shape']}, "
            f"Output shape: {result['dimension_info']['output_shape']}, "
            f"Reduction factor: {result['dimension_info']['reduction_factor']:.2f}x"
        )

        return PoolingLayersResponse(
            success=result['success'],
            input_feature_map=result['input_feature_map'],
            pooled_output=result['pooled_output'],
            max_positions=result['max_positions'],
            dimension_info=result['dimension_info'],
            pooling_windows=result['pooling_windows'],
            comparison_data=result['comparison_data'],
            visualization_data=result['visualization_data'],
            execution_time_ms=execution_time_ms,
            parameters_used=result['parameters_used']
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Computation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@router.get("/pooling-layers/info")
async def get_pooling_layers_info() -> Dict[str, Any]:
    """Get Pooling Layers algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Pooling Layers.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("pooling-layers")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Pooling Layers metadata not found"
        )

    dataset_info = get_pooling_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Learning Rate Scheduling metadata
lr_scheduling_metadata = AlgorithmMetadata(
    id="learning-rate-scheduling",
    name="Learning Rate Scheduling",
    slug="learning-rate-scheduling",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Adjust learning rate during training for better convergence",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "optimization", "hyperparameters", "training"],
    use_cases=[
        "Fine-tuning pre-trained models",
        "Training stability",
        "Faster convergence",
        "Escaping local minima",
        "Transfer learning"
    ],
    complexity=AlgorithmComplexity(
        time="O(1) per step",
        space="O(1)"
    ),
    parameters=[
        AlgorithmParameter(
            name="schedule_type",
            label="Schedule Type",
            type="select",
            default="step",
            options=[
                {"label": "Step Decay", "value": "step"},
                {"label": "Exponential Decay", "value": "exponential"},
                {"label": "Cosine Annealing", "value": "cosine"},
                {"label": "Reduce on Plateau", "value": "reduce_on_plateau"},
                {"label": "Cyclic", "value": "cyclic"}
            ],
            description="Type of learning rate schedule"
        ),
        AlgorithmParameter(
            name="initial_lr",
            label="Initial Learning Rate",
            type="range",
            default=0.1,
            min=0.001,
            max=1.0,
            step=0.001,
            description="Starting learning rate"
        ),
        AlgorithmParameter(
            name="step_size",
            label="Step Size",
            type="range",
            default=10,
            min=1,
            max=50,
            step=1,
            description="Steps before decay (for step/cyclic schedules)"
        ),
        AlgorithmParameter(
            name="gamma",
            label="Decay Factor",
            type="range",
            default=0.1,
            min=0.01,
            max=0.9,
            step=0.01,
            description="Multiplicative factor of learning rate decay"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=100,
            min=20,
            max=300,
            step=10,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="min_lr",
            label="Minimum Learning Rate",
            type="range",
            default=0.0001,
            min=0.0,
            max=0.1,
            step=0.0001,
            description="Minimum LR for cosine annealing"
        ),
        AlgorithmParameter(
            name="max_lr",
            label="Maximum Learning Rate",
            type="range",
            default=0.5,
            min=0.01,
            max=2.0,
            step=0.01,
            description="Maximum LR for cyclic schedule"
        ),
        AlgorithmParameter(
            name="patience",
            label="Patience",
            type="range",
            default=5,
            min=1,
            max=20,
            step=1,
            description="Patience for plateau-based schedule"
        )
    ],
    dataset_name="synthetic_classification",
    visualization_type="learning_rate_curves,loss_curves,convergence_comparison",
    theory=(
        "Learning Rate Scheduling adjusts the learning rate during training to improve "
        "convergence speed, stability, and final performance. The learning rate is one of the "
        "most important hyperparameters in neural network training, controlling how much to "
        "update model weights in response to the estimated error. Different schedules serve "
        "different purposes:\n\n"
        "1. Step Decay: Reduces LR by a factor (gamma) every N epochs. Simple and effective "
        "for many tasks, allowing initial fast learning followed by fine-tuning.\n\n"
        "2. Exponential Decay: Gradually decreases LR by multiplying by gamma each epoch. "
        "Provides smooth, continuous decay suitable for long training runs.\n\n"
        "3. Cosine Annealing: Follows a cosine curve from initial_lr to min_lr. Provides "
        "smooth decay with a gradual slowdown, often improving final convergence.\n\n"
        "4. Reduce on Plateau: Monitors validation loss and reduces LR when improvement stalls. "
        "Adapts to training dynamics, useful when optimal schedule is unknown.\n\n"
        "5. Cyclic Learning Rate: Cycles between base_lr and max_lr. Can help escape local "
        "minima and saddle points through periodic exploration.\n\n"
        "Benefits of LR scheduling:\n"
        "- Start with higher LR for fast initial progress\n"
        "- Reduce LR later for fine-grained convergence\n"
        "- Prevent overshooting optimal parameters\n"
        "- Improve training stability and final performance\n"
        "- Adapt learning dynamics to loss landscape\n\n"
        "The choice of schedule depends on the task, architecture, and dataset. Empirically, "
        "step decay and cosine annealing are popular for image classification, while plateau-based "
        "scheduling works well when training dynamics are unpredictable."
    ),
    pros=[
        "Improves convergence speed and final performance",
        "Prevents overshooting and oscillation around minima",
        "Enables faster initial learning followed by fine-tuning",
        "Can help escape local minima (cyclic schedules)",
        "Adapts to training dynamics (plateau-based)"
    ],
    cons=[
        "Adds hyperparameters to tune (schedule-specific)",
        "Requires understanding of training dynamics",
        "Some schedules need manual tuning (step size, gamma)",
        "May need different schedules for different tasks",
        "Cyclic schedules can be unstable without proper tuning"
    ],
    related_algorithms=["adam-optimizer", "sgd", "momentum", "training"]
)

AlgorithmRegistry.register(lr_scheduling_metadata)


@router.post("/learning-rate-scheduling/train", response_model=LearningRateSchedulingResponse)
async def train_with_lr_scheduling(request: LearningRateSchedulingRequest) -> LearningRateSchedulingResponse:
    """Train models with different learning rate schedules.

    This endpoint demonstrates various learning rate scheduling strategies by training
    simple neural networks on a synthetic classification task. It compares how different
    schedules affect convergence speed, training stability, and final performance.

    The demonstration trains separate models with:
    - Step Decay: Drops LR by gamma every step_size epochs
    - Exponential Decay: Multiplies LR by gamma each epoch
    - Cosine Annealing: Smoothly decreases LR following a cosine curve
    - Reduce on Plateau: Reduces LR when validation loss plateaus
    - Cyclic LR: Cycles between base and max learning rates

    Args:
        request: Learning rate scheduling parameters including schedule_type, initial_lr,
                step_size, gamma, epochs, min_lr, max_lr, and patience

    Returns:
        LearningRateSchedulingResponse with:
            - Learning rate curves for each schedule
            - Training loss curves for comparison
            - Convergence metrics (final loss, convergence epoch, time)
            - Comparison table showing relative performance
            - Visualization data for frontend charts

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training with LR scheduling, parameters: {request.model_dump()}")

        # Create and train models with all schedules
        model = LearningRateSchedulingModel(request)
        response = model.train()

        if not response.success:
            logger.error("LR scheduling training failed")
            raise HTTPException(status_code=400, detail="Training failed")

        logger.info(
            f"LR scheduling training completed in {response.execution_time_ms:.2f}ms. "
            f"Best schedule: {response.visualization_data['best_schedule']}"
        )

        # Log comparison results
        for entry in response.comparison_table[:3]:  # Top 3 schedules
            logger.info(
                f"{entry['schedule']}: final_loss={entry['final_loss']:.4f}, "
                f"converged at epoch {entry['convergence_epoch']}"
            )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/learning-rate-scheduling/info")
async def get_lr_scheduling_info() -> Dict[str, Any]:
    """Get Learning Rate Scheduling algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Learning Rate Scheduling.
    Includes detailed theory about different scheduling strategies and their effects
    on training dynamics.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("learning-rate-scheduling")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Learning Rate Scheduling metadata not found"
        )

    dataset_info = get_lr_scheduling_dataset_info()
    model_info = get_lr_scheduling_model_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info,
        "model": model_info
    }


@router.post("/gradient-descent/optimize", response_model=GradientDescentResponse)
async def optimize_gradient_descent(request: GradientDescentRequest) -> GradientDescentResponse:
    """Run gradient descent optimization with various algorithms.

    This endpoint demonstrates different gradient descent variants (SGD, Momentum,
    RMSprop, Adam, Adagrad) on 2D optimization test functions. It visualizes
    optimization trajectories on contour plots and compares convergence characteristics.

    Args:
        request: Gradient descent parameters including optimizer_type, learning_rate,
                momentum, iterations, compare_all, test_function, and random_state

    Returns:
        GradientDescentResponse with optimization trajectories, loss curves,
        contour data, and comparison statistics for all optimizers

    Raises:
        HTTPException: If optimization fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Running gradient descent optimization with parameters: {request.model_dump()}")

        # Create model and run optimization
        model = GradientDescentModel()
        response = model.run(request)

        logger.info(
            f"Gradient descent optimization completed in {response.execution_time_ms:.2f}ms"
        )

        if request.compare_all and response.results:
            logger.info(f"Compared {len(response.results)} optimizers:")
            for result in response.results:
                logger.info(
                    f"  {result.optimizer_name}: final_loss={result.final_loss:.6f}, "
                    f"iterations_to_converge={result.iterations_to_converge}, "
                    f"path_length={result.path_length:.4f}"
                )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/gradient-descent/info")
async def get_gradient_descent_info() -> Dict[str, Any]:
    """Get Gradient Descent Variants algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Gradient Descent variants.
    Includes detailed theory about SGD, Momentum, RMSprop, Adam, and Adagrad optimizers.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("gradient-descent")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Gradient Descent metadata not found"
        )

    dataset_info = get_gradient_descent_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Autoencoder Variants metadata
autoencoder_variants_metadata = AlgorithmMetadata(
    id="autoencoder-variants",
    name="Autoencoder Variants",
    slug="autoencoder-variants",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Compare different autoencoder architectures for unsupervised feature learning",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["deep-learning", "autoencoder", "unsupervised", "dimensionality-reduction", "denoising"],
    use_cases=[
        "Image denoising",
        "Anomaly detection",
        "Dimensionality reduction",
        "Feature learning",
        "Data compression",
        "Pretraining"
    ],
    complexity=AlgorithmComplexity(
        time="O(epochs*batch_size)",
        space="O(encoder_params + decoder_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="variant",
            label="Autoencoder Variant",
            type="select",
            default="vanilla",
            options=[
                {"label": "Vanilla", "value": "vanilla"},
                {"label": "Denoising", "value": "denoising"},
                {"label": "Sparse", "value": "sparse"},
                {"label": "Contractive", "value": "contractive"}
            ],
            description="Type of autoencoder to train"
        ),
        AlgorithmParameter(
            name="latent_dim",
            label="Latent Dimension",
            type="range",
            default=32,
            min=2,
            max=128,
            step=2,
            description="Size of the latent space bottleneck"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=10,
            min=5,
            max=50,
            step=5,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for Adam optimizer"
        ),
        AlgorithmParameter(
            name="noise_factor",
            label="Noise Factor (Denoising)",
            type="range",
            default=0.3,
            min=0.0,
            max=0.5,
            step=0.05,
            description="Amount of noise to add for denoising autoencoder"
        ),
        AlgorithmParameter(
            name="sparsity_weight",
            label="Sparsity Weight",
            type="range",
            default=0.001,
            min=0.0,
            max=0.1,
            step=0.001,
            description="Weight for sparsity penalty (Sparse/Contractive)"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=128,
            options=[
                {"label": "32", "value": 32},
                {"label": "64", "value": 64},
                {"label": "128", "value": 128},
                {"label": "256", "value": 256}
            ],
            description="Batch size for training"
        )
    ],
    dataset_name="mnist_digits",
    visualization_type="reconstructions,latent_space,loss_curves,filters,error_heatmap",
    theory=(
        "Autoencoders are neural networks trained to reconstruct their inputs through a "
        "bottleneck layer. Different variants introduce specific properties:\n\n"
        "1. Vanilla Autoencoder: Standard encoder-decoder with reconstruction loss\n"
        "   - Architecture: Input → Encoder (784→256→128→latent_dim) → "
        "Decoder (latent_dim→128→256→784) → Output\n"
        "   - Loss: MSE(input, output)\n\n"
        "2. Denoising Autoencoder (DAE): Learns to remove noise\n"
        "   - Trained on corrupted inputs to reconstruct clean targets\n"
        "   - Adds Gaussian noise: x_noisy = x + N(0, σ²)\n"
        "   - Forces robust feature learning\n"
        "   - Loss: MSE(x_clean, decoder(encoder(x_noisy)))\n\n"
        "3. Sparse Autoencoder (SAE): Encourages sparse activations\n"
        "   - Adds L1 penalty on latent activations\n"
        "   - Loss: MSE(x, x_recon) + λ * ||z||₁\n"
        "   - Learns more interpretable features\n"
        "   - Forces selective feature activation\n\n"
        "4. Contractive Autoencoder (CAE): Penalizes sensitivity to input\n"
        "   - Adds Frobenius norm of Jacobian to loss\n"
        "   - Loss: MSE(x, x_recon) + λ * ||∂h/∂x||²_F\n"
        "   - Makes learned features robust to small variations\n"
        "   - Encourages learning manifold structure\n\n"
        "All variants use the same encoder-decoder architecture but differ in their "
        "training objectives, leading to different learned representations and applications."
    ),
    pros=[
        "Learns meaningful features without labels (unsupervised)",
        "Denoising variant robust to input corruption",
        "Sparse variant learns interpretable features",
        "Contractive variant robust to input perturbations",
        "Useful for dimensionality reduction and visualization",
        "Can be used for anomaly detection (high reconstruction error)"
    ],
    cons=[
        "Requires careful hyperparameter tuning per variant",
        "Denoising needs appropriate noise level selection",
        "Sparse/Contractive add computational overhead",
        "May overfit on small datasets",
        "Latent space may not be smooth (unlike VAE)",
        "Cannot directly sample new data (unlike GAN/VAE)"
    ],
    related_algorithms=["autoencoder", "vae", "pca", "gan"]
)

AlgorithmRegistry.register(autoencoder_variants_metadata)


@router.post("/autoencoder-variants/train", response_model=AutoencoderVariantsResponse)
async def train_autoencoder_variants(request: AutoencoderVariantsRequest) -> AutoencoderVariantsResponse:
    """Train an Autoencoder variant (Vanilla, Denoising, Sparse, or Contractive).

    This endpoint trains one of four autoencoder variants on MNIST digits:

    1. Vanilla: Standard encoder-decoder
    2. Denoising: Trained to remove noise from corrupted inputs
    3. Sparse: L1 regularization on latent activations
    4. Contractive: Penalty on Jacobian for input robustness

    Returns comprehensive visualizations including reconstructions, latent space,
    learned filters, and reconstruction error heatmaps.

    Args:
        request: Autoencoder variant parameters including variant type, latent_dim,
                epochs, learning_rate, noise_factor, sparsity_weight, and batch_size

    Returns:
        AutoencoderVariantsResponse with:
            - Original, noisy (if denoising), and reconstructed images
            - 2D latent space visualization (PCA/t-SNE if latent_dim > 2)
            - Training loss curves
            - Learned encoder filters (first layer)
            - Per-sample reconstruction error
            - Model info and execution time

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training {request.variant} autoencoder with parameters: {request.model_dump()}")

        # Load MNIST data
        data = load_autoencoder_variants_data(
            n_samples=1000,
            test_size=0.2,
            random_state=request.random_state
        )

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        # Initialize model
        model = AutoencoderVariantsModel(
            variant=request.variant,
            latent_dim=request.latent_dim,
            learning_rate=request.learning_rate,
            noise_factor=request.noise_factor,
            sparsity_weight=request.sparsity_weight,
            random_state=request.random_state
        )

        # Train model
        training_results = model.train(
            X_train=X_train,
            X_test=X_test,
            epochs=request.epochs,
            batch_size=request.batch_size
        )

        # Evaluate
        metrics = model.evaluate(X_test)
        metrics['final_loss'] = training_results['final_val_loss']

        # Get reconstructions
        n_samples = min(10, len(X_test))
        X_sample = X_test[:n_samples]
        y_sample = y_test[:n_samples]

        # For denoising, also show noisy inputs
        if request.variant == 'denoising':
            from algorithms.deep_learning.autoencoder_variants.data import add_noise as add_noise_fn
            X_noisy = add_noise_fn(X_sample, request.noise_factor)
            reconstructed = model.reconstruct(X_sample, add_noise=True)
            noisy_images = X_noisy.tolist()
        else:
            reconstructed = model.reconstruct(X_sample)
            noisy_images = None

        # Get latent space visualization
        latent_vectors = model.encode(X_test)
        coords_2d, projection_method = compute_autoencoder_variants_latent_viz(
            latent_vectors,
            y_test,
            request.latent_dim,
            random_state=request.random_state
        )

        # Compute reconstruction errors
        from algorithms.deep_learning.autoencoder_variants.data import compute_reconstruction_errors
        reconstruction_errors = compute_reconstruction_errors(X_test, model.reconstruct(X_test))

        # Get learned filters
        learned_filters = model.get_learned_filters(n_filters=16)

        # Prepare response data
        loss_history = [
            {
                'epoch': i,
                'train_loss': float(model.training_history['train_loss'][i]),
                'val_loss': float(model.training_history['val_loss'][i])
            }
            for i in range(len(model.training_history['train_loss']))
        ]

        latent_space = coords_2d.tolist()
        latent_labels = y_test.tolist()

        visualization_data = {
            'n_samples': len(X_test),
            'image_shape': [8, 8],
            'latent_dim': request.latent_dim,
            'projection_method': projection_method,
            'variant': request.variant
        }

        # Get model info
        model_info = model.get_model_info()
        model_info['total_epochs'] = request.epochs

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"{request.variant.capitalize()} autoencoder training completed in {execution_time_ms:.2f}ms. "
            f"Final loss: {metrics['final_loss']:.4f}, "
            f"Reconstruction MSE: {metrics['reconstruction_mse']:.4f}"
        )

        return AutoencoderVariantsResponse(
            success=True,
            variant=request.variant,
            metrics=metrics,
            loss_history=loss_history,
            original_images=X_sample.tolist(),
            noisy_images=noisy_images,
            reconstructed_images=reconstructed.tolist(),
            latent_space=latent_space,
            latent_labels=latent_labels,
            reconstruction_errors=reconstruction_errors.tolist(),
            learned_filters=learned_filters.tolist(),
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            model_info=model_info,
            parameters_used={
                'variant': request.variant,
                'latent_dim': request.latent_dim,
                'epochs': request.epochs,
                'learning_rate': request.learning_rate,
                'noise_factor': request.noise_factor,
                'sparsity_weight': request.sparsity_weight,
                'batch_size': request.batch_size
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/autoencoder-variants/info")
async def get_autoencoder_variants_info() -> Dict[str, Any]:
    """Get Autoencoder Variants algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Autoencoder Variants.
    Includes detailed theory about the four variants and their applications.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("autoencoder-variants")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Autoencoder Variants metadata not found"
        )

    dataset_info = get_autoencoder_variants_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }


# Register Transfer Learning metadata
transfer_learning_metadata = AlgorithmMetadata(
    id="transfer-learning",
    name="Transfer Learning",
    slug="transfer-learning",
    category=AlgorithmCategory.DEEP_LEARNING,
    description="Fine-tune pre-trained models for new tasks with limited data",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["deep-learning", "transfer-learning", "fine-tuning", "pre-trained", "feature-extraction"],
    use_cases=[
        "Small dataset learning",
        "Domain adaptation",
        "Quick prototyping",
        "Medical imaging",
        "Specialized classification",
        "Few-shot learning"
    ],
    complexity=AlgorithmComplexity(
        time="O(epochs*samples)",
        space="O(pretrained_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="base_model",
            label="Pre-trained Model",
            type="select",
            default="resnet18",
            options=[
                {"label": "ResNet-18", "value": "resnet18"},
                {"label": "ResNet-50", "value": "resnet50"},
                {"label": "MobileNet V2", "value": "mobilenet_v2"},
                {"label": "EfficientNet-B0", "value": "efficientnet_b0"}
            ],
            description="Pre-trained model architecture to use as base"
        ),
        AlgorithmParameter(
            name="strategy",
            label="Transfer Strategy",
            type="select",
            default="fine_tune",
            options=[
                {"label": "Feature Extraction", "value": "feature_extraction"},
                {"label": "Fine-tuning", "value": "fine_tune"},
                {"label": "Full Training", "value": "full_train"}
            ],
            description="Transfer learning strategy"
        ),
        AlgorithmParameter(
            name="freeze_layers",
            label="Freeze Layers",
            type="select",
            default="auto",
            options=[
                {"label": "Auto (Recommended)", "value": "auto"},
                {"label": "None", "value": "none"},
                {"label": "Early Layers", "value": "early"},
                {"label": "Most Layers", "value": "most"},
                {"label": "All But Last", "value": "all_but_last"}
            ],
            description="Which layers to freeze during training"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for optimizer"
        ),
        AlgorithmParameter(
            name="epochs",
            label="Training Epochs",
            type="range",
            default=10,
            min=5,
            max=50,
            step=5,
            description="Number of training epochs"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="select",
            default=16,
            options=[
                {"label": "4", "value": 4},
                {"label": "8", "value": 8},
                {"label": "16", "value": 16},
                {"label": "32", "value": 32},
                {"label": "64", "value": 64}
            ],
            description="Training batch size"
        ),
        AlgorithmParameter(
            name="dataset",
            label="Dataset",
            type="select",
            default="flowers",
            options=[
                {"label": "Flowers (5 classes)", "value": "flowers"},
                {"label": "Animals (5 classes)", "value": "animals"},
                {"label": "Food (5 classes)", "value": "food"}
            ],
            description="Custom dataset for fine-tuning"
        )
    ],
    dataset_name="custom_small_datasets",
    visualization_type="training_curves,confusion_matrix,layer_freezing,feature_maps,convergence_comparison",
    theory=(
        "Transfer Learning leverages knowledge from pre-trained models to solve new tasks with "
        "limited data. Instead of training from scratch, we use models pre-trained on large datasets "
        "(like ImageNet) and adapt them to our specific task. This is especially powerful when "
        "data is scarce.\n\n"
        "Three Main Strategies:\n\n"
        "1. Feature Extraction:\n"
        "   - Freeze all pre-trained layers\n"
        "   - Train only the final classification layer\n"
        "   - Fastest training, works well when datasets are very small\n"
        "   - Treats pre-trained model as fixed feature extractor\n\n"
        "2. Fine-tuning:\n"
        "   - Freeze early layers (general features like edges, textures)\n"
        "   - Train later layers + classifier (task-specific features)\n"
        "   - Balanced approach between speed and adaptation\n"
        "   - Most commonly used strategy\n\n"
        "3. Full Training:\n"
        "   - Train all layers (baseline comparison)\n"
        "   - Requires more data and computation\n"
        "   - Allows complete adaptation but risks overfitting\n\n"
        "Why Transfer Learning Works:\n"
        "- Early layers learn universal features (edges, colors, textures)\n"
        "- Later layers learn task-specific features\n"
        "- Pre-trained weights provide good initialization\n"
        "- Reduces training time and data requirements\n"
        "- Often achieves better performance than training from scratch\n\n"
        "Layer Freezing:\n"
        "Frozen layers have requires_grad=False, meaning their weights don't update during "
        "backpropagation. This preserves learned features and reduces computation. The choice "
        "of which layers to freeze depends on:\n"
        "- Dataset size (smaller = freeze more)\n"
        "- Dataset similarity to pre-training data (similar = freeze more)\n"
        "- Computational resources (freeze more = faster training)\n\n"
        "Common Practice:\n"
        "- Very small dataset (<1000 samples): Feature extraction\n"
        "- Small dataset (1000-10000): Fine-tune last few layers\n"
        "- Medium dataset (10000-100000): Fine-tune half the network\n"
        "- Large dataset (>100000): Fine-tune entire network or train from scratch\n\n"
        "Benefits vs Training from Scratch:\n"
        "- Faster convergence (fewer epochs needed)\n"
        "- Better performance with limited data\n"
        "- Reduced computational requirements\n"
        "- Lower risk of overfitting\n"
        "- Access to features learned from millions of images"
    ),
    pros=[
        "Achieves high accuracy with small datasets (few hundred samples)",
        "Faster training than starting from scratch",
        "Leverages features learned from millions of images",
        "Reduces overfitting risk with limited data",
        "Lower computational cost than full training",
        "Proven to work across diverse domains",
        "Can adapt ImageNet models to specialized tasks"
    ],
    cons=[
        "Pre-trained models are large (50-500MB)",
        "May not work well if new task is very different from pre-training",
        "Requires understanding of layer freezing strategies",
        "Fine-tuning hyperparameters can be tricky",
        "Limited to architectures with available pre-trained weights",
        "May inherit biases from pre-training dataset"
    ],
    related_algorithms=["resnet", "vgg", "cnn", "feature-extraction", "domain-adaptation"]
)

AlgorithmRegistry.register(transfer_learning_metadata)


@router.post("/transfer-learning/train", response_model=TransferLearningResponse)
async def train_transfer_learning(request: TransferLearningRequest) -> TransferLearningResponse:
    """Train a transfer learning model on a small custom dataset.

    This endpoint demonstrates transfer learning by fine-tuning pre-trained models
    (ResNet, MobileNet, EfficientNet) on small custom datasets. It compares different
    strategies (feature extraction, fine-tuning, full training) and shows how
    pre-trained models can achieve high accuracy with limited data.

    Args:
        request: Transfer learning parameters including base_model, strategy,
                freeze_layers, learning_rate, epochs, batch_size, and dataset

    Returns:
        TransferLearningResponse with training results, layer freezing info,
        confusion matrix, sample predictions, and strategy comparison

    Raises:
        HTTPException: If training fails due to invalid parameters or errors
    """
    try:
        start_time = time.time()
        logger.info(f"Training transfer learning model with parameters: {request.model_dump()}")

        # Run transfer learning
        response = run_transfer_learning(request)

        logger.info(
            f"Transfer learning completed in {response.execution_time_ms:.2f}ms. "
            f"Test accuracy: {response.metrics['test_accuracy']:.4f}, "
            f"Trainable params: {response.model_info['trainable_parameters']}, "
            f"Frozen params: {response.model_info['frozen_parameters']}"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/transfer-learning/info")
async def get_transfer_learning_info() -> Dict[str, Any]:
    """Get Transfer Learning algorithm information and metadata.

    Returns metadata, parameters, and dataset information for Transfer Learning.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("transfer-learning")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Transfer Learning metadata not found"
        )

    dataset_info = get_transfer_learning_dataset_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": dataset_info
    }
