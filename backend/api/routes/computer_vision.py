from fastapi import APIRouter, HTTPException
import time
from typing import Dict, Any

from algorithms.computer_vision.yolo import YOLOModel, YOLORequest, YOLOResponse
from algorithms.computer_vision.semantic_segmentation import (
    SegmentationModel,
    SegmentationRequest,
    SegmentationResponse
)
from algorithms.computer_vision.edge_detection import (
    EdgeDetectionModel,
    EdgeDetectionRequest,
    EdgeDetectionResponse
)
from algorithms.computer_vision.sift import (
    SIFTModel,
    SIFTRequest,
    SIFTResponse
)
from algorithms.computer_vision.image_classification import (
    ImageClassificationModel,
    ImageClassificationRequest,
    ImageClassificationResponse
)
from algorithms.computer_vision.face_detection import (
    FaceDetectionModel,
    FaceDetectionRequest,
    FaceDetectionResponse
)
from algorithms.computer_vision.optical_flow import (
    OpticalFlowModel,
    OpticalFlowRequest,
    OpticalFlowResponse
)
from algorithms.computer_vision.instance_segmentation import (
    InstanceSegmentationModel,
    InstanceSegmentationRequest,
    InstanceSegmentationResponse
)
from algorithms.computer_vision.style_transfer import (
    StyleTransferModel,
    StyleTransferRequest,
    StyleTransferResponse
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
router = APIRouter(prefix="/computer-vision", tags=["Computer Vision"])


# Register YOLO metadata
yolo_metadata = AlgorithmMetadata(
    id="yolo-detection",
    name="YOLO Object Detection",
    slug="yolo-detection",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Real-time object detection with bounding boxes and class labels",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["computer-vision", "object-detection", "yolo", "real-time", "bounding-boxes"],
    use_cases=[
        "Autonomous vehicles",
        "Surveillance systems",
        "Retail analytics",
        "Sports analysis",
        "Wildlife monitoring",
        "Traffic monitoring"
    ],
    complexity=AlgorithmComplexity(
        time="O(image_size)",
        space="O(anchors*classes)"
    ),
    parameters=[
        AlgorithmParameter(
            name="model_version",
            label="Model Version",
            type="select",
            default="yolov8n",
            options=[
                {"label": "YOLOv8 Nano (Fastest)", "value": "yolov8n"},
                {"label": "YOLOv8 Small (Balanced)", "value": "yolov8s"},
                {"label": "YOLOv8 Medium (Most Accurate)", "value": "yolov8m"},
                {"label": "YOLOv5 Small", "value": "yolov5s"}
            ],
            description="YOLO version - trades off speed vs accuracy"
        ),
        AlgorithmParameter(
            name="confidence_threshold",
            label="Confidence Threshold",
            type="range",
            default=0.25,
            min=0.1,
            max=0.9,
            step=0.05,
            description="Minimum confidence score for detections (higher = fewer but more confident detections)"
        ),
        AlgorithmParameter(
            name="iou_threshold",
            label="IoU Threshold",
            type="range",
            default=0.45,
            min=0.1,
            max=0.9,
            step=0.05,
            description="IoU threshold for Non-Maximum Suppression (higher = more overlapping boxes)"
        ),
        AlgorithmParameter(
            name="max_detections",
            label="Max Detections",
            type="number",
            default=100,
            min=10,
            max=300,
            step=10,
            description="Maximum number of objects to detect in the image"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="number",
            default=0,
            min=0,
            max=14,
            step=1,
            description="Select sample image (0-14: diverse indoor/outdoor scenes)"
        ),
        AlgorithmParameter(
            name="class_filter",
            label="Class Filter",
            type="select",
            default="all",
            options=[
                {"label": "All Classes", "value": "all"},
                {"label": "Person", "value": "person"},
                {"label": "Vehicle", "value": "vehicle"},
                {"label": "Animal", "value": "animal"}
            ],
            description="Filter detections by object category"
        )
    ],
    dataset_name="COCO",
    visualization_type="image_with_bboxes",
    theory=(
        "YOLO (You Only Look Once) is a real-time object detection algorithm that "
        "treats object detection as a regression problem. Unlike traditional methods "
        "that apply classifiers to different regions, YOLO divides the image into a "
        "grid and predicts bounding boxes and class probabilities directly in a single "
        "forward pass. YOLOv8, the latest version, uses a modified CSPDarknet backbone, "
        "PANet neck for feature fusion, and anchor-free detection heads. The model "
        "predicts bounding box coordinates (x, y, width, height), objectness score, "
        "and class probabilities for each grid cell. Non-Maximum Suppression (NMS) "
        "is applied to remove duplicate detections. The algorithm achieves real-time "
        "performance while maintaining high accuracy, making it ideal for applications "
        "requiring fast inference like autonomous vehicles and surveillance systems."
    ),
    pros=[
        "Extremely fast - real-time detection (30+ FPS)",
        "Single unified network - end-to-end training",
        "Learns generalizable object representations",
        "Good at detecting small objects in groups",
        "Sees entire image during training (better context)"
    ],
    cons=[
        "May struggle with very small objects",
        "Less accurate than two-stage detectors (e.g., Faster R-CNN)",
        "Difficulty with objects in unusual aspect ratios",
        "Sensitive to objects in tight groups",
        "Requires significant computational resources for training"
    ],
    related_algorithms=["faster-rcnn", "ssd", "mask-rcnn", "retinanet"]
)

AlgorithmRegistry.register(yolo_metadata)


# Register Edge Detection metadata
edge_detection_metadata = AlgorithmMetadata(
    id="edge-detection",
    name="Edge Detection (Canny)",
    slug="edge-detection",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Detect edges in images using Canny edge detector",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["computer-vision", "image-processing", "edge-detection", "feature-extraction"],
    use_cases=[
        "Object detection preprocessing",
        "Image segmentation",
        "Feature extraction",
        "Medical imaging",
        "Document scanning"
    ],
    complexity=AlgorithmComplexity(
        time="O(width*height)",
        space="O(width*height)"
    ),
    parameters=[
        AlgorithmParameter(
            name="threshold1",
            label="Lower Threshold",
            type="range",
            default=50,
            min=0,
            max=255,
            step=5,
            description="Lower threshold for hysteresis (edges with gradient below this are discarded)"
        ),
        AlgorithmParameter(
            name="threshold2",
            label="Upper Threshold",
            type="range",
            default=150,
            min=0,
            max=255,
            step=5,
            description="Upper threshold for hysteresis (edges with gradient above this are kept)"
        ),
        AlgorithmParameter(
            name="aperture_size",
            label="Aperture Size",
            type="select",
            default=3,
            options=[
                {"label": "3x3 (Fast)", "value": 3},
                {"label": "5x5 (Balanced)", "value": 5},
                {"label": "7x7 (Smooth)", "value": 7}
            ],
            description="Sobel kernel size for gradient calculation"
        ),
        AlgorithmParameter(
            name="l2gradient",
            label="L2 Gradient",
            type="boolean",
            default=False,
            description="Use L2 norm for gradient magnitude (more accurate but slower)"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="number",
            default=0,
            min=0,
            max=7,
            step=1,
            description="Select sample image (0-7: bikes, valve, bus, ducks, receipt, cat, sports, architecture)"
        )
    ],
    dataset_name="Edge Detection Samples",
    visualization_type="image_comparison",
    theory=(
        "The Canny edge detector is a multi-stage algorithm invented by John F. Canny in 1986. "
        "It detects edges by finding local maxima of the image gradient. The algorithm follows "
        "five main stages: (1) Noise Reduction using Gaussian blur to reduce image noise, "
        "(2) Gradient Calculation using Sobel operators to compute intensity gradients, "
        "(3) Non-maximum Suppression to thin edges by keeping only local maxima, "
        "(4) Double Threshold to classify edges as strong, weak, or non-edges, and "
        "(5) Edge Tracking by Hysteresis to connect weak edges to strong edges. "
        "The two thresholds control sensitivity: threshold2 (upper) marks strong edges that are "
        "definitely kept, while threshold1 (lower) marks weak edges that are only kept if they "
        "connect to strong edges. The typical ratio between thresholds is 2:1 or 3:1."
    ),
    pros=[
        "Excellent edge localization with single-pixel precision",
        "Good noise resistance due to Gaussian smoothing",
        "Detects both strong and weak edges via hysteresis",
        "Fast computation - suitable for real-time applications",
        "Widely implemented and well-tested algorithm"
    ],
    cons=[
        "Sensitive to threshold parameter selection",
        "Cannot detect edges with very low contrast",
        "Produces binary output (loses gradient magnitude info)",
        "May create broken edges in noisy images",
        "Requires parameter tuning for different image types"
    ],
    related_algorithms=["sobel-edge", "prewitt-edge", "laplacian-edge", "harris-corner"]
)

AlgorithmRegistry.register(edge_detection_metadata)


@router.get("/")
async def computer_vision_root():
    return {"message": "Computer Vision algorithms endpoint"}


@router.get("/algorithms")
async def list_computer_vision_algorithms():
    """Get all registered Computer Vision algorithms.

    Returns:
        List of algorithm metadata for all registered CV algorithms
    """
    algorithms = AlgorithmRegistry.get_by_category(AlgorithmCategory.COMPUTER_VISION)
    return [algo.model_dump() for algo in algorithms]


@router.post("/yolo/detect", response_model=YOLOResponse)
async def detect_objects_yolo(request: YOLORequest):
    """Run YOLO object detection on a sample image.

    This endpoint performs real-time object detection using YOLOv8,
    identifying objects, their locations, and confidence scores.

    Args:
        request: YOLORequest containing detection parameters

    Returns:
        YOLOResponse with detection results, statistics, and visualization data

    Raises:
        HTTPException: If detection fails or parameters are invalid
    """
    try:
        logger.info(f"Running YOLO detection with parameters: {request.model_dump()}")

        # Initialize model
        model = YOLOModel(model_version=request.model_version)

        # Process request
        response = model.process_request(request)

        logger.info(
            f"YOLO detection completed in {response.execution_time_ms:.2f}ms. "
            f"Detected {response.statistics.total_detections} objects."
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@router.get("/yolo/info")
async def get_yolo_info() -> Dict[str, Any]:
    """Get YOLO algorithm information and metadata.

    Returns detailed information about the YOLO algorithm including
    parameters, complexity, use cases, and available sample images.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("yolo-detection")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="YOLO algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = YOLOModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "model_variants": algo_info['model_variants']
    }


# Register Semantic Segmentation metadata
segmentation_metadata = AlgorithmMetadata(
    id="semantic-segmentation",
    name="Semantic Segmentation (U-Net)",
    slug="semantic-segmentation",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Pixel-wise image classification for scene understanding",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["computer-vision", "segmentation", "deep-learning", "scene-understanding"],
    use_cases=[
        "Autonomous driving (road scene parsing)",
        "Medical imaging (organ segmentation)",
        "Satellite imagery analysis",
        "Augmented reality",
        "Video background removal"
    ],
    complexity=AlgorithmComplexity(
        time="O(H*W*C)",
        space="O(H*W)"
    ),
    parameters=[
        AlgorithmParameter(
            name="num_classes",
            label="Number of Classes",
            type="number",
            default=21,
            min=2,
            max=150,
            step=1,
            description="Number of segmentation classes (PASCAL VOC uses 21 classes)"
        ),
        AlgorithmParameter(
            name="confidence_threshold",
            label="Confidence Threshold",
            type="range",
            default=0.5,
            min=0.1,
            max=0.95,
            step=0.05,
            description="Minimum confidence for predictions (higher = more conservative)"
        ),
        AlgorithmParameter(
            name="model_backbone",
            label="Model Backbone",
            type="select",
            default="resnet50",
            options=[
                {"label": "ResNet50 (More Accurate)", "value": "resnet50"},
                {"label": "MobileNet (Faster)", "value": "mobilenet"}
            ],
            description="Backbone architecture - trades off speed vs accuracy"
        ),
        AlgorithmParameter(
            name="image_size",
            label="Input Image Size",
            type="select",
            default=512,
            options=[
                {"label": "256x256 (Fastest)", "value": 256},
                {"label": "512x512 (Balanced)", "value": 512},
                {"label": "1024x1024 (Best Quality)", "value": 1024}
            ],
            description="Input resolution for the model"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="number",
            default=0,
            min=0,
            max=2,
            step=1,
            description="Select sample image (0=dog portrait, 1=dog on grass, 2=street scene)"
        )
    ],
    dataset_name="PASCAL VOC 2012",
    visualization_type="image_with_segmentation",
    theory=(
        "Semantic segmentation assigns a class label to each pixel in an image, enabling "
        "detailed scene understanding. DeepLabV3 uses Atrous Spatial Pyramid Pooling (ASPP) "
        "to capture multi-scale context by applying dilated convolutions at multiple rates. "
        "The model consists of: (1) A backbone network (ResNet or MobileNet) for feature "
        "extraction, (2) ASPP module to capture multi-scale information, (3) A decoder that "
        "gradually recovers spatial information, (4) Final classification layer producing "
        "per-pixel class predictions. The atrous (dilated) convolutions allow the network to "
        "increase receptive field without reducing spatial resolution or increasing parameters. "
        "The output is a segmentation mask where each pixel is classified into one of the "
        "predefined classes (e.g., person, car, road, sky)."
    ),
    pros=[
        "Dense pixel-level predictions for detailed scene understanding",
        "Multi-scale context capture via ASPP",
        "Preserves spatial information better than detection",
        "Pre-trained models available for transfer learning",
        "Applicable to diverse domains (medical, autonomous driving)"
    ],
    cons=[
        "Computationally expensive (processes every pixel)",
        "Requires pixel-level annotations for training",
        "Slower inference than object detection",
        "May struggle with fine boundaries",
        "Limited to predefined classes"
    ],
    related_algorithms=["unet", "mask-rcnn", "fcn", "pspnet"]
)

AlgorithmRegistry.register(segmentation_metadata)


@router.post("/semantic-segmentation/segment", response_model=SegmentationResponse)
async def segment_image(request: SegmentationRequest):
    """Run semantic segmentation on a sample image.

    This endpoint performs pixel-wise classification using DeepLabV3,
    assigning each pixel to a semantic class.

    Args:
        request: SegmentationRequest containing segmentation parameters

    Returns:
        SegmentationResponse with segmentation results, statistics, and visualization data

    Raises:
        HTTPException: If segmentation fails or parameters are invalid
    """
    try:
        logger.info(f"Running semantic segmentation with parameters: {request.model_dump()}")

        # Initialize model
        model = SegmentationModel(
            model_backbone=request.model_backbone,
            num_classes=request.num_classes
        )

        # Process request
        response = model.process_request(request)

        logger.info(
            f"Semantic segmentation completed in {response.execution_time_ms:.2f}ms. "
            f"Found {response.statistics.total_classes} classes."
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )


@router.get("/semantic-segmentation/info")
async def get_segmentation_info() -> Dict[str, Any]:
    """Get semantic segmentation algorithm information and metadata.

    Returns detailed information about the semantic segmentation algorithm including
    parameters, complexity, use cases, and available sample images.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("semantic-segmentation")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Semantic segmentation algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = SegmentationModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "model_variants": algo_info['model_variants']
    }


@router.post("/edge-detection/detect", response_model=EdgeDetectionResponse)
async def detect_edges_canny(request: EdgeDetectionRequest):
    """Detect edges in an image using Canny edge detection algorithm.

    This endpoint performs edge detection using the Canny algorithm,
    which detects edges through a multi-stage process including noise
    reduction, gradient calculation, non-maximum suppression, and
    hysteresis thresholding.

    Args:
        request: EdgeDetectionRequest containing detection parameters

    Returns:
        EdgeDetectionResponse with edge detection results, statistics, and visualization data

    Raises:
        HTTPException: If edge detection fails or parameters are invalid
    """
    try:
        logger.info(f"Running Canny edge detection with parameters: {request.model_dump()}")

        # Initialize model
        model = EdgeDetectionModel()

        # Process request
        response = model.process_request(request)

        logger.info(
            f"Edge detection completed in {response.execution_time_ms:.2f}ms. "
            f"Detected {response.statistics.edge_pixel_count} edge pixels "
            f"({response.statistics.edge_density:.2f}% density)."
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Edge detection failed: {str(e)}"
        )


@router.get("/edge-detection/info")
async def get_edge_detection_info() -> Dict[str, Any]:
    """Get edge detection algorithm information and metadata.

    Returns detailed information about the Canny edge detection algorithm including
    parameters, complexity, use cases, and available sample images.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("edge-detection")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Edge detection algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = EdgeDetectionModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "algorithm_details": algo_info['algorithm_details']
    }


# Register Face Detection metadata
face_detection_metadata = AlgorithmMetadata(
    id="face-detection",
    name="Face Detection (Haar Cascades)",
    slug="face-detection",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Detect faces in images using Haar Cascade classifiers",
    difficulty=DifficultyLevel.BEGINNER,
    tags=["computer-vision", "face-detection", "object-detection", "haar-cascade"],
    use_cases=[
        "Photo organization and tagging",
        "Security and surveillance systems",
        "Attendance tracking",
        "Social media auto-tagging",
        "Demographics analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(image_size*scales)",
        space="O(cascade_size)"
    ),
    parameters=[
        AlgorithmParameter(
            name="scale_factor",
            label="Scale Factor",
            type="range",
            default=1.1,
            min=1.05,
            max=1.3,
            step=0.05,
            description="Scale reduction factor between successive scans (lower = more accurate but slower)"
        ),
        AlgorithmParameter(
            name="min_neighbors",
            label="Min Neighbors",
            type="range",
            default=5,
            min=1,
            max=10,
            step=1,
            description="Minimum neighbors required for detection (higher = fewer false positives)"
        ),
        AlgorithmParameter(
            name="min_size",
            label="Min Face Size (px)",
            type="number",
            default=30,
            min=20,
            max=100,
            step=5,
            description="Minimum face size in pixels (faces smaller than this are ignored)"
        ),
        AlgorithmParameter(
            name="max_size",
            label="Max Face Size (px)",
            type="number",
            default=None,
            min=50,
            max=500,
            step=10,
            description="Maximum face size in pixels (None for no limit)"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="number",
            default=0,
            min=0,
            max=7,
            step=1,
            description="Select sample image (0-7: portraits and group photos)"
        )
    ],
    dataset_name="Face Detection Samples",
    visualization_type="image_with_bboxes",
    theory=(
        "Haar Cascade face detection is a machine learning based approach developed by "
        "Paul Viola and Michael Jones in 2001. The algorithm uses Haar-like features to "
        "detect objects (in this case, faces) in images. The method works by: (1) Extracting "
        "Haar-like features from image regions - these are simple rectangular features that "
        "capture intensity differences, (2) Computing integral images for fast feature evaluation, "
        "(3) Using AdaBoost to select the most discriminative features and train weak classifiers, "
        "(4) Cascading classifiers so that simple tests quickly reject non-face regions while "
        "complex tests are only applied to promising regions, (5) Sliding a detection window "
        "across the image at multiple scales to find faces of different sizes. The cascade "
        "structure allows for very fast detection since most image regions are quickly rejected "
        "as non-faces in early stages. Pre-trained cascades are available for frontal faces, "
        "profile faces, eyes, and other objects."
    ),
    pros=[
        "Very fast detection - suitable for real-time applications",
        "Works well for frontal faces with good lighting",
        "Pre-trained models readily available in OpenCV",
        "Low computational requirements (no GPU needed)",
        "Simple to implement and use"
    ],
    cons=[
        "Struggles with profile views or tilted faces",
        "Performance degrades in poor lighting conditions",
        "Can produce false positives on face-like patterns",
        "Not robust to occlusions (glasses, masks, etc.)",
        "Less accurate than modern deep learning methods"
    ],
    related_algorithms=["dlib-face", "mtcnn", "yolo-face", "retinaface"]
)

AlgorithmRegistry.register(face_detection_metadata)


@router.post("/face-detection/detect", response_model=FaceDetectionResponse)
async def detect_faces_haar(request: FaceDetectionRequest):
    """Detect faces in an image using Haar Cascade classifiers.

    This endpoint performs face detection using the Viola-Jones algorithm
    with Haar-like features, identifying faces and their locations.

    Args:
        request: FaceDetectionRequest containing detection parameters

    Returns:
        FaceDetectionResponse with face detection results, statistics, and visualization data

    Raises:
        HTTPException: If face detection fails or parameters are invalid
    """
    try:
        logger.info(f"Running Haar Cascade face detection with parameters: {request.model_dump()}")

        # Initialize model
        model = FaceDetectionModel()

        # Process request
        response = model.process_request(request)

        logger.info(
            f"Face detection completed in {response.execution_time_ms:.2f}ms. "
            f"Detected {response.statistics.face_count} faces."
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Face detection failed: {str(e)}"
        )


@router.get("/face-detection/info")
async def get_face_detection_info() -> Dict[str, Any]:
    """Get face detection algorithm information and metadata.

    Returns detailed information about the Haar Cascade face detection algorithm including
    parameters, complexity, use cases, and available sample images.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("face-detection")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Face detection algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = FaceDetectionModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "algorithm_details": algo_info['algorithm_details']
    }


# Register Image Classification metadata
image_classification_metadata = AlgorithmMetadata(
    id="image-classification",
    name="Image Classification (CNN)",
    slug="image-classification",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Classify images using Convolutional Neural Networks",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["computer-vision", "deep-learning", "classification", "cnn"],
    use_cases=[
        "Photo organization",
        "Medical diagnosis",
        "Quality control",
        "Wildlife monitoring",
        "Content moderation"
    ],
    complexity=AlgorithmComplexity(
        time="O(image_size*filters*layers)",
        space="O(model_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="model_name",
            label="Model",
            type="select",
            default="resnet18",
            options=[
                {"label": "ResNet-18 (Fast, 11M params)", "value": "resnet18"},
                {"label": "ResNet-50 (Accurate, 25M params)", "value": "resnet50"},
                {"label": "MobileNetV2 (Fastest, 3.5M params)", "value": "mobilenet_v2"}
            ],
            description="Pre-trained model architecture - trades off speed vs accuracy"
        ),
        AlgorithmParameter(
            name="top_k",
            label="Top K Predictions",
            type="number",
            default=5,
            min=1,
            max=10,
            step=1,
            description="Number of top predictions to return"
        ),
        AlgorithmParameter(
            name="confidence_threshold",
            label="Confidence Threshold",
            type="range",
            default=0.1,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Minimum confidence for predictions (higher = fewer but more confident predictions)"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="number",
            default=0,
            min=0,
            max=14,
            step=1,
            description="Select sample image (0-14: cat, dog, airplane, car, bird, koala, apple, apple2, bus, coffee, panda, elephant, strawberry, pizza, banana)"
        )
    ],
    dataset_name="ImageNet",
    visualization_type="image_with_predictions",
    theory=(
        "Convolutional Neural Networks (CNNs) are deep learning models designed for image "
        "classification. They learn hierarchical feature representations through layers of "
        "convolutions, pooling, and fully connected operations. Modern architectures like "
        "ResNet use residual connections (skip connections) to enable training of very deep "
        "networks by addressing the vanishing gradient problem. MobileNetV2 uses depthwise "
        "separable convolutions and inverted residuals for efficiency. The network takes a "
        "fixed-size RGB image (224x224), applies learned convolutional filters to extract "
        "features at multiple scales, and produces probability distributions over 1000 ImageNet "
        "classes. Transfer learning with pre-trained models enables high accuracy without "
        "training from scratch. The softmax activation in the final layer converts raw scores "
        "(logits) into probabilities that sum to 1.0, representing the model's confidence in "
        "each class prediction."
    ),
    pros=[
        "High accuracy with pre-trained models (80-90% top-5 accuracy)",
        "Learns hierarchical features automatically from data",
        "Transfer learning enables use without massive training data",
        "Multiple architecture options for speed/accuracy tradeoffs",
        "Robust to translation, rotation, and scale variations"
    ],
    cons=[
        "Requires significant computational resources (GPU recommended)",
        "Limited to classes seen during training (1000 ImageNet classes)",
        "Black box nature - difficult to interpret decisions",
        "Sensitive to input preprocessing and normalization",
        "May fail on out-of-distribution images"
    ],
    related_algorithms=["vgg16", "inception", "efficientnet", "vision-transformer"]
)

AlgorithmRegistry.register(image_classification_metadata)


@router.post("/image-classification/classify", response_model=ImageClassificationResponse)
async def classify_image(request: ImageClassificationRequest):
    """Run image classification on a sample image.

    This endpoint performs image classification using pre-trained CNNs,
    identifying objects and assigning confidence scores to top predictions.

    Args:
        request: ImageClassificationRequest containing classification parameters

    Returns:
        ImageClassificationResponse with classification results, statistics, and visualization data

    Raises:
        HTTPException: If classification fails or parameters are invalid
    """
    try:
        logger.info(f"Running image classification with parameters: {request.model_dump()}")

        # Initialize model
        model = ImageClassificationModel(model_name=request.model_name)

        # Process request
        response = model.process_request(request)

        logger.info(
            f"Image classification completed in {response.execution_time_ms:.2f}ms. "
            f"Top prediction: {response.predictions[0].class_name if response.predictions else 'None'} "
            f"({response.predictions[0].confidence:.3f} confidence)" if response.predictions else ""
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.get("/image-classification/info")
async def get_image_classification_info() -> Dict[str, Any]:
    """Get image classification algorithm information and metadata.

    Returns detailed information about the image classification algorithm including
    parameters, complexity, use cases, and available sample images.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("image-classification")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Image classification algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = ImageClassificationModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "model_variants": algo_info['model_variants']
    }


# Register SIFT metadata
sift_metadata = AlgorithmMetadata(
    id="sift",
    name="SIFT (Scale-Invariant Feature Transform)",
    slug="sift",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Detect and describe local features in images invariant to scale and rotation",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["computer-vision", "feature-extraction", "keypoint-detection", "image-matching"],
    use_cases=[
        "Image matching",
        "Object recognition",
        "Panorama stitching",
        "3D reconstruction",
        "Augmented reality"
    ],
    complexity=AlgorithmComplexity(
        time="O(n*log(n))",
        space="O(keypoints*128)"
    ),
    parameters=[
        AlgorithmParameter(
            name="nfeatures",
            label="Maximum Features",
            type="number",
            default=500,
            min=50,
            max=2000,
            step=50,
            description="Maximum number of features to detect (more = slower but more detailed)"
        ),
        AlgorithmParameter(
            name="nOctaveLayers",
            label="Octave Layers",
            type="number",
            default=3,
            min=1,
            max=5,
            step=1,
            description="Number of layers in each octave (scale levels)"
        ),
        AlgorithmParameter(
            name="contrastThreshold",
            label="Contrast Threshold",
            type="range",
            default=0.04,
            min=0.01,
            max=0.1,
            step=0.01,
            description="Contrast threshold for filtering weak features (higher = fewer features)"
        ),
        AlgorithmParameter(
            name="edgeThreshold",
            label="Edge Threshold",
            type="range",
            default=10,
            min=5,
            max=20,
            step=1,
            description="Edge threshold for filtering edge-like features (higher = fewer edge features)"
        ),
        AlgorithmParameter(
            name="sigma",
            label="Gaussian Sigma",
            type="range",
            default=1.6,
            min=0.5,
            max=3.0,
            step=0.1,
            description="Gaussian sigma for the first octave (smoothing level)"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="number",
            default=0,
            min=0,
            max=9,
            step=1,
            description="Select sample image (0-9: buildings, objects with textures)"
        ),
        AlgorithmParameter(
            name="match_mode",
            label="Enable Matching",
            type="boolean",
            default=False,
            description="Enable feature matching between two images"
        ),
        AlgorithmParameter(
            name="match_image_index",
            label="Second Image (Match)",
            type="number",
            default=1,
            min=0,
            max=9,
            step=1,
            description="Index of second image for matching mode"
        )
    ],
    dataset_name="Feature Detection Samples",
    visualization_type="image_with_keypoints",
    theory=(
        "SIFT (Scale-Invariant Feature Transform) is a feature detection algorithm that "
        "identifies and describes local features invariant to image scale, rotation, and "
        "illumination changes. Invented by David Lowe in 1999, SIFT works in four stages: "
        "(1) Scale-space extrema detection using Difference-of-Gaussians (DoG) pyramid to "
        "identify potential keypoints across multiple scales, (2) Keypoint localization and "
        "filtering to eliminate weak features and edge responses, (3) Orientation assignment "
        "by computing gradient histograms to achieve rotation invariance, and (4) Descriptor "
        "generation creating 128-dimensional feature vectors from local image gradients. "
        "Each keypoint is described by a 128-element vector that captures the gradient "
        "distribution in the keypoint's neighborhood, making it robust to transformations. "
        "SIFT features can be matched between images using distance metrics (e.g., L2 norm) "
        "combined with Lowe's ratio test to filter unreliable matches."
    ),
    pros=[
        "Scale and rotation invariant - detects features at multiple scales",
        "Robust to affine distortions and viewpoint changes",
        "Partially illumination invariant",
        "Distinctive descriptors enable reliable matching",
        "Well-established and widely used in computer vision"
    ],
    cons=[
        "Computationally expensive for real-time applications",
        "Not invariant to large affine transformations",
        "128-dimensional descriptors require more memory",
        "Patented algorithm (though patent expired in 2020)",
        "May detect too many features in highly textured images"
    ],
    related_algorithms=["surf", "orb", "akaze", "harris-corner"]
)

AlgorithmRegistry.register(sift_metadata)


@router.post("/sift/detect", response_model=SIFTResponse)
async def detect_sift_features(request: SIFTRequest):
    """Detect SIFT keypoints and compute descriptors.

    This endpoint performs SIFT feature detection, identifying scale and
    rotation-invariant keypoints and computing 128-dimensional descriptors.
    Optionally supports feature matching between two images.

    Args:
        request: SIFTRequest containing detection parameters

    Returns:
        SIFTResponse with keypoint detection results, statistics, and visualization data

    Raises:
        HTTPException: If detection fails or parameters are invalid
    """
    try:
        logger.info(f"Running SIFT feature detection with parameters: {request.model_dump()}")

        # Initialize model
        model = SIFTModel(
            nfeatures=request.nfeatures,
            nOctaveLayers=request.nOctaveLayers,
            contrastThreshold=request.contrastThreshold,
            edgeThreshold=request.edgeThreshold,
            sigma=request.sigma
        )

        # Process request
        response = model.process_request(request)

        logger.info(
            f"SIFT detection completed in {response.execution_time_ms:.2f}ms. "
            f"Detected {response.statistics.keypoint_count} keypoints."
        )

        if response.match_statistics:
            logger.info(
                f"Found {response.match_statistics.match_count} matches between images."
            )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"SIFT detection failed: {str(e)}"
        )


@router.get("/sift/info")
async def get_sift_info() -> Dict[str, Any]:
    """Get SIFT algorithm information and metadata.

    Returns detailed information about the SIFT algorithm including
    parameters, complexity, use cases, and available sample images.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("sift")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="SIFT algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = SIFTModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "algorithm_details": algo_info['algorithm_details']
    }


# Register Optical Flow metadata
optical_flow_metadata = AlgorithmMetadata(
    id="optical-flow",
    name="Optical Flow",
    slug="optical-flow",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Estimate motion between consecutive video frames",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["computer-vision", "motion-estimation", "video-analysis", "optical-flow"],
    use_cases=[
        "Video stabilization",
        "Object tracking",
        "Motion detection",
        "Autonomous navigation",
        "Sports analysis"
    ],
    complexity=AlgorithmComplexity(
        time="O(width*height*pyramid_levels)",
        space="O(width*height)"
    ),
    parameters=[
        AlgorithmParameter(
            name="method",
            label="Flow Algorithm",
            type="select",
            default="farneback",
            options=[
                {"label": "Farneback (Dense)", "value": "farneback"},
                {"label": "Lucas-Kanade (Sparse)", "value": "lucas-kanade"}
            ],
            description="Optical flow algorithm - Farneback computes dense flow, Lucas-Kanade computes sparse flow"
        ),
        AlgorithmParameter(
            name="pyr_scale",
            label="Pyramid Scale",
            type="range",
            default=0.5,
            min=0.3,
            max=0.9,
            step=0.1,
            description="Pyramid scale factor - smaller values detect larger motions"
        ),
        AlgorithmParameter(
            name="levels",
            label="Pyramid Levels",
            type="number",
            default=3,
            min=1,
            max=5,
            step=1,
            description="Number of pyramid levels - more levels detect larger motion range"
        ),
        AlgorithmParameter(
            name="winsize",
            label="Window Size",
            type="number",
            default=15,
            min=5,
            max=50,
            step=2,
            description="Window size for flow calculation - larger windows are more robust but less precise (must be odd)"
        ),
        AlgorithmParameter(
            name="iterations",
            label="Iterations",
            type="number",
            default=3,
            min=1,
            max=10,
            step=1,
            description="Number of iterations at each pyramid level - more iterations improve accuracy"
        ),
        AlgorithmParameter(
            name="image_pair_index",
            label="Sample Image Pair",
            type="number",
            default=0,
            min=0,
            max=2,
            step=1,
            description="Select sample image pair (0=horizontal motion, 1=mixed motion, 2=zoom motion)"
        )
    ],
    dataset_name="Optical Flow Sample Frames",
    visualization_type="flow_visualization",
    theory=(
        "Optical flow estimates the motion of objects between consecutive frames by analyzing "
        "changes in pixel intensities. It produces a vector field where each vector represents "
        "the apparent velocity of pixels. Two main approaches exist: (1) Farneback method uses "
        "polynomial expansion to approximate neighborhoods of pixels and compute dense flow for "
        "every pixel, making it suitable for general motion estimation. (2) Lucas-Kanade method "
        "assumes brightness constancy and computes sparse flow only at detected feature points, "
        "making it faster but less complete. Both methods use image pyramids (coarse-to-fine) to "
        "handle large motions. The pyramid approach first computes flow at a coarse scale, then "
        "refines it at finer scales. Key assumptions: brightness constancy (pixel intensities "
        "don't change between frames), small motion (displacement is small relative to image size), "
        "and spatial coherence (neighboring pixels have similar motion). Applications include "
        "video compression, action recognition, 3D reconstruction, and autonomous navigation."
    ),
    pros=[
        "Captures dense motion information across entire frame",
        "No training required - classical computer vision approach",
        "Works in real-time with efficient implementations",
        "Pyramid approach handles both small and large motions",
        "Useful for many downstream tasks (stabilization, tracking)"
    ],
    cons=[
        "Assumes brightness constancy (fails with lighting changes)",
        "Sensitive to noise and occlusions",
        "Aperture problem - ambiguous motion for uniform regions",
        "Cannot handle motion discontinuities well",
        "Parameter tuning required for different scenarios"
    ],
    related_algorithms=["lucas-kanade-tracker", "deepflow", "flownet", "raft"]
)

AlgorithmRegistry.register(optical_flow_metadata)


@router.post("/optical-flow/compute", response_model=OpticalFlowResponse)
async def compute_optical_flow(request: OpticalFlowRequest):
    """Compute optical flow between consecutive frames.

    This endpoint estimates motion between frames using either Farneback
    (dense flow) or Lucas-Kanade (sparse flow) algorithms.

    Args:
        request: OpticalFlowRequest containing flow computation parameters

    Returns:
        OpticalFlowResponse with flow results, statistics, and visualization data

    Raises:
        HTTPException: If flow computation fails or parameters are invalid
    """
    try:
        logger.info(f"Computing optical flow with parameters: {request.model_dump()}")

        # Initialize model
        model = OpticalFlowModel()

        # Process request
        response = model.process_request(request)

        logger.info(
            f"Optical flow computation completed in {response.execution_time_ms:.2f}ms. "
            f"Average magnitude: {response.statistics.average_magnitude:.3f}, "
            f"Flow coverage: {response.statistics.flow_coverage:.2f}%"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Optical flow computation failed: {str(e)}"
        )


@router.get("/optical-flow/info")
async def get_optical_flow_info() -> Dict[str, Any]:
    """Get optical flow algorithm information and metadata.

    Returns detailed information about the optical flow algorithm including
    parameters, complexity, use cases, and available sample image pairs.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("optical-flow")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Optical flow algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = OpticalFlowModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "algorithm_details": algo_info['algorithm_details']
    }


# Register Neural Style Transfer metadata
style_transfer_metadata = AlgorithmMetadata(
    id="style-transfer",
    name="Neural Style Transfer",
    slug="style-transfer",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Apply artistic style from one image to the content of another using CNNs",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["computer-vision", "style-transfer", "neural-networks", "generative", "vgg"],
    use_cases=[
        "Artistic image generation",
        "Photo enhancement",
        "Video stylization",
        "Game asset creation",
        "Creative design tools",
        "Social media filters"
    ],
    complexity=AlgorithmComplexity(
        time="O(iterations*image_size)",
        space="O(vgg_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="content_image_index",
            label="Content Image",
            type="number",
            default=0,
            min=0,
            max=9,
            step=1,
            description="Select content image (landscapes, portraits, architecture)"
        ),
        AlgorithmParameter(
            name="style_image_index",
            label="Style Image",
            type="number",
            default=0,
            min=0,
            max=9,
            step=1,
            description="Select artistic style (Van Gogh, Picasso, Monet, etc.)"
        ),
        AlgorithmParameter(
            name="iterations",
            label="Optimization Steps",
            type="number",
            default=300,
            min=50,
            max=1000,
            step=50,
            description="Number of optimization iterations (more = better quality but slower)"
        ),
        AlgorithmParameter(
            name="content_weight",
            label="Content Weight",
            type="range",
            default=1.0,
            min=0.1,
            max=10.0,
            step=0.1,
            description="Weight for content preservation (higher = more content preservation)"
        ),
        AlgorithmParameter(
            name="style_weight",
            label="Style Weight",
            type="range",
            default=1000000.0,
            min=100000.0,
            max=10000000.0,
            step=100000.0,
            description="Weight for style transfer (higher = stronger style application)"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.003,
            min=0.001,
            max=0.01,
            step=0.001,
            description="Optimizer learning rate (higher = faster convergence but less stable)"
        ),
        AlgorithmParameter(
            name="image_size",
            label="Output Size (px)",
            type="select",
            default=512,
            options=[
                {"label": "256x256 (Fast)", "value": 256},
                {"label": "512x512 (Balanced)", "value": 512},
                {"label": "1024x1024 (High Quality)", "value": 1024}
            ],
            description="Output image size - larger is higher quality but slower"
        )
    ],
    dataset_name="Style Transfer Image Pairs",
    visualization_type="three_panel_comparison",
    theory=(
        "Neural Style Transfer, introduced by Gatys et al. in 2015, is a technique that "
        "combines the content of one image with the artistic style of another. The algorithm "
        "uses a pre-trained CNN (VGG19) to extract feature representations at multiple layers. "
        "Content is represented by high-level features from deeper layers (conv4_2), which "
        "capture the image structure and objects. Style is represented by Gram matrices of "
        "features from multiple layers (conv1_1 through conv5_1), which capture texture and "
        "artistic patterns. The algorithm starts with a random or content-initialized image "
        "and iteratively optimizes it to minimize a weighted combination of content loss "
        "(MSE between content features) and style loss (MSE between Gram matrices). The Gram "
        "matrix computes correlations between feature maps, capturing style information while "
        "discarding spatial information. By adjusting content_weight and style_weight, users "
        "can control the balance between preserving content structure and applying artistic style. "
        "The optimization typically uses Adam or LBFGS optimizers in image space."
    ),
    pros=[
        "Creates stunning artistic effects from any style image",
        "No training required - uses pre-trained VGG19",
        "Flexible control via content/style weight balance",
        "Works with any content and style image combination",
        "Captures both texture and color of artistic styles"
    ],
    cons=[
        "Slow optimization (requires many iterations)",
        "Computationally expensive (GPU recommended)",
        "Sensitive to weight parameter selection",
        "May lose fine content details with strong stylization",
        "Different from fast feed-forward style transfer networks"
    ],
    related_algorithms=["fast-neural-style", "cyclegan", "pix2pix", "adain"]
)

AlgorithmRegistry.register(style_transfer_metadata)


@router.post("/style-transfer/stylize", response_model=StyleTransferResponse)
async def perform_style_transfer(request: StyleTransferRequest):
    """Perform neural style transfer on images.

    This endpoint applies the artistic style from a style image to the
    content of a content image using the Gatys et al. algorithm with VGG19.

    Args:
        request: StyleTransferRequest containing style transfer parameters

    Returns:
        StyleTransferResponse with stylized image, loss curves, and statistics

    Raises:
        HTTPException: If style transfer fails or parameters are invalid
    """
    try:
        logger.info(f"Running neural style transfer with parameters: {request.model_dump()}")

        # Initialize model
        model = StyleTransferModel()

        # Process request
        response = model.process_request(request)

        logger.info(
            f"Style transfer completed in {response.execution_time_ms:.2f}ms. "
            f"Total iterations: {response.statistics.total_iterations}, "
            f"Loss reduction: {response.statistics.loss_reduction:.2f}%"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Style transfer failed: {str(e)}"
        )


@router.get("/style-transfer/info")
async def get_style_transfer_info() -> Dict[str, Any]:
    """Get neural style transfer algorithm information and metadata.

    Returns detailed information about the neural style transfer algorithm including
    parameters, complexity, use cases, and available content/style images.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("style-transfer")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Style transfer algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = StyleTransferModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "algorithm_details": algo_info['algorithm_details']
    }


# Register Instance Segmentation metadata
instance_segmentation_metadata = AlgorithmMetadata(
    id="instance-segmentation",
    name="Instance Segmentation (Mask R-CNN)",
    slug="instance-segmentation",
    category=AlgorithmCategory.COMPUTER_VISION,
    description="Detect and segment individual object instances with pixel-level masks",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["computer-vision", "instance-segmentation", "mask-rcnn", "object-detection", "pixel-level"],
    use_cases=[
        "Autonomous driving",
        "Medical imaging",
        "Robotics perception",
        "Video analytics",
        "Retail automation",
        "Agricultural monitoring"
    ],
    complexity=AlgorithmComplexity(
        time="O(image_size*instances)",
        space="O(model_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="confidence_threshold",
            label="Confidence Threshold",
            type="range",
            default=0.5,
            min=0.1,
            max=0.9,
            step=0.05,
            description="Minimum confidence score for detections (higher = fewer but more confident instances)"
        ),
        AlgorithmParameter(
            name="model_backbone",
            label="Model Backbone",
            type="select",
            default="resnet50",
            options=[
                {"label": "ResNet50 (Balanced)", "value": "resnet50"},
                {"label": "ResNet101 (More Accurate)", "value": "resnet101"}
            ],
            description="Backbone network - trades off speed vs accuracy"
        ),
        AlgorithmParameter(
            name="mask_threshold",
            label="Mask Threshold",
            type="range",
            default=0.5,
            min=0.1,
            max=0.9,
            step=0.05,
            description="Binary threshold for mask generation (higher = tighter masks)"
        ),
        AlgorithmParameter(
            name="max_instances",
            label="Max Instances",
            type="number",
            default=100,
            min=10,
            max=200,
            step=10,
            description="Maximum number of instances to detect"
        ),
        AlgorithmParameter(
            name="image_index",
            label="Sample Image",
            type="number",
            default=0,
            min=0,
            max=9,
            step=1,
            description="Select sample image (0-9: diverse scenes with multiple objects)"
        ),
        AlgorithmParameter(
            name="nms_threshold",
            label="NMS Threshold",
            type="range",
            default=0.5,
            min=0.1,
            max=0.9,
            step=0.05,
            description="Non-Maximum Suppression IoU threshold (higher = more overlapping boxes)"
        )
    ],
    dataset_name="COCO",
    visualization_type="instance_segmentation",
    theory=(
        "Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation "
        "masks on each Region of Interest (RoI), in parallel with the existing branch for "
        "classification and bounding box regression. The architecture consists of: "
        "(1) A backbone CNN (ResNet-50-FPN or ResNet-101-FPN) for feature extraction, "
        "(2) Region Proposal Network (RPN) that generates candidate object proposals, "
        "(3) RoI Align layer that extracts features for each proposal without spatial quantization, "
        "(4) Box head that predicts class labels and refines bounding boxes, and "
        "(5) Mask head that predicts a binary mask for each instance using a small FCN. "
        "The mask branch predicts an m×m mask for each RoI using a fully convolutional "
        "network, allowing pixel-level instance segmentation. Unlike semantic segmentation "
        "which assigns each pixel a class label, instance segmentation distinguishes between "
        "different instances of the same class. RoI Align improves mask accuracy by avoiding "
        "quantization of RoI boundaries. The model is trained end-to-end with a multi-task "
        "loss combining classification, box regression, and mask prediction losses."
    ),
    pros=[
        "Pixel-accurate instance segmentation",
        "Handles multiple instances of the same class",
        "Unified architecture for detection and segmentation",
        "State-of-the-art accuracy on COCO dataset",
        "Pre-trained models available for transfer learning",
        "Robust to overlapping objects"
    ],
    cons=[
        "Computationally expensive (requires GPU for real-time)",
        "Slower than object detection alone",
        "Requires significant memory for large images",
        "Training requires pixel-level mask annotations",
        "May struggle with very small or heavily occluded objects"
    ],
    related_algorithms=["faster-rcnn", "yolo", "semantic-segmentation", "panoptic-segmentation"]
)

AlgorithmRegistry.register(instance_segmentation_metadata)


@router.post("/instance-segmentation/segment", response_model=InstanceSegmentationResponse)
async def segment_instances(request: InstanceSegmentationRequest):
    """Run instance segmentation on a sample image.

    This endpoint performs instance segmentation using Mask R-CNN,
    detecting individual object instances and generating pixel-level masks.

    Args:
        request: InstanceSegmentationRequest containing segmentation parameters

    Returns:
        InstanceSegmentationResponse with instance results, statistics, and visualization data

    Raises:
        HTTPException: If segmentation fails or parameters are invalid
    """
    try:
        logger.info(f"Running instance segmentation with parameters: {request.model_dump()}")

        # Initialize model
        model = InstanceSegmentationModel(
            model_backbone=request.model_backbone
        )

        # Process request
        response = model.process_request(request)

        logger.info(
            f"Instance segmentation completed in {response.execution_time_ms:.2f}ms. "
            f"Detected {response.statistics.total_instances} instances "
            f"across {response.statistics.unique_classes} classes."
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Instance segmentation failed: {str(e)}"
        )


@router.get("/instance-segmentation/info")
async def get_instance_segmentation_info() -> Dict[str, Any]:
    """Get instance segmentation algorithm information and metadata.

    Returns detailed information about the Mask R-CNN algorithm including
    parameters, complexity, use cases, and available sample images.

    Returns:
        Dictionary containing algorithm metadata and configuration
    """
    metadata = AlgorithmRegistry.get("instance-segmentation")
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Instance segmentation algorithm metadata not found"
        )

    # Get additional algorithm info
    algo_info = InstanceSegmentationModel.get_algorithm_info()

    return {
        "metadata": metadata.model_dump(),
        "dataset": algo_info['dataset'],
        "algorithm_details": algo_info['algorithm_details']
    }
