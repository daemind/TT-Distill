"""Model Tools for Hugging Face model inference.

This module provides tool functions for running inference on Hugging Face models
for vision, audio, and NLP tasks. These tools can be called by agents as needed.

Tools are registered in the MCPManager and can be used by any agent with
appropriate rights.
"""

from __future__ import annotations

# ruff: noqa: PLC0415, PERF401
import os
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)

# Model cache for lazy loading
_model_cache: dict[str, Any] = {}


def get_model_path(model_id: str) -> Path:
    """Get the local path for a model (cached or to be downloaded).

    Args:
        model_id: Hugging Face model ID

    Returns:
        Path to the model directory
    """
    cache_dir = Path(os.getenv("HF_CACHE_DIR", "./hf_cache"))
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / model_id.replace("/", "_")


async def yolo_inference(
    image_path: str,
    model_id: str = "yolov8",
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Perform object detection using YOLO model.

    Args:
        image_path: Path to input image
        model_id: YOLO model ID (default: 'yolov8')
        confidence_threshold: Confidence threshold for detections

    Returns:
        Dictionary with detections and metadata
    """
    try:
        from ultralytics import YOLO

        model_path = get_model_path(model_id)
        if model_id not in _model_cache:
            logger.info(f"Loading YOLO model: {model_id}")
            _model_cache[model_id] = YOLO(model_path / "yolov8.pt")

        model = _model_cache[model_id]
        results = model(
            image_path,
            conf=confidence_threshold,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            detections.append(
                {
                    "class": results[0].names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": [float(x) for x in box.xyxy[0]],
                }
            )

        return {
            "success": True,
            "model_id": model_id,
            "image_path": image_path,
            "detections": detections,
            "count": len(detections),
        }

    except ImportError:
        logger.error("ultralytics not installed. Install with: pip install ultralytics")
        return {
            "success": False,
            "error": "ultralytics not installed",
            "model_id": model_id,
            "image_path": image_path,
        }
    except Exception as e:
        logger.error(f"YOLO inference failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_id": model_id,
            "image_path": image_path,
        }


async def whisper_transcribe(
    audio_path: str,
    model_id: str = "whisper-small",
    language: str | None = None,
) -> dict[str, Any]:
    """Transcribe audio using Whisper model.

    Args:
        audio_path: Path to audio file
        model_id: Whisper model ID (default: 'whisper-small')
        language: Optional language code (e.g., 'en', 'fr')

    Returns:
        Dictionary with transcription and metadata
    """
    try:
        import whisper

        model_path = get_model_path(model_id)
        if model_id not in _model_cache:
            logger.info(f"Loading Whisper model: {model_id}")
            _model_cache[model_id] = whisper.load_model(
                model_id.replace("whisper-", ""),
                download_root=str(model_path.parent),
            )

        model = _model_cache[model_id]
        result = model.transcribe(
            audio_path,
            language=language,
            verbose=False,
        )

        return {
            "success": True,
            "model_id": model_id,
            "audio_path": audio_path,
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in result.get("segments", [])
            ],
        }

    except ImportError:
        logger.error(
            "openai-whisper not installed. Install with: pip install openai-whisper"
        )
        return {
            "success": False,
            "error": "openai-whisper not installed",
            "model_id": model_id,
            "audio_path": audio_path,
        }
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_id": model_id,
            "audio_path": audio_path,
        }


async def vit_classify(
    image_path: str,
    model_id: str = "vit-base",
    top_k: int = 5,
) -> dict[str, Any]:
    """Classify image using Vision Transformer model.

    Args:
        image_path: Path to input image
        model_id: ViT model ID (default: 'vit-base')
        top_k: Number of top predictions to return

    Returns:
        Dictionary with classifications and metadata
    """
    try:
        import torch
        from PIL import Image
        from transformers import ViTForImageClassification, ViTImageProcessor

        get_model_path(model_id)
        if model_id not in _model_cache:
            logger.info(f"Loading ViT model: {model_id}")
            _model_cache[model_id] = {
                "processor": ViTImageProcessor.from_pretrained(model_id),
                "model": ViTForImageClassification.from_pretrained(model_id),
            }

        cache = _model_cache[model_id]
        processor = cache["processor"]
        model = cache["model"]

        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]

        top_probs, top_indices = torch.topk(probabilities, top_k)
        classifications = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=False):
            label = model.config.id2label[idx]
            classifications.append(
                {
                    "label": label,
                    "confidence": prob,
                }
            )

        return {
            "success": True,
            "model_id": model_id,
            "image_path": image_path,
            "classifications": classifications,
        }

    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_id": model_id,
            "image_path": image_path,
        }
    except Exception as e:
        logger.error(f"ViT classification failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_id": model_id,
            "image_path": image_path,
        }


async def segment_anything(
    image_path: str,
    model_id: str = "sam-vit-base",
) -> dict[str, Any]:
    """Perform semantic segmentation using Segment Anything model.

    Args:
        image_path: Path to input image
        model_id: SAM model ID (default: 'sam-vit-base')

    Returns:
        Dictionary with segmentation masks and metadata
    """
    try:
        from PIL import Image
        from transformers import SamModel, SamProcessor

        get_model_path(model_id)
        if model_id not in _model_cache:
            logger.info(f"Loading SAM model: {model_id}")
            _model_cache[model_id] = {
                "processor": SamProcessor.from_pretrained(model_id),
                "model": SamModel.from_pretrained(model_id),
            }

        cache = _model_cache[model_id]
        processor = cache["processor"]
        model = cache["model"]

        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        masks = outputs.pred_masks

        return {
            "success": True,
            "model_id": model_id,
            "image_path": image_path,
            "masks_shape": list(masks.shape),
            "num_masks": masks.shape[1],
        }

    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_id": model_id,
            "image_path": image_path,
        }
    except Exception as e:
        logger.error(f"SAM segmentation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_id": model_id,
            "image_path": image_path,
        }


# Type alias for tool metadata
ToolMetadata = dict[str, Any]

# Tool registration metadata for MCP
MODEL_TOOLS: dict[str, ToolMetadata] = {
    "yolo_inference": {
        "function": yolo_inference,
        "description": "Perform object detection using YOLO model",
        "parameters": {
            "image_path": {
                "type": "string",
                "description": "Path to input image",
            },
            "model_id": {
                "type": "string",
                "description": "YOLO model ID",
                "default": "yolov8",
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Confidence threshold for detections",
                "default": 0.5,
            },
        },
        "required": ["image_path"],
    },
    "whisper_transcribe": {
        "function": whisper_transcribe,
        "description": "Transcribe audio using Whisper model",
        "parameters": {
            "audio_path": {
                "type": "string",
                "description": "Path to audio file",
            },
            "model_id": {
                "type": "string",
                "description": "Whisper model ID",
                "default": "whisper-small",
            },
            "language": {
                "type": "string",
                "description": "Optional language code (e.g., 'en', 'fr')",
                "default": None,
            },
        },
        "required": ["audio_path"],
    },
    "vit_classify": {
        "function": vit_classify,
        "description": "Classify image using Vision Transformer model",
        "parameters": {
            "image_path": {
                "type": "string",
                "description": "Path to input image",
            },
            "model_id": {
                "type": "string",
                "description": "ViT model ID",
                "default": "vit-base",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top predictions to return",
                "default": 5,
            },
        },
        "required": ["image_path"],
    },
    "segment_anything": {
        "function": segment_anything,
        "description": "Perform semantic segmentation using SAM model",
        "parameters": {
            "image_path": {
                "type": "string",
                "description": "Path to input image",
            },
            "model_id": {
                "type": "string",
                "description": "SAM model ID",
                "default": "sam-vit-base",
            },
        },
        "required": ["image_path"],
    },
}
