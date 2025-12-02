"""Algorithm plugins for image processing."""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .image_processor import AlgorithmPlugin


class YOLOPlugin(AlgorithmPlugin):
    """YOLO object detection plugin."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize YOLO plugin.
        
        Args:
            model_path: Path to YOLO model file (.onnx, .weights, etc.)
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            logger: Optional logger
        """
        self.log = logger or logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._model = None
        self._net = None
        self._output_layers = None
        self._input_size = (640, 640)  # Default YOLO input size
        self._classes = []
        self._initialized = False
    
    def _load_model(self) -> bool:
        """Load YOLO model (lazy loading)."""
        if self._initialized:
            return True
        
        try:
            if self.model_path and self.model_path.endswith('.onnx'):
                # ONNX model
                import onnxruntime as ort
                self._model = ort.InferenceSession(self.model_path)
                self._initialized = True
                self.log.info(f"Loaded ONNX model: {self.model_path}")
                return True
            elif self.model_path and self.model_path.endswith(('.weights', '.cfg')):
                # Darknet YOLO
                import cv2
                cfg_path = self.model_path.replace('.weights', '.cfg')
                self._net = cv2.dnn.readNetFromDarknet(cfg_path, self.model_path)
                layer_names = self._net.getLayerNames()
                self._output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]
                self._initialized = True
                self.log.info(f"Loaded Darknet model: {self.model_path}")
                return True
            else:
                self.log.warning("No valid model path provided")
                return False
        except ImportError as e:
            self.log.error(f"Required library not installed: {e}")
            return False
        except Exception as e:
            self.log.error(f"Failed to load model: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if YOLO model is ready."""
        if not self._initialized:
            return self._load_model()
        return True
    
    def process(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run YOLO detection on image.
        
        Returns:
            Dictionary with detections:
                - boxes: List of [x1, y1, x2, y2]
                - scores: List of confidence scores
                - classes: List of class IDs
                - class_names: List of class names (if available)
        """
        if not self.is_ready():
            return {"error": "Model not ready"}
        
        try:
            if self._model is not None:
                # ONNX inference
                return self._process_onnx(image)
            elif self._net is not None:
                # Darknet inference
                return self._process_darknet(image)
            else:
                return {"error": "No model loaded"}
        except Exception as e:
            self.log.error(f"YOLO inference failed: {e}")
            return {"error": str(e)}
    
    def _process_onnx(self, image: np.ndarray) -> Dict[str, Any]:
        """Process with ONNX model."""
        try:
            import cv2
            # Preprocess
            blob = cv2.dnn.blobFromImage(
                image, 1/255.0, self._input_size, swapRB=True, crop=False
            )
            
            # Inference
            outputs = self._model.run(None, {self._model.get_inputs()[0].name: blob})
            
            # Post-process (simplified, actual implementation depends on model)
            # This is a placeholder - actual YOLO post-processing is more complex
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "class_names": []
            }
        except Exception as e:
            self.log.error(f"ONNX processing failed: {e}")
            return {"error": str(e)}
    
    def _process_darknet(self, image: np.ndarray) -> Dict[str, Any]:
        """Process with Darknet model."""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, self._input_size, swapRB=True, crop=False
        )
        
        self._net.setInput(blob)
        outputs = self._net.forward(self._output_layers)
        
        # Parse detections (simplified)
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    box_w = int(detection[2] * w)
                    box_h = int(detection[3] * h)
                    
                    x1 = int(center_x - box_w / 2)
                    y1 = int(center_y - box_h / 2)
                    
                    boxes.append([x1, y1, x1 + box_w, y1 + box_h])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            final_boxes = [boxes[i] for i in indices]
            final_scores = [confidences[i] for i in indices]
            final_classes = [class_ids[i] for i in indices]
        else:
            final_boxes = []
            final_scores = []
            final_classes = []
        
        return {
            "boxes": final_boxes,
            "scores": final_scores,
            "classes": final_classes,
            "class_names": [self._classes[i] if i < len(self._classes) else f"class_{i}" 
                          for i in final_classes]
        }


class SAM3Plugin(AlgorithmPlugin):
    """SAM3 (Segment Anything Model 3) plugin for image segmentation."""
    
    def __init__(
        self,
        text_prompt: Optional[str] = None,
        enabled: bool = True,
        output_segmented_image: bool = True,
        mask_threshold: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SAM3 plugin.
        
        Args:
            text_prompt: Text prompt for segmentation (e.g., "person", "car")
            enabled: Whether processing is enabled
            output_segmented_image: Whether to output segmented image in results
            mask_threshold: Threshold for mask binarization
            logger: Optional logger
        """
        self.log = logger or logging.getLogger(self.__class__.__name__)
        self.text_prompt = text_prompt
        self.enabled = enabled
        self.output_segmented_image = output_segmented_image
        self.mask_threshold = mask_threshold
        
        self._model = None
        self._processor = None
        self._inference_state = None
        self._initialized = False
    
    def _load_model(self) -> bool:
        """Load SAM3 model (lazy loading)."""
        if self._initialized:
            return True
        
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            self._model = build_sam3_image_model()
            self._processor = Sam3Processor(self._model)
            self._initialized = True
            self.log.info("SAM3 model loaded successfully")
            return True
        except ImportError as e:
            self.log.error(f"SAM3 library not installed: {e}")
            self.log.error("Install with: pip install sam3")
            return False
        except Exception as e:
            self.log.error(f"Failed to load SAM3 model: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if SAM3 model is ready."""
        if not self._initialized:
            return self._load_model()
        return self._model is not None and self._processor is not None
    
    def enable(self) -> None:
        """Enable processing."""
        self.enabled = True
        self.log.info("SAM3 processing enabled")
    
    def disable(self) -> None:
        """Disable processing."""
        self.enabled = False
        self.log.info("SAM3 processing disabled")
    
    def set_text_prompt(self, prompt: str) -> None:
        """Update text prompt."""
        self.text_prompt = prompt
        self._inference_state = None  # Reset state when prompt changes
        self.log.info(f"Text prompt updated: {prompt}")
    
    def process(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run SAM3 segmentation on image.
        
        Args:
            image: Input image (BGR format)
            metadata: Optional metadata
            
        Returns:
            Dictionary with segmentation results:
                - masks: List of segmentation masks
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - scores: List of confidence scores
                - segmented_image: Segmented image (if output_segmented_image=True)
                - enabled: Whether processing was enabled
        """
        if not self.enabled:
            return {
                "enabled": False,
                "message": "SAM3 processing is disabled"
            }
        
        if not self.is_ready():
            return {"error": "SAM3 model not ready"}
        
        if not self.text_prompt:
            return {"error": "No text prompt provided"}
        
        try:
            from PIL import Image
            
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Set image (always set for new image, inference state is per-image)
            self._inference_state = self._processor.set_image(pil_image)
            
            # Set text prompt and get output
            output = self._processor.set_text_prompt(
                state=self._inference_state,
                prompt=self.text_prompt
            )
            
            masks = output.get("masks", [])
            boxes = output.get("boxes", [])
            scores = output.get("scores", [])
            
            result = {
                "enabled": True,
                "masks": masks,
                "boxes": boxes,
                "scores": scores,
                "num_segments": len(masks),
                "text_prompt": self.text_prompt
            }
            
            # Generate segmented image if requested
            if self.output_segmented_image and masks:
                segmented_image = self._apply_masks(image, masks, boxes, scores)
                result["segmented_image"] = segmented_image
            
            return result
        except Exception as e:
            self.log.error(f"SAM3 inference failed: {e}")
            return {"error": str(e)}
    
    def _apply_masks(
        self,
        image: np.ndarray,
        masks: list,
        boxes: list,
        scores: list
    ) -> np.ndarray:
        """
        Apply masks to image and return segmented image.
        
        Args:
            image: Original image
            masks: List of mask arrays
            boxes: List of bounding boxes
            scores: List of scores
            
        Returns:
            Segmented image with masks applied
        """
        try:
            # Create overlay image
            overlay = image.copy()
            h, w = image.shape[:2]
            
            # Apply each mask
            for i, mask in enumerate(masks):
                if mask is None:
                    continue
                
                # Convert mask to numpy array if needed
                if not isinstance(mask, np.ndarray):
                    # Assume mask is in PIL or other format, convert to numpy
                    if hasattr(mask, 'numpy'):
                        mask_np = mask.numpy()
                    elif hasattr(mask, 'array'):
                        mask_np = np.array(mask.array())
                    else:
                        mask_np = np.array(mask)
                else:
                    mask_np = mask
                
                # Ensure mask is 2D and matches image size
                if len(mask_np.shape) > 2:
                    mask_np = mask_np.squeeze()
                
                # Resize mask if needed
                if mask_np.shape[0] != h or mask_np.shape[1] != w:
                    mask_np = cv2.resize(mask_np.astype(np.float32), (w, h))
                
                # Binarize mask
                mask_binary = (mask_np > self.mask_threshold).astype(np.uint8)
                
                # Create colored mask (green overlay)
                color_mask = np.zeros_like(image)
                color_mask[mask_binary > 0] = [0, 255, 0]  # Green color
                
                # Blend with original image
                overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
                
                # Draw bounding box if available
                if i < len(boxes) and boxes[i]:
                    box = boxes[i]
                    if len(box) >= 4:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw score if available
                        if i < len(scores) and scores[i] is not None:
                            score_text = f"{scores[i]:.2f}"
                            cv2.putText(
                                overlay, score_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
            
            return overlay
        except Exception as e:
            self.log.error(f"Failed to apply masks: {e}")
            return image


class DummyPlugin(AlgorithmPlugin):
    """Dummy plugin for testing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(self.__class__.__name__)
    
    def is_ready(self) -> bool:
        return True
    
    def process(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return dummy results."""
        return {
            "status": "processed",
            "image_shape": image.shape,
            "timestamp": metadata.get("timestamp") if metadata else None
        }

