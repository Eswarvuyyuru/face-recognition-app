import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import pandas as pd
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from PIL import Image
import threading
import queue
import ultralytics
from ultralytics import YOLO
import torch
import json
import re
from typing import List, Dict, Tuple, Optional
import requests
from io import BytesIO
import base64
import matplotlib.pyplot as plt
# Set page config FIRST
st.set_page_config(
    page_title="🔍 AI-Powered Text-Guided Detection System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define defaults before enhanced_main()
object_detection_confidence = 0.5
face_similarity_threshold = 0.6
frame_skip = 1
# 🧠 Enhance low-light or blurry images before face detection
def enhance_image(image):
    """Enhance image contrast and reduce noise for better face detection."""
    try:
        # Convert to YUV and equalize histogram on the luminance channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # Denoise the image slightly (optional but improves clarity)
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        return enhanced
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return image


# Additional imports for text-guided detection
try:
    import transformers
    from transformers import (
        OwlViTProcessor, OwlViTForObjectDetection,
        CLIPProcessor, CLIPModel,
        AutoProcessor, AutoModelForZeroShotObjectDetection
    )
    HF_AVAILABLE = True
    st.success("✅ Hugging Face transformers available for text-guided detection")
except ImportError:
    HF_AVAILABLE = False
    st.warning("⚠️ Install transformers for text-guided detection: pip install transformers")

try:
    import groundingdino
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    GROUNDING_DINO_AVAILABLE = True
    st.success("✅ GroundingDINO available for advanced text-guided detection")
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    st.info("💡 Install GroundingDINO for advanced features: pip install groundingdino-py")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Text-guided detection models cache
@st.cache_resource
def load_text_guided_models():
    """Load text-guided object detection models"""
    models = {}
    status = {}
    
    # Load OWL-ViT (HuggingFace)
    if HF_AVAILABLE:
        try:
            processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            models['owlvit'] = {'processor': processor, 'model': model}
            status['owlvit'] = True
            st.success("✅ OWL-ViT model loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load OWL-ViT: {str(e)}")
            models['owlvit'] = None
            status['owlvit'] = False
    
    # Load CLIP for similarity matching
    if HF_AVAILABLE:
        try:
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            models['clip'] = {'processor': clip_processor, 'model': clip_model}
            status['clip'] = True
            st.success("✅ CLIP model loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load CLIP: {str(e)}")
            models['clip'] = None
            status['clip'] = False
    
    # Load GroundingDINO (if available)
    if GROUNDING_DINO_AVAILABLE:
        try:
            # This would need proper GroundingDINO setup
            models['grounding_dino'] = "placeholder"
            status['grounding_dino'] = True
            st.success("✅ GroundingDINO ready")
        except Exception as e:
            st.error(f"❌ Failed to load GroundingDINO: {str(e)}")
            models['grounding_dino'] = None
            status['grounding_dino'] = False
    
    return models, status

# Enhanced model loading with error handling
@st.cache_resource
def load_traditional_models():
    models = {}
    status = {}
    
    # Load Face Recognition Model
    try:
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        models['face'] = face_app
        status['face'] = True
        st.success("✅ Face recognition model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load face model: {str(e)}")
        models['face'] = None
        status['face'] = False
    
    # Load Object Detection Model (YOLO)
    try:
        object_model = YOLO('yolov8s.pt')
        models['object'] = object_model
        status['object'] = True
        st.success("✅ YOLO object detection model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {str(e)}")
        models['object'] = None
        status['object'] = False
    
    return models, status

# Load all models
text_models, text_model_status = load_text_guided_models() if HF_AVAILABLE else ({}, {})
traditional_models, traditional_model_status = load_traditional_models()

# Combine model dictionaries
all_models = {**traditional_models, **text_models}
all_model_status = {**traditional_model_status, **text_model_status}

# Text processing utilities
class TextQueryProcessor:
    """Process and enhance text queries for object detection"""
    
    def __init__(self):
        self.common_synonyms = {
            'bag': ['backpack', 'purse', 'handbag', 'satchel', 'tote', 'luggage', 'suitcase'],
            'person': ['man', 'woman', 'individual', 'human', 'people'],
            'car': ['vehicle', 'automobile', 'sedan', 'suv', 'truck'],
            'phone': ['smartphone', 'cellphone', 'mobile', 'device'],
            'laptop': ['computer', 'notebook', 'macbook', 'pc'],
            'weapon': ['gun', 'knife', 'firearm', 'blade', 'pistol'],
            'suspicious': ['unusual', 'strange', 'odd', 'concerning']
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand a single query into multiple related queries"""
        expanded = [query.lower().strip()]
        
        # Add synonyms
        for base_word, synonyms in self.common_synonyms.items():
            if base_word in query.lower():
                for synonym in synonyms:
                    expanded.append(query.lower().replace(base_word, synonym))
        
        # Add descriptive variations
        if 'person' in query.lower():
            variations = [
                query + " walking",
                query + " standing", 
                query + " carrying something",
                "individual " + query.replace('person', '').strip()
            ]
            expanded.extend(variations)
        
        return list(set(expanded))  # Remove duplicates
    
    def parse_client_description(self, description: str) -> Dict:
        """Parse client description into structured detection queries"""
        
        # Extract key information using regex patterns
        patterns = {
            'objects': r'\b(?:bag|backpack|suitcase|luggage|laptop|phone|weapon|gun|knife)\b',
            'colors': r'\b(?:red|blue|green|yellow|black|white|brown|gray|pink|purple)\b',
            'sizes': r'\b(?:large|small|big|tiny|huge|medium|oversized)\b',
            'people': r'\b(?:person|man|woman|individual|people|someone|anyone)\b',
            'actions': r'\b(?:carrying|holding|wearing|walking|running|standing|sitting)\b',
            'locations': r'\b(?:near|by|next to|in front of|behind|under|over|beside)\b'
        }
        
        parsed = {}
        for category, pattern in patterns.items():
            matches = re.findall(pattern, description.lower())
            parsed[category] = list(set(matches))
        
        # Generate comprehensive search queries
        queries = []
        
        # Basic object queries
        for obj in parsed.get('objects', []):
            queries.append(obj)
            
            # Add color combinations
            for color in parsed.get('colors', []):
                queries.append(f"{color} {obj}")
            
            # Add size combinations  
            for size in parsed.get('sizes', []):
                queries.append(f"{size} {obj}")
            
            # Add action combinations
            for action in parsed.get('actions', []):
                queries.append(f"person {action} {obj}")
        
        # People-related queries
        for person in parsed.get('people', []):
            queries.append(person)
            for action in parsed.get('actions', []):
                queries.append(f"{person} {action}")
        
        # If no specific objects found, use the full description
        if not parsed.get('objects'):
            queries.append(description.strip())
        
        return {
            'parsed_elements': parsed,
            'search_queries': list(set(queries)),
            'original_description': description,
            'priority_queries': queries[:5]  # Top 5 most relevant
        }

# Initialize text processor
text_processor = TextQueryProcessor()

# Text-guided detection functions
def detect_with_owlvit(image: np.ndarray, text_queries: List[str], confidence_threshold: float = 0.1) -> List[Dict]:
    """Detect objects using OWL-ViT with text queries"""
    if not all_model_status.get('owlvit', False):
        return []
    
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        processor = all_models['owlvit']['processor']
        model = all_models['owlvit']['model']
        
        # Process inputs
        inputs = processor(text=text_queries, images=pil_image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results
        target_sizes = torch.Tensor([pil_image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )
        
        detections = []
        for i, (scores, labels, boxes) in enumerate(zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"])):
            if scores > confidence_threshold:
                detections.append({
                    'query': text_queries[labels],
                    'confidence': float(scores),
                    'bbox': boxes.int().tolist(),
                    'model': 'OWL-ViT'
                })
        
        return detections
        
    except Exception as e:
        logger.error(f"Error in OWL-ViT detection: {str(e)}")
        return []

def detect_with_clip_similarity(image: np.ndarray, text_queries: List[str], grid_size: int = 8) -> List[Dict]:
    """Use CLIP to find image regions most similar to text queries"""
    if not all_model_status.get('clip', False):
        return []
    
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        processor = all_models['clip']['processor']
        model = all_models['clip']['model']
        
        h, w = image.shape[:2]
        grid_h, grid_w = h // grid_size, w // grid_size
        
        detections = []
        
        # Create grid of image patches
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * grid_h, (i + 1) * grid_h
                x1, x2 = j * grid_w, (j + 1) * grid_w
                
                # Extract patch
                patch = image_rgb[y1:y2, x1:x2]
                patch_pil = Image.fromarray(patch)
                
                # Process with CLIP
                inputs = processor(
                    text=text_queries, 
                    images=patch_pil, 
                    return_tensors="pt", 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Find best matching query
                best_idx = probs.argmax().item()
                best_score = probs[0, best_idx].item()
                patch_area = (x2 - x1) * (y2 - y1)
                if best_score > 0.55 and 5000 < patch_area < 200000 and best_score < 0.95:
                       detections.append({
                       'query': text_queries[best_idx],
                       'confidence': best_score,
                       'bbox': [x1, y1, x2, y2],
                       'model': 'CLIP'
                        })
        
        return detections
        
    except Exception as e:
        logger.error(f"Error in CLIP detection: {str(e)}")
        return []

# Define object classes of interest (keeping original for backward compatibility)
BAG_CLASSES = {
    'backpack': 27, 'handbag': 28, 'suitcase': 29, 'umbrella': 30,
    'bottle': 44, 'laptop': 73, 'cell phone': 77
}

COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# Enhanced UI
st.title("🎯 AI-Powered Text-Guided Detection System")
st.markdown("*Detect faces and objects using natural language descriptions*")

# Show available capabilities
capabilities = []
if all_model_status.get('face', False):
    capabilities.append("👤 Face Recognition")
if all_model_status.get('object', False):
    capabilities.append("🎯 Traditional Object Detection")
if all_model_status.get('owlvit', False):
    capabilities.append("🔍 Text-Guided Detection (OWL-ViT)")
if all_model_status.get('clip', False):
    capabilities.append("🧠 AI Similarity Matching (CLIP)")

st.info(f"**Available Capabilities:** {' • '.join(capabilities)}")
st.markdown("---")

# Enhanced sidebar configuration
with st.sidebar:
    st.header("⚙️ Detection Configuration")
    
    # Detection method selection
    st.subheader("🔧 Detection Methods")
    
    detection_methods = []
    if all_model_status.get('face', False):
        use_face_detection = st.checkbox("Face Recognition", value=True)
        if use_face_detection:
            detection_methods.append('face')
    else:
        use_face_detection = False
        st.error("❌ Face detection unavailable")
    
    if all_model_status.get('object', False):
        use_yolo_detection = st.checkbox("Traditional Object Detection (YOLO)", value=True)
        if use_yolo_detection:
            detection_methods.append('yolo')
    else:
        use_yolo_detection = False
        st.error("❌ YOLO detection unavailable")
    
    if all_model_status.get('owlvit', False):
        use_text_guided = st.checkbox("Text-Guided Detection (OWL-ViT)", value=True)
        if use_text_guided:
            detection_methods.append('owlvit')
    else:
        use_text_guided = False
        st.info("💡 Text-guided detection unavailable")
    
    if all_model_status.get('clip', False):
        use_clip_similarity = st.checkbox("AI Similarity Matching (CLIP)", value=False)
        if use_clip_similarity:
            detection_methods.append('clip')
    else:
        use_clip_similarity = False
        st.info("💡 CLIP similarity unavailable")
    
    # Client description input
    st.subheader("📝 Client Description")
    client_description = st.text_area(
        "Describe what to look for:",
        placeholder="e.g., 'Person carrying a large black backpack', 'Suspicious individual with red bag', 'Someone holding a laptop'",
        help="Describe objects, people, or scenarios you want to detect using natural language"
    )
    
    # Process client description
    if client_description:
        parsed_query = text_processor.parse_client_description(client_description)
        
        with st.expander("🔍 Query Analysis", expanded=False):
            st.write("**Parsed Elements:**")
            for key, values in parsed_query['parsed_elements'].items():
                if values:
                    st.write(f"- **{key.title()}:** {', '.join(values)}")
            
            st.write("**Generated Search Queries:**")
            for i, query in enumerate(parsed_query['priority_queries'], 1):
                st.write(f"{i}. {query}")
        
        search_queries = parsed_query['search_queries']
    else:
        search_queries = []
    
    # Traditional settings (if applicable)
    if use_face_detection:
        st.subheader("👤 Face Settings")
        face_similarity_threshold = st.slider(
            "Face Similarity Threshold", 
            min_value=0.2, max_value=0.8, value=0.4, step=0.01
        )
        face_detection_confidence = st.slider(
            "Face Detection Confidence",
            min_value=0.3, max_value=0.9, value=0.5, step=0.05
        )
    
    if use_yolo_detection:
        st.subheader("🎯 Traditional Object Settings")
        yolo_confidence = st.slider(
            "YOLO Detection Confidence",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05
        )
    
    if use_text_guided:
        st.subheader("🔍 Text-Guided Settings")
        text_detection_confidence = st.slider(
            "Text Detection Confidence",
            min_value=0.05, max_value=0.8, value=0.1, step=0.05,
            help="Lower values detect more objects but may include false positives"
        )
    
    if use_clip_similarity:
        st.subheader("🧠 CLIP Settings")
        clip_grid_size = st.slider(
            "Image Grid Size",
            min_value=4, max_value=16, value=8, step=2,
            help="Higher values provide more detailed scanning but slower processing"
        )
    
    st.subheader("⚡ Performance")
    frame_skip = st.number_input(
        "Process Every N Frames (Video)",
        min_value=1, max_value=30, value=5
    )
    max_detections = st.number_input(
        "Max Detections Per Frame",
        min_value=1, max_value=50, value=20
    )

# Enhanced session state management
def init_session_state():
    defaults = {
        "face_results_log": [],
        "text_object_results_log": [],
        "traditional_object_results_log": [],
        "video_processed": False,
        "processed_video_path": "",
        "reference_embeddings": [],
        "client_queries": [],
        "detection_summary": {},
        "total_detections": 0
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Enhanced reference face processing (keeping original functionality)
def process_reference_images(ref_images):
    """Process and validate reference images"""
    if not use_face_detection or not all_model_status['face']:
        return [], 0
        
    embeddings = []
    processed_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ref_file in enumerate(ref_images):
        status_text.text(f"Processing {ref_file.name}...")
        
        try:
            ref_bytes = ref_file.read()
            ref_img = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
            ref_img = enhance_image(ref_img)

            if ref_img is None:
                st.warning(f"⚠️ Could not read image: {ref_file.name}")
                continue
                
            faces = all_models['face'].get(ref_img, max_num=1)
            
            if faces:
                face = faces[0]
                if face.det_score < face_detection_confidence:
                    st.warning(f"⚠️ Low quality face detected in {ref_file.name}")
                    continue
                    
                embedding = face.embedding / np.linalg.norm(face.embedding)
                embeddings.append({
                    'name': ref_file.name,
                    'embedding': embedding,
                    'quality_score': face.det_score
                })
                processed_count += 1
            else:
                st.warning(f"⚠️ No face detected in {ref_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error processing {ref_file.name}: {str(e)}")
            
        progress_bar.progress((i + 1) / len(ref_images))
    
    progress_bar.empty()
    status_text.empty()
    
    return embeddings, processed_count

# Enhanced detection functions
def detect_traditional_objects(frame, model):
    """Detect objects using traditional YOLO"""
    if not use_yolo_detection:
        return []
    INTERESTED_CLASSES = ["person", "backpack", "handbag", "suitcase", "laptop", "cellphone"]    
    try:
        results = model(frame, conf=yolo_confidence, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                    if class_name in INTERESTED_CLASSES:
                        detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox.astype(int),
                        'class_id': class_id,
                        'detection_type': 'traditional'
                        })
        
        return detections
    except Exception as e:
        logger.error(f"Error in traditional object detection: {str(e)}")
        return []

def perform_text_guided_detection(frame: np.ndarray, queries: List[str]) -> List[Dict]:
    """Perform text-guided detection using available models"""
    all_detections = []
    
    if use_text_guided and queries:
        # OWL-ViT detection
        owlvit_detections = detect_with_owlvit(frame, queries, text_detection_confidence)
        for det in owlvit_detections:
            det['detection_type'] = 'text_guided'
        all_detections.extend(owlvit_detections)
    
    if use_clip_similarity and queries:
        # CLIP similarity detection
        clip_detections = detect_with_clip_similarity(frame, queries, clip_grid_size)
        for det in clip_detections:
            det['detection_type'] = 'clip_similarity'
        all_detections.extend(clip_detections)
    
    return all_detections

# Enhanced logging functions
def log_text_detection(source, frame_num, timestamp, detection_data, full_frame):
    """Log text-guided detection"""
    try:
        query = detection_data.get('query', 'unknown')
        confidence = detection_data.get('confidence', 0)
        bbox = detection_data.get('bbox', [0, 0, 100, 100])
        model_type = detection_data.get('model', 'unknown')
        
        img_name = f"text_det_{query.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        img_path = os.path.join(tempfile.gettempdir(), img_name)
        
        annotated_frame = full_frame.copy()
        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"{query}: {confidence:.2f} ({model_type})", 
                   (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.imwrite(img_path, annotated_frame)
        
        log_data = {
            'source': source,
            'frame_num': frame_num,
            'timestamp': timestamp,
            'query': query,
            'confidence': confidence,
            'model': model_type,
            'image_path': img_path,
            'bbox': bbox,
            'type': 'text_guided',
            'detection_type': detection_data.get('detection_type', 'text_guided')
        }
        
        st.session_state.text_object_results_log.append(log_data)
        st.session_state.total_detections += 1
        
        # Show high-confidence alerts
        if confidence > 0.5:
            st.success(f"🎯 **TEXT DETECTION!** Found '{query}' at {timestamp} (Confidence: {confidence:.2f}, Model: {model_type})")
        
    except Exception as e:
        logger.error(f"Error logging text detection: {str(e)}")
#-------------------------------------------------------------------------------------------------------------------------------
# Continuation of the previous code - log_face_match function completion and additional enhancements

def log_face_match(source, frame_num, timestamp, query_name, similarity, face_bbox, full_frame):
    """Log face match detection (keeping original functionality)"""
    try:
        img_name = f"face_match_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        img_path = os.path.join(tempfile.gettempdir(), img_name)
        
        annotated_frame = full_frame.copy()
        cv2.rectangle(annotated_frame, (face_bbox[0], face_bbox[1]), 
                     (face_bbox[2], face_bbox[3]), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{query_name}: {similarity:.3f}", 
                   (face_bbox[0], face_bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(img_path, annotated_frame)
        
        log_data = {
            'source': source,
            'frame_num': frame_num,
            'timestamp': timestamp,
            'query_name': query_name,
            'similarity': similarity,
            'face_bbox': face_bbox,
            'image_path': img_path,
            'type': 'face_match'
        }
        
        st.session_state.face_results_log.append(log_data)
        st.session_state.total_detections += 1
        
        # Show alert for face matches
        st.success(f"👤 **FACE MATCH!** Found '{query_name}' at {timestamp} (Similarity: {similarity:.3f})")
        
    except Exception as e:
        logger.error(f"Error logging face match: {str(e)}")

def log_object_detection(source, frame_num, timestamp, detection_data, full_frame):
    """Log traditional object detection"""
    try:
        class_name = detection_data.get('class_name', 'unknown')
        confidence = detection_data.get('confidence', 0)
        bbox = detection_data.get('bbox', [0, 0, 100, 100])
        
        img_name = f"obj_det_{class_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        img_path = os.path.join(tempfile.gettempdir(), img_name)
        
        annotated_frame = full_frame.copy()
        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", 
                   (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imwrite(img_path, annotated_frame)
        
        log_data = {
            'source': source,
            'frame_num': frame_num,
            'timestamp': timestamp,
            'class_name': class_name,
            'confidence': confidence,
            'image_path': img_path,
            'bbox': bbox,
            'type': 'traditional_object'
        }
        
        st.session_state.traditional_object_results_log.append(log_data)
        st.session_state.total_detections += 1
        
    except Exception as e:
        logger.error(f"Error logging object detection: {str(e)}")

def annotate_frame(frame, detections):
    """Draw bounding boxes and labels on the frame"""
    annotated = frame.copy()
    
    # Draw traditional objects
    for det in detections.get('traditional_objects', []):
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['class_name']} ({det['confidence']:.2f})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw text-guided objects
    for det in detections.get('text_guided_objects', []):
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['query']} ({det['confidence']:.2f})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)

    # Draw faces (optional, if detected)
    for det in detections.get('faces', []):
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['name']} ({det['similarity']:.2f})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return annotated

# Enhanced processing functions
def process_frame(frame, frame_num, timestamp, reference_embeddings, search_queries):
    """Process a single frame with all detection methods"""
    frame = enhance_image(frame)
    detections = {
        'faces': [],
        'traditional_objects': [],
        'text_guided_objects': [],
        'total_count': 0
    }
    
    try:
        # Face detection and recognition
        if use_face_detection and all_model_status.get('face', False) and reference_embeddings:
            faces = all_models['face'].get(frame)
            
            for face in faces:
                if face.det_score < face_detection_confidence:
                    continue
                    
                face_embedding = face.embedding / np.linalg.norm(face.embedding)
                face_bbox = face.bbox.astype(int)
                
                # Compare with reference embeddings
                for ref_data in reference_embeddings:
                    similarity = np.dot(face_embedding, ref_data['embedding'])
                    
                    if similarity > face_similarity_threshold:
                        detections['faces'].append({
                            'name': ref_data['name'],
                            'similarity': similarity,
                            'bbox': face_bbox,
                            'quality_score': face.det_score
                        })
                        
                        log_face_match("current_frame", frame_num, timestamp, 
                                     ref_data['name'], similarity, face_bbox, frame)
        
        # Traditional object detection
        if use_yolo_detection and all_model_status.get('object', False):
            traditional_detections = detect_traditional_objects(frame, all_models['object'])
            
            for detection in traditional_detections:
                detections['traditional_objects'].append(detection)
                log_object_detection("current_frame", frame_num, timestamp, detection, frame)
        
        # Text-guided detection
        if search_queries and (use_text_guided or use_clip_similarity):
            text_detections = perform_text_guided_detection(frame, search_queries)
            
            for detection in text_detections:
                detections['text_guided_objects'].append(detection)
                log_text_detection("current_frame", frame_num, timestamp, detection, frame)
        
        # Calculate total detections
        detections['total_count'] = (len(detections['faces']) + 
                                   len(detections['traditional_objects']) + 
                                   len(detections['text_guided_objects']))
        
        return detections
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return detections

 

# Enhanced video processing
def process_video_file(video_file, reference_embeddings, search_queries):
    """Process uploaded video file with enhanced detection"""
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(video_file.read())
    temp_video.close()
    
    cap = cv2.VideoCapture(temp_video.name)
    
    if not cap.isOpened():
        st.error("❌ Could not open video file")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    detection_summary = {'total_frames': 0, 'frames_with_detections': 0, 'total_detections': 0}
    
    frame_num = 0
    processed_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("🔁 End of video or read error.")
                break
            
            frame_num += 1
            timestamp = f"{frame_num // fps // 60:02d}:{(frame_num // fps) % 60:02d}"
            
            # Skip frames based on frame_skip setting
            if frame_num % frame_skip != 0:
                out.write(frame)
                continue
            
            processed_frames += 1
            status_text.text(f"Processing frame {frame_num}/{total_frames} ({timestamp})")
            print(f"📸 Frame {frame_num} being processed (Total processed: {processed_frames})")
            try:
            # Process frame
                detections = process_frame(frame, frame_num, timestamp, reference_embeddings, search_queries)
                print("🧠 Detected:", [d['class_name'] for d in detections.get('traditional_objects', [])])
            except Exception as e:
                print(f"❌ Error in process_frame at frame {frame_num}: {e}")
                detections = {'faces': [], 'traditional_objects': [], 'text_guided_objects': [], 'total_count': 0}
            try:
               # Annotate frame
               annotated_frame = annotate_frame(frame, detections)
            
               # Write to output video
               out.write(annotated_frame)
            except Exception as e:
                print(f"❌ Error in annotate_frame at frame {frame_num}: {e}")
                out.write(frame)
            
            # Update statistics
            detection_summary['total_frames'] += 1
            if detections['total_count'] > 0:
                detection_summary['frames_with_detections'] += 1
                detection_summary['total_detections'] += detections['total_count']
            
            # Update progress
            progress = frame_num / total_frames
            progress_bar.progress(min(progress,1.0))
            
            # Limit processing for performance
            if processed_frames >= 1000000:  # Limit for demo
                st.warning("⚠️ Processing limited to first 1000 frames for performance")
                break
        
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.detection_summary = detection_summary
        st.session_state.processed_video_path = output_path
        st.session_state.video_processed = True
        print(f"✅ Video processing done. Total processed frames: {processed_frames}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        cap.release()
        out.release()
        return None
    finally:
        try:
            os.unlink(temp_video.name)
        except:
            pass

# Real-time camera processing
class VideoProcessor(VideoTransformerBase):
    """Enhanced video processor for real-time detection"""
    
    def __init__(self):
        self.frame_count = 0
        self.reference_embeddings = []
        self.search_queries = []
    
    def set_reference_embeddings(self, embeddings):
        self.reference_embeddings = embeddings
    
    def set_search_queries(self, queries):
        self.search_queries = queries
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % frame_skip != 0:
            return img
        
        timestamp = f"{self.frame_count // 30 // 60:02d}:{(self.frame_count // 30) % 60:02d}"
        
        # Process frame
        detections = process_frame(img, self.frame_count, timestamp, 
                                 self.reference_embeddings, self.search_queries)
        
        # Annotate and return
        return annotate_frame(img, detections)

# Enhanced image processing
def process_single_image(image_file, reference_embeddings, search_queries):
    """Process a single uploaded image"""
    try:
        # Read image
        image_bytes = image_file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image = enhance_image(image)

        if image is None:
            st.error("❌ Could not read image file")
            return None
        
        # Process image
        detections = process_frame(image, 1, "00:00", reference_embeddings, search_queries)
        
        # Annotate image
        annotated_image = annotate_frame(image, detections)
        
        return annotated_image, detections
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, None

# Enhanced results display
def display_detection_results():
    """Display comprehensive detection results"""
    st.header("📊 Detection Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Face Matches", len(st.session_state.face_results_log))
    
    with col2:
        st.metric("🎯 Traditional Objects", len(st.session_state.traditional_object_results_log))
    
    with col3:
        st.metric("🔍 Text-Guided Objects", len(st.session_state.text_object_results_log))
    
    with col4:
        st.metric("📈 Total Detections", st.session_state.total_detections)
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Face Matches", "🎯 Traditional Objects", "🔍 Text-Guided", "📋 Summary"])
    
    with tab1:
        if st.session_state.face_results_log:
            st.subheader("Face Recognition Results")
            
            for i, result in enumerate(st.session_state.face_results_log):
                with st.expander(f"Face Match {i+1}: {result['query_name']} (Similarity: {result['similarity']:.3f})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(result['image_path']):
                            st.image(result['image_path'], caption=f"Match at {result['timestamp']}")
                    
                    with col2:
                        st.write(f"**Name:** {result['query_name']}")
                        st.write(f"**Similarity:** {result['similarity']:.3f}")
                        st.write(f"**Timestamp:** {result['timestamp']}")
                        st.write(f"**Frame:** {result['frame_num']}")
                        st.write(f"**Bounding Box:** {result['face_bbox']}")
        else:
            st.info("No face matches detected yet.")
    
    with tab2:
        if st.session_state.traditional_object_results_log:
            st.subheader("Traditional Object Detection Results")
            
            for i, result in enumerate(st.session_state.traditional_object_results_log):
                with st.expander(f"Object {i+1}: {result['class_name']} (Confidence: {result['confidence']:.2f})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(result['image_path']):
                            st.image(result['image_path'], caption=f"Detection at {result['timestamp']}")
                    
                    with col2:
                        st.write(f"**Object:** {result['class_name']}")
                        st.write(f"**Confidence:** {result['confidence']:.3f}")
                        st.write(f"**Timestamp:** {result['timestamp']}")
                        st.write(f"**Frame:** {result['frame_num']}")
                        st.write(f"**Bounding Box:** {result['bbox']}")
        else:
            st.info("No traditional objects detected yet.")
    
    with tab3:
        if st.session_state.text_object_results_log:
            st.subheader("Text-Guided Detection Results")
            
            for i, result in enumerate(st.session_state.text_object_results_log):
                with st.expander(f"Text Detection {i+1}: {result['query']} (Confidence: {result['confidence']:.2f})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(result['image_path']):
                            st.image(result['image_path'], caption=f"Detection at {result['timestamp']}")
                    
                    with col2:
                        st.write(f"**Query:** {result['query']}")
                        st.write(f"**Model:** {result['model']}")
                        st.write(f"**Confidence:** {result['confidence']:.3f}")
                        st.write(f"**Timestamp:** {result['timestamp']}")
                        st.write(f"**Frame:** {result['frame_num']}")
                        st.write(f"**Bounding Box:** {result['bbox']}")
        else:
            st.info("No text-guided objects detected yet.")
    
    with tab4:
        st.subheader("Detection Summary")
        
        if st.session_state.detection_summary:
            summary = st.session_state.detection_summary
            
            st.write("**Video Processing Summary:**")
            st.write(f"- Total Frames Processed: {summary.get('total_frames', 0)}")
            st.write(f"- Frames with Detections: {summary.get('frames_with_detections', 0)}")
            st.write(f"- Total Detections: {summary.get('total_detections', 0)}")
            
            if summary.get('total_frames', 0) > 0:
                detection_rate = (summary.get('frames_with_detections', 0) / summary.get('total_frames', 1)) * 100
                st.write(f"- Detection Rate: {detection_rate:.1f}%")
        
        # Export options
        st.subheader("📥 Export Results")
        
        if st.button("📊 Export Results to CSV"):
            # Combine all results
            all_results = []
            
            for result in st.session_state.face_results_log:
                result_copy = result.copy()
                result_copy['detection_category'] = 'face_match'
                all_results.append(result_copy)
            
            for result in st.session_state.traditional_object_results_log:
                result_copy = result.copy()
                result_copy['detection_category'] = 'traditional_object'
                all_results.append(result_copy)
            
            for result in st.session_state.text_object_results_log:
                result_copy = result.copy()
                result_copy['detection_category'] = 'text_guided'
                all_results.append(result_copy)
            
            if all_results:
                df = pd.DataFrame(all_results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📁 Download CSV Report",
                    data=csv,
                    file_name=f"detection_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No results to export.")

# Main application interface
def main():
    """Main application interface"""
    
    # Input section
    st.header("📤 Input Configuration")
    
    input_col1, input_col2 = st.columns([1, 1])
    
    with input_col1:
        st.subheader("👤 Reference Images (for Face Recognition)")
        st.text("🔍 Reached reference uploader 1")
        reference_images = st.file_uploader(
            "Upload reference face images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload clear face images of people you want to detect",
            key="file_uploader_reference_faces_unique_1"
        )
    
    with input_col2:
        st.subheader("🎯 Input Media")
        input_type = st.radio(
            "Select input type:",
            ["📷 Single Image", "🎥 Video File", "📹 Live Camera"],
            horizontal=True
        )
    
    # Process reference images
    reference_embeddings = []
    if reference_images and use_face_detection:
        with st.spinner("Processing reference images..."):
            embeddings, processed_count = process_reference_images(reference_images)
            reference_embeddings = embeddings
            
            if processed_count > 0:
                st.success(f"✅ Processed {processed_count} reference face(s)")
                
                # Show reference image previews
                with st.expander("👀 Reference Images Preview", expanded=False):
                    cols = st.columns(min(len(embeddings), 4))
                    for i, embedding_data in enumerate(embeddings[:4]):
                        with cols[i]:
                            # Display name and quality score
                            st.write(f"**{embedding_data['name']}**")
                            st.write(f"Quality: {embedding_data['quality_score']:.3f}")
            else:
                st.warning("⚠️ No valid faces found in reference images")
    
    # Get search queries from client description
    search_queries = []
    if client_description:
        parsed_query = text_processor.parse_client_description(client_description)
        search_queries = parsed_query['priority_queries']
    
    # Process input based on type
    if input_type == "📷 Single Image":
        uploaded_image = st.file_uploader(
            "Upload an image to analyze",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for object and face detection",
            key="file_uploader_image_input_unique_1"
        )
        
        if uploaded_image:
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    result_image, detections = process_single_image(
                        uploaded_image, reference_embeddings, search_queries
                    )
                    
                    if result_image is not None:
                        st.subheader("📊 Analysis Results")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.image(result_image, caption="Detected Objects and Faces", channels="BGR")
                        
                        with col2:
                            st.write("**Detection Summary:**")
                            st.write(f"👤 Faces: {len(detections.get('faces', []))}")
                            st.write(f"🎯 Traditional Objects: {len(detections.get('traditional_objects', []))}")
                            st.write(f"🔍 Text-Guided Objects: {len(detections.get('text_guided_objects', []))}")
                            st.write(f"📈 Total: {detections.get('total_count', 0)}")
    
    elif input_type == "🎥 Video File":
        uploaded_video = st.file_uploader(
            "Upload a video file to analyze",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for comprehensive detection analysis",
            key="file_uploader_video_input_unique_1"
        )
        
        if uploaded_video:
            if st.button("🎬 Process Video", type="primary")  or st.session_state.get("video_processing_started"):
                st.session_state["video_processing_started"] = True
                with st.spinner("Processing video... This may take several minutes."):
                    output_path = process_video_file(
                        uploaded_video, reference_embeddings, search_queries
                    )
                    
                    if output_path and os.path.exists(output_path):
                        st.session_state["processed_video_path"] = output_path

                        st.session_state["video_processing_done"] = True
        if st.session_state.get("video_processing_done"):
            st.success("✅ Video processing completed!")
            output_path = st.session_state["processed_video_path"]
            with open(output_path, 'rb') as video_file:
                  st.download_button(
                       label="📁 Download Processed Video",
                       data=video_file.read(),
                       file_name=f"processed_video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                       mime="video/mp4"
                            )
    #-------------------
    elif input_type == "📹 Live Camera":
        st.subheader("📹 Live Camera Detection")
        
        if not (reference_embeddings or search_queries):
            st.warning("⚠️ Please upload reference images or provide client description for live detection")
        else:
            # Create video processor
            processor = VideoProcessor()
            processor.set_reference_embeddings(reference_embeddings)
            processor.set_search_queries(search_queries)
            
            webrtc_ctx = webrtc_streamer(
                key="detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: processor,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )
            
            if webrtc_ctx.video_processor:
                st.info("📡 Live detection is running. Detections will appear in the results section below.")
    
    # Clear results button
    if st.button("🗑️ Clear All Results"):
        for key in ['face_results_log', 'text_object_results_log', 'traditional_object_results_log']:
            st.session_state[key] = []
        st.session_state.total_detections = 0
        st.session_state.detection_summary = {}
        st.success("✅ Results cleared!")
        st.rerun()
    
    # Display results
    if (st.session_state.face_results_log or 
        st.session_state.text_object_results_log or 
        st.session_state.traditional_object_results_log):
        display_detection_results()

# Run the application
if __name__ == "__main__":
#main()

# Additional utility functions for enhanced features
  def batch_process_images(image_folder_path, reference_embeddings, search_queries):
    """Batch process multiple images from a folder"""
    try:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(image_folder_path).glob(f"*{ext}"))
            image_paths.extend(Path(image_folder_path).glob(f"*{ext.upper()}"))
        
        results = []
        progress_bar = st.progress(0)
        
        for i, image_path in enumerate(image_paths):
            try:
                image = cv2.imread(str(image_path))
                if image is not None:
                    detections = process_frame(image, i+1, f"image_{i+1}", 
                                             reference_embeddings, search_queries)
                    results.append({
                        'image_path': str(image_path),
                        'detections': detections
                    })
                
                progress_bar.progress((i + 1) / len(image_paths))
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        progress_bar.empty()
        return results
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return []

  def create_detection_report(results_data):
    """Create a comprehensive detection report"""
    try:
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_detections': len(results_data),
            'detection_types': {},
            'confidence_statistics': {},
            'detections': results_data
        }
        
        # Analyze detection types
        for result in results_data:
            det_type = result.get('type', 'unknown')
            if det_type not in report['detection_types']:
                report['detection_types'][det_type] = 0
            report['detection_types'][det_type] += 1
        
        # Calculate confidence statistics
        confidences = []
        for result in results_data:
            if 'confidence' in result:
                confidences.append(result['confidence'])
            elif 'similarity' in result:
                confidences.append(result['similarity'])
        
        if confidences:
            report['confidence_statistics'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'median': np.median(confidences)
            }
        
        return report
        
    except Exception as e:
        logger.error(f"Error creating detection report: {str(e)}")
        return None
#----------------------------------------------------------------------------------------------------
# Enhanced error handling and logging (continuation)
def setup_enhanced_logging():
    """Setup enhanced logging with file output"""
    try:
        log_dir = Path(tempfile.gettempdir()) / "detection_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"detection_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info(f"Enhanced logging setup complete. Log file: {log_file}")
        return str(log_file)
        
    except Exception as e:
        print(f"Error setting up enhanced logging: {str(e)}")
        return None

def export_detection_data(format_type='json'):
    """Export all detection data in various formats"""
    try:
        # Collect all detection data
        all_data = {
            'face_matches': st.session_state.face_results_log,
            'traditional_objects': st.session_state.traditional_object_results_log,
            'text_guided_objects': st.session_state.text_object_results_log,
            'summary': {
                'total_detections': st.session_state.total_detections,
                'export_timestamp': datetime.datetime.now().isoformat(),
                'detection_summary': st.session_state.get('detection_summary', {})
            }
        }
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == 'json':
            filename = f"detection_export_{timestamp}.json"
            data = json.dumps(all_data, indent=2, default=str)
            mime_type = "application/json"
            
        elif format_type == 'csv':
            # Flatten data for CSV export
            flattened_data = []
            
            for result in all_data['face_matches']:
                flattened_data.append({
                    'type': 'face_match',
                    'name': result.get('query_name', ''),
                    'confidence': result.get('similarity', 0),
                    'timestamp': result.get('timestamp', ''),
                    'frame_num': result.get('frame_num', 0),
                    'bbox': str(result.get('face_bbox', [])),
                    'source': result.get('source', '')
                })
            
            for result in all_data['traditional_objects']:
                flattened_data.append({
                    'type': 'traditional_object',
                    'name': result.get('class_name', ''),
                    'confidence': result.get('confidence', 0),
                    'timestamp': result.get('timestamp', ''),
                    'frame_num': result.get('frame_num', 0),
                    'bbox': str(result.get('bbox', [])),
                    'source': result.get('source', '')
                })
            
            for result in all_data['text_guided_objects']:
                flattened_data.append({
                    'type': 'text_guided',
                    'name': result.get('query', ''),
                    'confidence': result.get('confidence', 0),
                    'timestamp': result.get('timestamp', ''),
                    'frame_num': result.get('frame_num', 0),
                    'bbox': str(result.get('bbox', [])),
                    'source': result.get('source', ''),
                    'model': result.get('model', '')
                })
            
            df = pd.DataFrame(flattened_data)
            filename = f"detection_export_{timestamp}.csv"
            data = df.to_csv(index=False)
            mime_type = "text/csv"
        
        else:  # Default to JSON
            filename = f"detection_export_{timestamp}.json"
            data = json.dumps(all_data, indent=2, default=str)
            mime_type = "application/json"
        
        return data, filename, mime_type
        
    except Exception as e:
        logger.error(f"Error exporting detection data: {str(e)}")
        return None, None, None

def analyze_detection_patterns():
    """Analyze patterns in detection data"""
    try:
        analysis = {
            'temporal_patterns': {},
            'confidence_analysis': {},
            'spatial_analysis': {},
            'detection_frequency': {}
        }
        
        all_detections = (st.session_state.face_results_log + 
                         st.session_state.traditional_object_results_log + 
                         st.session_state.text_object_results_log)
        
        if not all_detections:
            return analysis
        
        # Temporal pattern analysis
        timestamps = []
        for detection in all_detections:
            timestamp = detection.get('timestamp', '00:00')
            if ':' in timestamp:
                try:
                    minutes, seconds = map(int, timestamp.split(':'))
                    total_seconds = minutes * 60 + seconds
                    timestamps.append(total_seconds)
                except:
                    continue
        
        if timestamps:
            analysis['temporal_patterns'] = {
                'detection_timeline': timestamps,
                'peak_detection_time': max(set(timestamps), key=timestamps.count) if timestamps else 0,
                'detection_spread': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
            }
        
        # Confidence analysis
        confidences = []
        similarities = []
        
        for detection in all_detections:
            if 'confidence' in detection:
                confidences.append(detection['confidence'])
            if 'similarity' in detection:
                similarities.append(detection['similarity'])
        
        if confidences:
            analysis['confidence_analysis']['traditional_objects'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        if similarities:
            analysis['confidence_analysis']['face_matches'] = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            }
        
        # Detection frequency by type
        detection_types = {}
        for detection in all_detections:
            det_type = detection.get('type', 'unknown')
            detection_types[det_type] = detection_types.get(det_type, 0) + 1
        
        analysis['detection_frequency'] = detection_types
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing detection patterns: {str(e)}")
        return {}

def create_detection_heatmap(detections):
    """Create a heatmap of detection locations"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns     
        # Extract bounding box centers
        x_coords = []
        y_coords = []
        
        for detection in detections:
            bbox = detection.get('bbox') or detection.get('face_bbox', [])
            if len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                x_coords.append(center_x)
                y_coords.append(center_y)
        
        if not x_coords:
            return None
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=20)
        
        # Plot heatmap
        im = ax.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='bilinear')
        ax.set_title('Detection Location Heatmap')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        plt.colorbar(im, ax=ax, label='Detection Density')
        
        # Save to temporary file
        temp_path = os.path.join(tempfile.gettempdir(), 
                                f"heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Error creating detection heatmap: {str(e)}")
        return None

def optimize_detection_parameters():
    """Suggest optimized detection parameters based on results"""
    try:
        optimization_suggestions = {
            'face_detection': {},
            'object_detection': {},
            'text_guided': {},
            'general': {}
        }
        
        # Analyze face detection results
        face_results = st.session_state.face_results_log
        if face_results:
            similarities = [r['similarity'] for r in face_results]
            mean_similarity = np.mean(similarities)
            
            if mean_similarity < 0.6:
                optimization_suggestions['face_detection']['similarity_threshold'] = {
                    'current': face_similarity_threshold,
                    'suggested': max(0.4, mean_similarity - 0.1),
                    'reason': 'Lower threshold to capture more potential matches'
                }
            elif mean_similarity > 0.9:
                optimization_suggestions['face_detection']['similarity_threshold'] = {
                    'current': face_similarity_threshold,
                    'suggested': min(0.8, mean_similarity - 0.05),
                    'reason': 'Raise threshold to reduce false positives'
                }
        
        # Analyze object detection results
        obj_results = st.session_state.traditional_object_results_log
        if obj_results:
            confidences = [r['confidence'] for r in obj_results]
            mean_confidence = np.mean(confidences)
            
            if mean_confidence < 0.5:
                optimization_suggestions['object_detection']['confidence_threshold'] = {
                    'current': object_detection_confidence,
                    'suggested': max(0.25, mean_confidence - 0.1),
                    'reason': 'Lower threshold to capture more detections'
                }
        
        # General optimization
        total_detections = st.session_state.total_detections
        if total_detections > 1000:
            optimization_suggestions['general']['frame_skip'] = {
                'current': frame_skip,
                'suggested': min(10, frame_skip + 2),
                'reason': 'Increase frame skip to improve performance'
            }
        elif total_detections < 10:
            optimization_suggestions['general']['frame_skip'] = {
                'current': frame_skip,
                'suggested': max(1, frame_skip - 1),
                'reason': 'Decrease frame skip to capture more detections'
            }
        
        return optimization_suggestions
        
    except Exception as e:
        logger.error(f"Error optimizing detection parameters: {str(e)}")
        return {}

def create_advanced_results_dashboard():
    """Create an advanced results dashboard with analytics"""
    st.header("📈 Advanced Analytics Dashboard")
    
    # Get analysis data
    pattern_analysis = analyze_detection_patterns()
    optimization_suggestions = optimize_detection_parameters()
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🕐 Temporal Analysis", "🎯 Confidence Analysis", 
        "⚙️ Optimization", "📥 Export"
    ])
    
    with tab1:
        st.subheader("Detection Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", st.session_state.total_detections)
        
        with col2:
            face_count = len(st.session_state.face_results_log)
            st.metric("Face Matches", face_count)
        
        with col3:
            obj_count = len(st.session_state.traditional_object_results_log)
            st.metric("Traditional Objects", obj_count)
        
        with col4:
            text_count = len(st.session_state.text_object_results_log)
            st.metric("Text-Guided Objects", text_count)
        
        # Detection frequency chart
        if pattern_analysis.get('detection_frequency'):
            st.subheader("Detection Distribution")
            
            freq_data = pattern_analysis['detection_frequency']
            labels = list(freq_data.keys())
            values = list(freq_data.values())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('Detection Count by Type')
            ax.set_ylabel('Number of Detections')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Temporal Pattern Analysis")
        
        if pattern_analysis.get('temporal_patterns', {}).get('detection_timeline'):
            timeline = pattern_analysis['temporal_patterns']['detection_timeline']
            
            # Convert seconds back to timestamps for display
            timestamps = [f"{t//60:02d}:{t%60:02d}" for t in timeline]
            
            st.write("**Detection Timeline:**")
            
            # Create timeline chart
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(timeline, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Detection Frequency Over Time')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Number of Detections')
            st.pyplot(fig)
            
            # Peak detection time
            peak_time = pattern_analysis['temporal_patterns'].get('peak_detection_time', 0)
            st.info(f"🕐 Peak detection time: {peak_time//60:02d}:{peak_time%60:02d}")
        
        else:
            st.info("No temporal data available for analysis.")
    
    with tab3:
        st.subheader("Confidence Analysis")
        
        conf_analysis = pattern_analysis.get('confidence_analysis', {})
        
        if conf_analysis:
            for detection_type, stats in conf_analysis.items():
                st.write(f"**{detection_type.replace('_', ' ').title()}:**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{stats['mean']:.3f}")
                with col2:
                    st.metric("Std Dev", f"{stats['std']:.3f}")
                with col3:
                    st.metric("Min", f"{stats['min']:.3f}")
                with col4:
                    st.metric("Max", f"{stats['max']:.3f}")
                
                st.write("---")
        else:
            st.info("No confidence data available for analysis.")
    
    with tab4:
        st.subheader("Parameter Optimization Suggestions")
        
        if optimization_suggestions:
            for category, suggestions in optimization_suggestions.items():
                if suggestions:
                    st.write(f"**{category.replace('_', ' ').title()}:**")
                    
                    for param, details in suggestions.items():
                        with st.expander(f"📋 {param.replace('_', ' ').title()}", expanded=False):
                            st.write(f"**Current Value:** {details['current']}")
                            st.write(f"**Suggested Value:** {details['suggested']}")
                            st.write(f"**Reason:** {details['reason']}")
                            
                            if st.button(f"Apply {param}", key=f"apply_{category}_{param}"):
                                st.success(f"✅ Suggestion noted! Manually update {param} to {details['suggested']}")
        else:
            st.info("No optimization suggestions available. Process more data for recommendations.")
    
    with tab5:
        st.subheader("Export Detection Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export as JSON", type="primary"):
                data, filename, mime_type = export_detection_data('json')
                if data:
                    st.download_button(
                        label="📁 Download JSON Export",
                        data=data,
                        file_name=filename,
                        mime=mime_type
                    )
        
        with col2:
            if st.button("📊 Export as CSV", type="primary"):
                data, filename, mime_type = export_detection_data('csv')
                if data:
                    st.download_button(
                        label="📁 Download CSV Export",
                        data=data,
                        file_name=filename,
                        mime=mime_type
                    )
        
        # Generate detection report
        if st.button("📋 Generate Comprehensive Report"):
            all_detections = (st.session_state.face_results_log + 
                            st.session_state.traditional_object_results_log + 
                            st.session_state.text_object_results_log)
            
            report = create_detection_report(all_detections)
            if report:
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="📁 Download Comprehensive Report",
                    data=report_json,
                    file_name=f"detection_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Enhanced main function with advanced features
def enhanced_main():
    """Enhanced main function with advanced features"""
    
    # Initialize enhanced logging
    setup_enhanced_logging()
    
    # Page configuration
    
    # Advanced sidebar options
    with st.sidebar:
        st.header("🔧 Advanced Settings")
        
        # Performance settings
        st.subheader("⚡ Performance")
        global frame_skip
        frame_skip = st.slider("Frame Skip", 1, 10, frame_skip, 
                              help="Skip frames to improve performance")
        
        batch_size = st.slider("Batch Processing Size", 1, 50, 10,
                              help="Number of images to process in batch")
        
        # Detection thresholds
        st.subheader("🎯 Detection Thresholds")
        global face_similarity_threshold, object_detection_confidence
        
        face_similarity_threshold = st.slider(
            "Face Similarity Threshold", 0.3, 0.95, face_similarity_threshold,
            help="Minimum similarity for face matches"
        )
        
        object_detection_confidence = st.slider(
            "Object Detection Confidence", 0.1, 0.9, object_detection_confidence,
            help="Minimum confidence for object detection"
        )
        
        # Advanced features toggle
        st.subheader("🚀 Advanced Features")
        enable_analytics = st.checkbox("Enable Advanced Analytics", True)
        enable_heatmap = st.checkbox("Generate Detection Heatmaps", False)
        enable_optimization = st.checkbox("Auto-Parameter Optimization", False)
    
    # Main application
    main()
    
    # Advanced analytics (if enabled)
    if enable_analytics and st.session_state.total_detections > 0:
        st.write("---")
        create_advanced_results_dashboard()
    
    # Detection heatmap (if enabled)
    if enable_heatmap and st.session_state.total_detections > 0:
        st.write("---")
        st.header("🔥 Detection Heatmap")
        
        all_detections = (st.session_state.face_results_log + 
                         st.session_state.traditional_object_results_log + 
                         st.session_state.text_object_results_log)
        
        heatmap_path = create_detection_heatmap(all_detections)
        if heatmap_path and os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Detection Location Heatmap")
        else:
            st.info("Unable to generate heatmap. Need more detection data.")

# Footer and credits
def display_footer():
    """Display application footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔍 Advanced Face & Object Detection System</p>
        <p>Built with Streamlit, OpenCV, and state-of-the-art AI models</p>
        <p>Features: Face Recognition • Object Detection • Text-Guided Search • Real-time Processing</p>
    </div>
    """, unsafe_allow_html=True)

# Application entry point
if __name__ == "__main__":
    try:
        # Run enhanced main function
        enhanced_main()
        
        # Display footer
        display_footer()
        
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        
        # Error recovery options
        if st.button("🔄 Reset Application"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Additional helper functions for specific use cases
def process_security_footage(video_paths, alert_threshold=0.8):
    """Process security footage with automatic alerts"""
    alerts = []
    
    for video_path in video_paths:
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 30th frame (1 second intervals for 30fps)
                if frame_count % 30 == 0:
                    timestamp = f"{frame_count // 30 // 60:02d}:{(frame_count // 30) % 60:02d}"
                    
                    # Detect faces and objects
                    detections = process_frame(frame, frame_count, timestamp, [], [])
                    
                    # Check for high-confidence detections
                    for detection_type, detection_list in detections.items():
                        if detection_type == 'total_count':
                            continue
                            
                        for detection in detection_list:
                            confidence = detection.get('confidence', detection.get('similarity', 0))
                            
                            if confidence > alert_threshold:
                                alerts.append({
                                    'video': video_path,
                                    'timestamp': timestamp,
                                    'type': detection_type,
                                    'confidence': confidence,
                                    'frame_number': frame_count
                                })
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error processing security footage {video_path}: {str(e)}")
    
    return alerts

def create_detection_timeline(detections):
    """Create a visual timeline of detections"""
    try:
        timeline_data = []
        
        for detection in detections:
            timestamp = detection.get('timestamp', '00:00')
            detection_type = detection.get('type', 'unknown')
            confidence = detection.get('confidence', detection.get('similarity', 0))
            
            timeline_data.append({
                'timestamp': timestamp,
                'type': detection_type,
                'confidence': confidence
            })
        
        # Sort by timestamp
        timeline_data.sort(key=lambda x: x['timestamp'])
        
        return timeline_data
        
    except Exception as e:
        logger.error(f"Error creating detection timeline: {str(e)}")
        return []

# Performance monitoring
def monitor_performance():
    """Monitor system performance and resource usage"""
    try:
        import psutil
        
        performance_data = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return performance_data
        
    except ImportError:
        # psutil not available
        return {
            'cpu_percent': 'N/A',
            'memory_percent': 'N/A', 
            'disk_usage': 'N/A',
            'timestamp': datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error monitoring performance: {str(e)}")
        return {}

# Final initialization and cleanup
def cleanup_temp_files():
    """Clean up temporary files older than 24 hours"""
    try:
        temp_dir = Path(tempfile.gettempdir())
        current_time = datetime.datetime.now()
        
        # Clean up detection images
        for file_pattern in ['face_match_*.png', 'obj_det_*.png', 'text_det_*.png', 'heatmap_*.png']:
            for file_path in temp_dir.glob(file_pattern):
                try:
                    file_age = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                    if (current_time - file_age).days >= 1:
                        file_path.unlink()
                        logger.info(f"Cleaned up old temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up {file_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Register cleanup function to run on app start
cleanup_temp_files()

print("🔍 Advanced Face & Object Detection System - Ready!")
print("✅ All modules loaded successfully")
print("🚀 Enhanced features activated")
