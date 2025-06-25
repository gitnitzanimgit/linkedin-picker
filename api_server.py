#!/usr/bin/env python3
"""
LinkedIn Photo Optimizer API Server (Step-by-Step)
Each pipeline step has its own API endpoint for full control and visibility.
"""

import os
import logging
import uuid
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import random
import signal
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
from PIL import Image as PILImage

# --- DETERMINISTIC EXECUTION SETUP ---
random_seed = int(os.getenv("RANDOM_SEED", "42"))
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = os.getenv("PYTHON_HASH_SEED", "42")

# Add PyTorch deterministic behavior
import torch
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- DISABLE SIGNAL HANDLING IN WORKER THREADS ---
signal.signal = lambda signum, handler: None

# --- Centralized Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
    datefmt='%H:%M:%S'
)

# Import services and models
from services.image_service import ImageService
from services.face_analysis_service import FaceAnalysisService
from pipeline.reference_image_validator import is_good_reference
from pipeline.detect_and_crop import detect_and_crop
from pipeline.background_replacer import replace_background
from pipeline.final_scorer import score_final_photos
from pipeline.top_image_enhancer import enhance_top_images
from models.image import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "data/temp_uploads")
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "data/api_results") 
ALLOWED_EXTENSIONS = set(os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,gif,bmp,webp").split(","))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100")) * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

# Initialize services
image_service = ImageService()
face_analysis_service = FaceAnalysisService()

logger = logging.getLogger(__name__)

# Session storage for pipeline state
sessions = {}


class PipelineSession:
    """Manages state for a single user's pipeline session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.reference_image: Optional[Image] = None
        self.batch_images: List[Image] = []
        self.cropped_images: List[Image] = []
        self.background_replaced_images: List[Image] = []
        self.scored_images: List[Any] = []  # ScoredImage objects
        self.enhanced_images: List[Any] = []  # Enhanced ScoredImage objects
        self.step = 0  # Track current pipeline step
        
    def clear(self):
        """Clear all images from memory."""
        self.reference_image = None
        self.batch_images.clear()
        self.cropped_images.clear()
        self.background_replaced_images.clear()
        self.scored_images.clear()
        self.enhanced_images.clear()
        self.step = 0


def get_or_create_session(session_id: str = None) -> PipelineSession:
    """Get existing session or create new one."""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = PipelineSession(session_id)
    
    return sessions[session_id]


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image: Image) -> str:
    """Convert Image object to base64 string for JSON response."""
    # Convert numpy array to PIL Image
    pil_image = PILImage.fromarray(image.pixels)
    
    # Save to bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=int(os.getenv("JPEG_QUALITY", "95")))
    buffer.seek(0)
    
    # Encode to base64
    img_bytes = buffer.getvalue()
    base64_string = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_string}"


def save_image_for_serving(image: Image, filename: str) -> str:
    """Save image to results folder and return URL path."""
    results_path = Path(RESULTS_FOLDER) / filename
    
    # Convert to PIL and save
    pil_image = PILImage.fromarray(image.pixels)
    pil_image.save(results_path, 'JPEG', quality=int(os.getenv("JPEG_QUALITY", "95")))
    
    return f"/api/image/{filename}"


@app.route('/api/validate-reference', methods=['POST'])
def validate_reference():
    """Validate reference image for face detection."""
    try:
        if 'reference_image' not in request.files:
            return jsonify({'valid': False, 'message': 'No reference image provided'}), 400
        
        file = request.files['reference_image']
        if file.filename == '':
            return jsonify({'valid': False, 'message': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = Path(UPLOAD_FOLDER) / f"ref_temp_{uuid.uuid4().hex}_{filename}"
        file.save(str(temp_path))
        
        try:
            # Load and validate image
            image = image_service.load(str(temp_path))
            logger.info(f"Reference image loaded: {image.pixels.shape}")
            
            # Use comprehensive reference image validator
            is_valid, details = is_good_reference(image, face_analysis_service, image_service)
            
            # Convert numpy types to native Python types for JSON serialization
            if details:
                json_details = {}
                for key, value in details.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        json_details[key] = value.item()
                    else:
                        json_details[key] = value
            else:
                json_details = details
            
            if is_valid:
                return jsonify({
                    'valid': True,
                    'message': 'Great reference image! All quality checks passed.',
                    'details': json_details
                })
            else:
                return jsonify({
                    'valid': False,
                    'message': f"Reference image not suitable: {json_details.get('reason', 'quality check failed')}",
                    'details': json_details
                })
                
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        logger.error(f"Reference validation error: {e}")
        return jsonify({'valid': False, 'message': 'Error processing reference image'}), 500


@app.route('/api/load-images', methods=['POST'])
def load_images():
    """Load reference and batch images into session."""
    try:
        # Get or create session
        session_id = request.form.get('session_id')
        session = get_or_create_session(session_id)
        session.clear()  # Start fresh
        
        # Validate request
        if 'reference_image' not in request.files:
            return jsonify({'success': False, 'message': 'No reference image provided'}), 400
        
        reference_file = request.files['reference_image']
        batch_files = []
        
        # Collect batch images
        for key in request.files:
            if key.startswith('image_'):
                batch_files.append(request.files[key])
        
        if not batch_files:
            return jsonify({'success': False, 'message': 'No batch images provided'}), 400
        
        logger.info(f"Loading {len(batch_files)} batch images with reference for session {session.session_id}")
        
        # Save and load reference image
        ref_filename = secure_filename(reference_file.filename)
        ref_path = Path(UPLOAD_FOLDER) / f"ref_{session.session_id}_{ref_filename}"
        reference_file.save(str(ref_path))
        session.reference_image = image_service.load(str(ref_path))
        
        # Save and load batch images
        for i, batch_file in enumerate(batch_files):
            if batch_file and allowed_file(batch_file.filename):
                filename = secure_filename(batch_file.filename)
                batch_path = Path(UPLOAD_FOLDER) / f"batch_{session.session_id}_{i:03d}_{filename}"
                batch_file.save(str(batch_path))
                
                # Load image and immediately preserve original state
                batch_image = image_service.load(str(batch_path))
                image_service.preserve_original_state(batch_image)  # Preserve TRULY original pixels
                session.batch_images.append(batch_image)
                
                logger.info(f"Loaded batch image {i+1}: {batch_image.pixels.shape}")
        
        session.step = 1
        
        return jsonify({
            'success': True,
            'session_id': session.session_id,
            'reference_loaded': True,
            'batch_count': len(session.batch_images),
            'message': f'Loaded reference and {len(session.batch_images)} batch images'
        })
        
    except Exception as e:
        logger.error(f"Image loading error: {e}")
        return jsonify({'success': False, 'message': f'Error loading images: {str(e)}'}), 500


@app.route('/api/detect-and-crop', methods=['POST'])
def detect_and_crop_step():
    """Detect faces and crop images."""
    try:
        session_id = request.json.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'message': 'Invalid session'}), 400
        
        session = sessions[session_id]
        if session.step != 1:
            return jsonify({'success': False, 'message': f'Invalid step. Expected step 1, got {session.step}'}), 400
        
        logger.info(f"Running face detection and cropping for session {session_id}")
        
        # Run detect and crop pipeline
        session.cropped_images = detect_and_crop(
            ref_img=session.reference_image,
            gallery=session.batch_images
        )
        
        # Original state already preserved during load step
        
        session.step = 2
        
        logger.info(f"Face detection complete. {len(session.cropped_images)} images with matching faces")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'detected_count': len(session.cropped_images),
            'message': f'Detected and cropped {len(session.cropped_images)} images with matching faces'
        })
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return jsonify({'success': False, 'message': f'Error in face detection: {str(e)}'}), 500


@app.route('/api/replace-backgrounds', methods=['POST'])
def replace_backgrounds_step():
    """Replace backgrounds in cropped images."""
    try:
        session_id = request.json.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'message': 'Invalid session'}), 400
        
        session = sessions[session_id]
        if session.step != 2:
            return jsonify({'success': False, 'message': f'Invalid step. Expected step 2, got {session.step}'}), 400
        
        logger.info(f"Running background replacement for session {session_id}")
        
        # Run background replacement
        session.background_replaced_images = replace_background(session.cropped_images)
        
        session.step = 3
        
        logger.info(f"Background replacement complete. {len(session.background_replaced_images)} images processed")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'processed_count': len(session.background_replaced_images),
            'message': f'Replaced backgrounds in {len(session.background_replaced_images)} images'
        })
        
    except Exception as e:
        logger.error(f"Background replacement error: {e}")
        return jsonify({'success': False, 'message': f'Error in background replacement: {str(e)}'}), 500


@app.route('/api/score-and-filter', methods=['POST'])
def score_and_filter_step():
    """Score images and filter to top 5."""
    try:
        session_id = request.json.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'message': 'Invalid session'}), 400
        
        session = sessions[session_id]
        if session.step != 3:
            return jsonify({'success': False, 'message': f'Invalid step. Expected step 3, got {session.step}'}), 400
        
        logger.info(f"Running scoring and filtering for session {session_id}")
        
        # Run scoring and filtering (returns top 5 ScoredImage objects)
        session.scored_images = score_final_photos(session.background_replaced_images)
        
        session.step = 4
        
        # Create response with ORIGINAL image data for frontend display
        results = []
        for i, scored_img in enumerate(session.scored_images):
            # Save ORIGINAL image (pre-crop, pre-background) for display
            original_filename = f"original_display_{session_id}_{i}.jpg"
            original_url = save_image_for_serving(
                type('TempImage', (), {'pixels': scored_img.image.original_pixels})(), 
                original_filename
            )
            
            results.append({
                'index': i,
                'final_score': round(scored_img.final_score, 2),
                'linkedin_score': round(scored_img.linkedin_score, 1),
                'attire_score': round(scored_img.attire_score, 1),
                'neutrality_score': round(scored_img.face_neutrality_score, 1),
                'image_url': original_url,  # Show original for display
                'filename': original_filename
            })
        
        logger.info(f"Scoring complete. Top {len(session.scored_images)} images selected")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'top_images': results,
            'message': f'Selected top {len(session.scored_images)} highest-scoring images'
        })
        
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        return jsonify({'success': False, 'message': f'Error in scoring: {str(e)}'}), 500


@app.route('/api/enhance-images', methods=['POST'])
def enhance_images_step():
    """Enhance the top images and save final results."""
    try:
        session_id = request.json.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'message': 'Invalid session'}), 400
        
        session = sessions[session_id]
        if session.step != 4:
            return jsonify({'success': False, 'message': f'Invalid step. Expected step 4, got {session.step}'}), 400
        
        logger.info(f"Running image enhancement for session {session_id}")
        
        # Run enhancement
        session.enhanced_images = enhance_top_images(session.scored_images)
        
        # Save enhanced images
        image_service.save_scored_gallery(session.enhanced_images)
        
        session.step = 5  # Complete
        
        # Create response with before/after comparisons
        results = []
        for i, enhanced_img in enumerate(session.enhanced_images):
            # Save original (before enhancement)
            original_filename = f"original_{session_id}_{i}.jpg"
            original_url = save_image_for_serving(
                type('TempImage', (), {'pixels': enhanced_img.image.original_pixels})(), 
                original_filename
            )
            
            # Save enhanced (after enhancement)
            enhanced_filename = f"enhanced_{session_id}_{i}.jpg"
            enhanced_url = save_image_for_serving(enhanced_img.image, enhanced_filename)
            
            results.append({
                'index': i,
                'final_score': round(enhanced_img.final_score, 2),
                'linkedin_score': round(enhanced_img.linkedin_score, 1),
                'attire_score': round(enhanced_img.attire_score, 1),
                'neutrality_score': round(enhanced_img.face_neutrality_score, 1),
                'original_url': original_url,
                'enhanced_url': enhanced_url,
                'original_filename': original_filename,
                'enhanced_filename': enhanced_filename
            })
        
        logger.info(f"Enhancement complete. {len(session.enhanced_images)} images enhanced and saved")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'enhanced_images': results,
            'message': f'Enhanced and saved {len(session.enhanced_images)} images'
        })
        
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return jsonify({'success': False, 'message': f'Error in enhancement: {str(e)}'}), 500


@app.route('/api/image/<filename>')
def serve_image(filename):
    """Serve processed images."""
    try:
        image_path = Path(RESULTS_FOLDER) / filename
        if image_path.exists():
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({'error': 'Error serving image'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy', 
        'message': 'LinkedIn Photo Optimizer API (Step-by-Step) is running',
        'active_sessions': len(sessions)
    })


@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """Clear a session and free memory."""
    try:
        session_id = request.json.get('session_id')
        if session_id and session_id in sessions:
            sessions[session_id].clear()
            del sessions[session_id]
            return jsonify({'success': True, 'message': 'Session cleared'})
        else:
            return jsonify({'success': False, 'message': 'Session not found'})
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({'success': False, 'message': 'Error clearing session'}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413


@app.errorhandler(400)
def bad_request(e):
    """Handle bad request error."""
    return jsonify({'error': 'Bad request'}), 400


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting LinkedIn Photo Optimizer API Server (Step-by-Step)...")
    print(f"📁 Upload directory: {UPLOAD_FOLDER}")
    print(f"📁 Results directory: {RESULTS_FOLDER}")
    print(f"🔧 Max upload size: {MAX_CONTENT_LENGTH // (1024*1024)}MB")
    print("🌐 CORS enabled for frontend communication")
    print("📋 Pipeline Steps:")
    print("   1. /api/validate-reference")
    print("   2. /api/load-images")  
    print("   3. /api/detect-and-crop")
    print("   4. /api/replace-backgrounds")
    print("   5. /api/score-and-filter")
    print("   6. /api/enhance-images")
    print("="*60)
    
    # Run development server (threading disabled for testing)
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("API_SERVER_PORT", "5002")), threaded=False)