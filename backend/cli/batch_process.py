import os
import logging
import numpy as np
import random
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# --- DETERMINISTIC EXECUTION SETUP ---
# Set seeds for reproducible results
random_seed = int(os.getenv("RANDOM_SEED", "42"))
np.random.seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = os.getenv("PYTHON_HASH_SEED", "42")

# --- Centralized Logging Configuration ---
# This should be the first thing to run to ensure all modules use the same config.
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce noise, or DEBUG for full detail
    format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
    datefmt='%H:%M:%S'
)
# ───────────────────────────────────────────

import sys
from pathlib import Path

from ..pipeline.detect_and_crop import detect_and_crop
from ..pipeline.background_replacer import replace_background
from ..pipeline.final_scorer import score_final_photos
from ..pipeline.top_image_enhancer import enhance_top_images, log_final_results
from ..services.image_service import ImageService

image_service = ImageService()

# Load Reference Image
ref_img = image_service.load("data/target_image/very_good_image.jpeg")

# Load gallery
gallery = image_service.stream_gallery("data/image_batch")

print(f"\nStarting efficient in-memory pipeline...")

# Reset gallery iterator since we consumed it above for counting
gallery = image_service.stream_gallery("data/image_batch")

# Step 1: Detect and crop (in-memory processing)
print(f"\nStep 1: Face detection and cropping...")
detected_cropped_gallery = detect_and_crop(gallery=gallery, ref_img=ref_img)
print(f"Cropped {len(detected_cropped_gallery)} images with matching faces")

# Preserve original state before pipeline modifications
for img in detected_cropped_gallery:
    image_service.preserve_original_state(img)

# Step 2: Replace backgrounds (in-memory processing)
print(f"\nStep 2: Background replacement...")
replaced_background_gallery = replace_background(detected_cropped_gallery)
print(f"Processed {len(replaced_background_gallery)} images with new backgrounds")

# Step 3: Score and filter (returns top 5 only)
print(f"\nStep 3: Quality scoring and filtering...")
final_scored_gallery = score_final_photos(replaced_background_gallery)
print(f"Selected top {len(final_scored_gallery)} highest-quality images")

# Step 4: Final enhancement (in-memory processing)
print(f"\nStep 4: Final image enhancement...")
enhanced_final_gallery = enhance_top_images(final_scored_gallery)

# Step 5: Only save the final enhanced results
print(f"\nSaving final enhanced images...")
image_service.save_scored_gallery(enhanced_final_gallery)

# Log the final results
log_final_results(enhanced_final_gallery)

print(f"\nPipeline complete!")
print(f"Efficiency: Processed entire pipeline in-memory with only 1 save operation!")
print(f"Final images available in enhanced_gallery/ directory")
print(f"Original images preserved for comparison in each Image object")
