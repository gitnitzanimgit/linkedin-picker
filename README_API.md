# LinkedIn Photo Optimizer - API Integration

This document describes how to run the LinkedIn Photo Optimizer with the new web API and frontend.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements_minimal.txt
   ```

2. **Start the API Server**
   ```bash
   python api_server.py
   ```
   
   The API server will start on `http://localhost:5000`

3. **Open the Frontend**
   ```bash
   cd frontend
   python -m http.server 8080
   ```
   
   Open `http://localhost:8080` in your browser

## API Endpoints

### `/api/validate-reference` (POST)
Validates a reference image for face detection.

**Request:**
- Form data with `reference_image` file

**Response:**
```json
{
  "valid": true,
  "message": "Great reference image! Face detected successfully.",
  "face_count": 1
}
```

### `/api/process-images` (POST)
Processes batch images through the complete LinkedIn optimization pipeline.

**Request:**
- Form data with:
  - `reference_image`: The validated reference image
  - `image_0`, `image_1`, etc.: Batch images to process

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "index": 0,
      "filename": "photo1.jpg",
      "final_score": 8.7,
      "linkedin_score": 9.2,
      "attire_score": 8.5,
      "neutrality_score": 8.4,
      "face_similarity": 0.85
    }
  ],
  "message": "Successfully processed 5 images"
}
```

### `/api/health` (GET)
Health check endpoint.

## Usage Flow

1. **Upload Reference Image**: User uploads a clear photo of themselves
2. **Validation**: API validates the reference image (single face, good quality)
3. **Batch Upload**: User uploads multiple photos to optimize
4. **Processing**: API runs the complete pipeline:
   - Face detection and matching
   - Cropping to square format
   - Background replacement
   - LinkedIn scoring (photo quality, attire, facial expression)
5. **Results**: API returns top 5 scored images
6. **Enhancement**: Frontend shows enhancement animations (currently mock)

## Error Handling

The API provides detailed error messages for:
- Invalid file types
- No faces detected
- Multiple faces in reference
- Face too small
- No matching faces in batch
- Network errors

## Technical Details

- **Max File Size**: 16MB per image
- **Supported Formats**: PNG, JPG, JPEG, GIF, BMP, WEBP
- **Face Similarity Threshold**: 0.3 (30% similarity required)
- **Top Results**: Maximum 5 images returned
- **CORS**: Enabled for frontend communication

## Development Notes

The current implementation:
- ✅ Real face detection and validation
- ✅ Complete LinkedIn scoring pipeline
- ✅ Face matching between reference and batch images
- ⚠️ Enhancement step uses mock data (enhancement pipeline exists but not connected)
- ⚠️ No image serving endpoints (images served from original files)

Next steps would be to:
1. Connect the enhancement pipeline to return actual enhanced images
2. Add image serving endpoints to serve processed results
3. Add progress tracking for long-running operations
4. Add image caching for better performance