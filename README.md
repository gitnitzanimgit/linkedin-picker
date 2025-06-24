# LinkedIn Photo Optimizer

An AI-powered system that automatically selects and enhances the best LinkedIn profile photos from a batch of images. Uses advanced computer vision, face recognition, and CLIP-guided enhancement to create professional headshots.

## Features

- **Reference-based face matching** - Upload a reference photo to find similar faces
- **Automatic face detection and cropping** - Smart cropping to professional headshot format
- **AI background replacement** - Clean, professional backgrounds
- **LinkedIn-optimized scoring** - AI models trained to score photo professionalism
- **CLIP-guided enhancement** - Automatic brightness/contrast/gamma optimization
- **Web interface** - React frontend with step-by-step processing
- **REST API** - Full API access for programmatic use

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/LinkedInPicker
cd LinkedInPicker
pip install -r requirements_minimal.txt
```

### 2. Download Models

Download the required model files and place them in the project root:

- **LinkedIn Classifiers**:
  - `linkedin_efficientb0_cost_min.pth` (recommended)
  - `linkedin_resnet18_cost_min.pth` (faster alternative)

### 3. Run the Application

**Web Interface:**
```bash
python api_server.py
```
Server runs on `http://localhost:5002`

**CLI Mode:**
```bash
python main.py
```

**Single Image Enhancement:**
```bash
python enhance_single.py path/to/your/image.jpg
```

## API Documentation

The step-by-step API provides full control over each processing step:

1. **POST /api/validate-reference** - Validate reference image
2. **POST /api/load-images** - Upload batch images  
3. **POST /api/detect-and-crop** - Face detection and cropping
4. **POST /api/replace-backgrounds** - Background replacement
5. **POST /api/score-and-filter** - Score and select top 5 images
6. **POST /api/enhance-images** - Final enhancement

### Example Usage

```python
import requests

# 1. Start session with reference image
files = {'reference_image': open('reference.jpg', 'rb')}
response = requests.post('http://localhost:5002/api/validate-reference', files=files)
session_id = response.json()['session_id']

# 2. Upload batch images
files = [('batch_images', open(f'photo{i}.jpg', 'rb')) for i in range(10)]
requests.post('http://localhost:5002/api/load-images', 
              files=files, 
              data={'session_id': session_id})

# 3. Process through pipeline
for endpoint in ['detect-and-crop', 'replace-backgrounds', 'score-and-filter', 'enhance-images']:
    requests.post(f'http://localhost:5002/api/{endpoint}', 
                  json={'session_id': session_id})
```

## Architecture

Clean, modular design with separation of concerns:

```
├── models/           # Data models (Image, Face, ScoredImage, etc.)
├── services/         # Business logic layer
├── repositories/     # Data access layer  
├── pipeline/         # Processing pipelines
├── frontend/         # React web interface
└── api_server.py     # Flask REST API server
```

## Development

### Project Structure

```
LinkedInPicker/
├── models/
│   ├── image.py              # Core image data model
│   ├── face.py              # Face detection/recognition
│   ├── scored_image.py      # Scored image container
│   └── clip_model.py        # CLIP model wrapper
├── services/
│   ├── image_service.py     # Image operations
│   ├── face_service.py      # Face processing
│   ├── segmentation_service.py  # Background segmentation
│   └── linkedin_photo_service.py # LinkedIn scoring
├── pipeline/
│   ├── detect_and_crop.py   # Face detection pipeline
│   ├── background_replacer.py   # Background replacement
│   ├── final_scorer.py      # Scoring pipeline
│   └── top_image_enhancer.py    # Enhancement pipeline
├── repositories/
│   ├── image_repository.py  # Image data access
│   └── face_repository.py   # Face data access
└── frontend/                # React web interface
```

### Configuration

Key parameters can be tuned in the service classes:
- Face similarity threshold: `0.3` (in face matching)
- LinkedIn score weights: Configurable in scoring service
- Enhancement iterations: `200` steps (in enhancement service)

### Hardware Requirements

- **CPU**: Modern multi-core processor
- **RAM**: 8GB+ recommended  
- **GPU**: Optional (CUDA/MPS supported for faster processing)
- **Storage**: 2GB+ for models and temporary files

## Troubleshooting

### Common Issues

**Dark enhanced images:**
- Threading is disabled by default for consistent results
- PyTorch deterministic settings ensure reproducible enhancement

**Memory issues:**
- Reduce batch size
- Process images sequentially

**Model loading errors:**
- Verify model files are downloaded and in correct location
- Check file permissions

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with PyTorch, OpenCV, and CLIP
- Uses MediaPipe for face detection
- Background segmentation via custom models
- React frontend for user interface
