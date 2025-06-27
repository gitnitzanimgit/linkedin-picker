# LinkedInPicker - AI-Powered Professional Photo Enhancement Pipeline

An intelligent photo finder and optimizer that automatically identifies, ranks, and enhances the best photos of a person from a gallery for professional LinkedIn profiles using computer vision and machine learning.

## 🎬 Demo

[![Demo Video](https://img.youtube.com/vi/AVd5tHkiXtU/0.jpg)](https://youtu.be/AVd5tHkiXtU)

## 📋 Overview

LinkedInPicker finds and optimizes photos through a sophisticated multi-stage pipeline:

1. **Validates reference image quality** to ensure reliable identity matching
2. **Identifies target person** in photo galleries using face embedding similarity
3. **Detects and crops faces** using InsightFace with optimal LinkedIn-style framing
4. **Separates background and applies professional blur** using MediaPipe segmentation and Laplacian variance-based downsampling
5. **Enhances lighting and color** through CLIP-based gradient descent optimization
6. **Ranks photos by LinkedIn suitability** using quality scoring, attire analysis, and expression evaluation

## ⚙️ Technical Implementation

### 🔍 Reference Image Validation
- **Quality-gated embedding extraction**: Validates reference image meets strict criteria (single face, detection confidence, pose alignment, sharpness, lighting) before generating face embeddings
- **Prevents cascade failures**: Poor reference images lead to unreliable identity matching, so validation acts as a quality gate protecting the entire pipeline
- **Six-point validation**: Face count → detection confidence → size ratio → pose angles → Laplacian sharpness → brightness levels

### 👤 Face Detection & Analysis
- **InsightFace (buffalo_l model)** for robust face detection and landmark extraction
- **Identity matching** using validated reference embeddings with configurable similarity thresholds
- Validates face pose and facial quality metrics
- Supports multi-face images with automatic target person identification

### 🎨 Background Processing
- **MediaPipe Selfie Segmentation** for precise person/background separation
- **Professional blur effect**: Images are downscaled, blurred, then upscaled to create a creamy, DSLR-like background blur
- Gaussian edge feathering for smooth alpha blending

### 🎯 Professional Suitability Assessment
- **Custom ResNet18** fine-tuned on preprocessed FFHQ dataset to assess LinkedIn suitability
- **Data augmentation pipeline**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomErasing
- **Business-optimized loss function**: Weighted Binary Cross-Entropy with 3:1 False Positive to False Negative cost ratio
- **Expression analysis**: CLIP-based facial neutrality scoring using contrastive prompts (neutral vs. non-neutral expressions)

```
Cost Function: L = 3×FP + 1×FN
```

This design philosophy prioritizes precision over recall - it's better to reject a good photo than accept a poor one for professional use.

### 👔 Attire Analysis
- **CLIP-based semantic evaluation** for professional attire appropriateness

### ⚡ CLIP-Based Gradient Descent Image Optimization
- **Optimization target**: Professional studio-style lighting and clarity
- **Learnable parameters**: Brightness, contrast, and gamma adjustments
- **Optimization**: SGD with momentum, adaptive learning rate scheduling, early stopping
- **Parameter space**: Uses hyperbolic tangent mapping to constrain adjustments within realistic bounds


## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (optional, will fallback to CPU)
- 8GB+ RAM recommended

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Model Setup
All required models are included or downloaded automatically:

- **LinkedIn Quality Model**: Pre-trained `linkedin_resnet18_cost_min.pth` (included in repository)
- **InsightFace Models**: Downloaded automatically on first use
- **MediaPipe Segmentation**: Downloaded automatically on first use
- **CLIP Models**: Downloaded automatically on first use

## 💻 Usage

### 🌐 Web Interface (Recommended)

1. **Start the API server**:
```bash
python api_server.py
```

2. **Start the frontend** (in a separate terminal):
```bash
cd frontend
python -m http.server 8080
```

3. **Access the application**:
Open your browser to `http://localhost:8080`

### 📦 Batch Processing (Optional)
For finding and enhancing the best photos of a target person from large photo galleries:

```bash
# Place reference image of target person in data/target_image/very_good_image.jpeg
# Place photo gallery (containing the target person) in data/image_batch/
python scripts/batch_process.py
```

The system will identify all photos containing the target person, rank them by LinkedIn suitability, enhance the best candidates, and save results to `data/enhanced_gallery/`

## 🔬 Algorithm Details

### Quality Scoring Pipeline
1. **Face validation**: Pose angle, brightness, sharpness checks
2. **LinkedIn model inference**: Custom ResNet18 with 0-100 quality score
3. **Attire assessment**: CLIP semantic similarity scoring
4. **Expression neutrality**: Contrastive prompt evaluation
5. **Composite scoring**: Weighted combination of quality, attire, and expression metrics

### Enhancement Process
1. **Parameter initialization**: Brightness, contrast, gamma at neutral (0.0)
2. **CLIP encoding**: Target prompt and current image
3. **Gradient descent**: GD optimization of semantic similarity
4. **Learning rate scheduling**: Adaptive reduction with patience
5. **Early stopping**: Stops optimization when no further improvement is detected

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- InsightFace team for robust face detection
- OpenAI for CLIP models
- MediaPipe team for segmentation models
- FFHQ dataset creators for training data