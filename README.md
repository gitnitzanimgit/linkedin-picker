# LinkedInPicker - AI-Powered Professional Photo Enhancement Pipeline

An intelligent photo processing pipeline that automatically transforms casual photos into LinkedIn-ready professional headshots using computer vision and machine learning.

## 🎬 Demo

[![Demo Video](https://img.youtube.com/vi/AVd5tHkiXtU/0.jpg)](https://youtu.be/AVd5tHkiXtU)

## 📋 Overview

LinkedInPicker processes photos through a sophisticated multi-stage pipeline that:

0. **Validates reference image quality** to ensure reliable identity matching
1. **Detects and crops faces** using InsightFace with optimal LinkedIn-style framing
2. **Separates background from subject** using MediaPipe selfie segmentation
3. **Applies professional background blur** with Laplacian variance-based downsampling
4. **Scores photo quality** using a custom fine-tuned ResNet18 model trained on FFHQ dataset
5. **Evaluates attire appropriateness** using CLIP-based semantic analysis
6. **Enhances lighting and color** through CLIP-guided gradient descent optimization
7. **Provides both web API and batch processing** interfaces

## ⚙️ Technical Implementation

### 🔍 Reference Image Validation
- **Quality-gated embedding extraction**: Validates reference image meets strict criteria (single face, detection confidence, pose alignment, sharpness, lighting) before generating face embeddings
- **Prevents cascade failures**: Poor reference images lead to unreliable identity matching, so validation acts as a quality gate protecting the entire pipeline
- **Six-point validation**: Face count → detection confidence → size ratio → pose angles → Laplacian sharpness → brightness levels

### 👤 Face Detection & Analysis
- **InsightFace (buffalo_l model)** for robust face detection and landmark extraction
- **Identity matching** using validated reference embeddings with configurable similarity thresholds
- Validates face pose, age range (18-65), and facial quality metrics
- Supports multi-face images with automatic largest face selection

### 🎨 Background Processing
- **MediaPipe Selfie Segmentation** for precise person/background separation
- **Professional blur effect**: Images are downscaled, blurred, then upscaled to create a creamy, DSLR-like background blur
- Gaussian edge feathering for smooth alpha blending

### 🎯 LinkedIn Quality Model
- **Custom ResNet18** fine-tuned on preprocessed FFHQ dataset
- **Data augmentation pipeline**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomErasing
- **Business-optimized loss function**: Weighted Binary Cross-Entropy with 3:1 False Positive to False Negative cost ratio

```
Cost Function: L = 3×FP + 1×FN
```

This design philosophy prioritizes precision over recall - it's better to reject a good photo than accept a poor one for professional use.

### ⚡ CLIP-Guided Enhancement
- **Optimization target**: "Subject studio-style lit, face clear and crisp, Overall image brightness is high"
- **Learnable parameters**: Brightness, contrast, and gamma adjustments
- **Optimization**: SGD with momentum, adaptive learning rate scheduling, early stopping
- **Parameter space**: Uses hyperbolic tangent mapping to constrain adjustments within realistic bounds

### 👔 Attire & Expression Analysis
- **CLIP-based semantic evaluation** for professional attire appropriateness
- **Facial neutrality scoring** using contrastive prompts (neutral vs. non-neutral expressions)
- Configurable scoring thresholds for different quality standards

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
For processing large batches of images offline:

```bash
# Place reference image in data/target_image/very_good_image.jpeg
# Place batch images in data/image_batch/
python scripts/batch_process.py
```

Results will be saved to `data/enhanced_gallery/`

## 🔬 Algorithm Details

### Quality Scoring Pipeline
1. **Face validation**: Pose angle, brightness, sharpness checks
2. **LinkedIn model inference**: Custom ResNet18 with 0-100 quality score
3. **Attire assessment**: CLIP semantic similarity scoring
4. **Expression neutrality**: Contrastive prompt evaluation
5. **Composite scoring**: Weighted combination with configurable thresholds

### Enhancement Process
1. **Parameter initialization**: Brightness, contrast, gamma at neutral (0.0)
2. **CLIP encoding**: Target prompt and current image
3. **Gradient descent**: SGD optimization of visual similarity
4. **Learning rate scheduling**: Adaptive reduction with patience
5. **Early stopping**: Convergence detection with improvement threshold

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- InsightFace team for robust face detection
- OpenAI for CLIP models
- MediaPipe team for segmentation models
- FFHQ dataset creators for training data