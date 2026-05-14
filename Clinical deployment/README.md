# Spinal Lesion Detection System

![Python](https://img.shields.io/badge/python-v3.9+-blue)
![Flask](https://img.shields.io/badge/flask-3.0.0-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-red)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-production-success)

> **A production-ready medical AI web application for automated spinal lesion detection and classification from DICOM radiographs using deep learning ensemble models.**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Deployment](#deployment)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development](#development)
- [Security](#security)
- [Medical Disclaimer](#medical-disclaimer)
- [Contributing](#contributing)
- [License](#license)

## Overview

This application implements a state-of-the-art computer vision system for automated analysis of spine radiographs. The system leverages an ensemble of deep convolutional neural networks combined with object detection to provide both classification (Normal/Abnormal) and lesion localization capabilities.

### Key Capabilities
- **Binary Classification**: Automated categorization of spine X-rays as Normal or Abnormal
- **Lesion Detection**: Precise localization of spinal abnormalities with bounding boxes
- **Confidence Scoring**: Probabilistic outputs for clinical decision support
- **DICOM Compatibility**: Native support for medical imaging standards

### Scientific Foundation
Built upon the VinDr-SpineXR dataset, a comprehensive annotated medical image repository for spinal lesion analysis. The ensemble architecture demonstrates superior performance compared to baseline methods published in peer-reviewed literature.

**Key Performance Indicators:**
- Classification AUROC: **91.03%**
- F1-Score: **83.09%**
- Detection mAP50-95: **18.99%**

## Features

### Core Functionality
- ✅ Multi-model ensemble learning (DenseNet121, ResNet50, EfficientNetV2-S)
- ✅ Real-time object detection using YOLO11
- ✅ DICOM format validation and preprocessing
- ✅ Automated photometric interpretation handling
- ✅ RESTful API for programmatic access
- ✅ Interactive web interface with drag-and-drop upload
- ✅ Comprehensive metadata extraction and display

### Technical Features
- Weighted ensemble predictions for improved accuracy
- Automatic MONOCHROME1 inversion handling
- GPU-accelerated inference (CUDA compatible)
- Stateless architecture for horizontal scalability
- Containerized deployment with Docker
- Production-ready logging and error handling

## System Architecture

### High-Level Pipeline

```
         ┌─────────────┐
         │   Client    │
         │  (Browser)  │
         └──────┬──────┘
                │ HTTP POST /upload
                ▼
┌─────────────────────────────────┐
│       Flask Application         │
│  ┌──────────────────────────┐   │
│  │  DICOM Validation Layer  │   │
│  └───────────┬──────────────┘   │
│              │                  │
│  ┌───────────▼──────────────┐   │
│  │  Image Preprocessing     │   │
│  │  - Normalization         │   │
│  │  - RGB Conversion        │   │
│  │  - Resize (224×224)      │   │
│  └───────────┬──────────────┘   │
│              │                  │
│      ┌───────┴────────┐         │
│      ▼                ▼         │
│  ┌─────────┐    ┌──────────┐    │
│  │Ensemble │    │ YOLO11   │    │
│  │ Models  │    │ Detector │    │
│  │(3 CNNs) │    │          │    │
│  └────┬────┘    └─────┬────┘    │
│       │               │         │
│  ┌────▼───────────────▼─────┐   │
│  │  Result Aggregation      │   │
│  │  & Response Generation   │   │
│  └──────────────────────────┘   │
└────────────┬────────────────────┘
             │ JSON Response
             ▼
      ┌──────────────┐
      │   Results    │
      │  Rendering   │
      └──────────────┘
```

### Model Architecture

**Classification Ensemble (Weighted Voting)**
| Model | Architecture | Parameters | Weight | Purpose |
|-------|-------------|------------|---------|---------|
| DenseNet121 | Dense Connectivity | 7.98M | 42% | Feature reuse & gradient flow |
| ResNet50 | Residual Learning | 25.6M | 26% | Deep network optimization |
| EfficientNetV2-S | Compound Scaling | 21.5M | 32% | Efficiency & accuracy balance |

**Detection Model**
- **Architecture**: YOLO11 (You Only Look Once v11)
- **Input Size**: 640×640 pixels
- **Output**: Bounding boxes with class probabilities
- **Training**: 35 epochs on VinDr-SpineXR annotations

### Technology Stack

**Backend**
- Python 3.9+
- Flask 3.0.0 (Web framework)
- PyTorch 2.1.0 (Deep learning)
- Ultralytics (YOLO implementation)
- PyDICOM (Medical imaging I/O)
- Gunicorn (WSGI server)

**Frontend**
- HTML5/CSS3
- JavaScript (ES6+)
- Responsive design

**Infrastructure**
- Docker containerization
- Multi-platform deployment support
- CI/CD ready

## Performance Metrics

### Classification Performance (Test Set)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **AUROC** | 91.03% | Area Under ROC Curve |
| **F1-Score** | 83.09% | Harmonic mean of precision/recall |
| **Sensitivity** | 84.91% | True positive rate (recall) |
| **Specificity** | 81.68% | True negative rate |
| **Optimal Threshold** | 0.449 | Probability cutoff |

### Detection Performance

| Metric | Value | Configuration |
|--------|-------|--------------|
| **mAP50-95** | 18.99% | Mean Average Precision |
| **Training Epochs** | 35 | Convergence point |
| **Batch Size** | 12 | Training configuration |
| **Image Resolution** | 640×640 | Input dimensions |

### Computational Requirements

**Inference Time** (GPU - NVIDIA T4)
- Classification: ~100ms per image
- Detection: ~150ms per image
- Total pipeline: ~250ms per image

**Memory Requirements**
- Model weights: ~177 MB total
- Runtime memory: ~2 GB (CPU mode)
- GPU memory: ~4 GB (CUDA mode)

## Installation

### Prerequisites

**System Requirements**
- Operating System: Linux, macOS, or Windows
- Python: 3.8 or higher
- RAM: Minimum 4 GB (8 GB recommended)
- GPU: CUDA-compatible GPU (optional, for acceleration)

**Required Tools**
- Git
- pip (Python package manager)
- virtualenv (recommended)

### Local Development Setup

**1. Clone Repository**
```bash
git clone https://github.com/pronad1/Deploy-Model.git
cd Deploy-Model
```

**2. Create Virtual Environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Verify Model Files**

Ensure the following model checkpoints exist:
```
ensemble output/
├── densenet121_balanced/model_best.pth      (80 MB)
├── resnet50_optimized/model_best.pth        (26 MB)
└── tf_efficientnetv2_s_optimized/model_best.pth  (23 MB)

detection output/
└── yolo11/weights/best.pt                   (48 MB)
```

**5. Run Application**
```bash
python app.py
```

**6. Access Application**
```
http://localhost:5000
```

### Docker Installation

**Build Image**
```bash
docker build -t spine-detection:latest .
```

**Run Container**
```bash
docker run -d \
  -p 5000:5000 \
  --name spine-detector \
  spine-detection:latest
```

**With GPU Support**
```bash
docker run -d \
  --gpus all \
  -p 5000:5000 \
  --name spine-detector \
  spine-detection:latest
```

## Deployment

### Deployment Options

The application supports multiple deployment platforms. Refer to [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for comprehensive instructions.

**Recommended Platforms:**

| Platform | Tier | Pros | Documentation |
|----------|------|------|--------------|
| **Render** | Free | Easy setup, auto-deploy from Git | [Guide](DEPLOYMENT_GUIDE.md#option-1-rendercom-recommended---free--easy) |
| **Railway** | $5/month credit | Better performance, generous limits | [Guide](DEPLOYMENT_GUIDE.md#option-2-railwayapp-easy-with-better-free-tier) |
| **Hugging Face Spaces** | Free | ML-optimized, community visibility | [Guide](DEPLOYMENT_GUIDE.md#option-3-hugging-face-spaces-best-for-ml-apps) |

**Enterprise Platforms:**
- AWS (Elastic Beanstalk, ECS, Lambda)
- Azure (App Service, Container Instances)
- Google Cloud (App Engine, Cloud Run)

### Environment Configuration

**Environment Variables**
```bash
FLASK_ENV=production          # production or development
MAX_CONTENT_LENGTH=16777216   # 16MB file upload limit
MODEL_DEVICE=cuda             # cuda or cpu
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
```

### Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Configure reverse proxy (nginx/Apache)
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring and logging
- [ ] Configure automatic restarts
- [ ] Implement rate limiting
- [ ] Set up backup/disaster recovery
- [ ] Review security headers

## Usage

### Web Interface

**Step 1: Upload DICOM File**
1. Navigate to the application URL
2. Click the upload area or drag-and-drop a DICOM file
3. Supported formats: `.dcm`, `.dicom`
4. Maximum file size: 16 MB

**Step 2: Processing**
- Automatic validation of DICOM structure
- Image preprocessing and normalization
- Parallel model inference
- Result aggregation

**Step 3: Review Results**
- **Classification Output**: Normal/Abnormal with confidence score
- **Individual Model Predictions**: DenseNet, ResNet, EfficientNet scores
- **Detection Output**: Annotated image with bounding boxes (if abnormal)
- **DICOM Metadata**: Patient info, acquisition parameters

### Programmatic Access

**Example: Python**
```python
import requests

url = "http://localhost:5000/upload"
files = {"file": open("spine_xray.dcm", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**Example: cURL**
```bash
curl -X POST \
  -F "file=@spine_xray.dcm" \
  http://localhost:5000/upload
```

## API Reference

### Endpoints

#### `POST /upload`

Upload and analyze DICOM file.

**Request**
- Content-Type: `multipart/form-data`
- Body: `file` (DICOM file, max 16 MB)

**Response** (200 OK)
```json
{
  "status": "success",
  "diagnosis": "Abnormal",
  "confidence": 0.87,
  "ensemble_prediction": {
    "densenet121": 0.91,
    "resnet50": 0.82,
    "efficientnetv2": 0.88
  },
  "detections": [
    {
      "class": "lesion",
      "confidence": 0.76,
      "bbox": [120, 85, 245, 210]
    }
  ],
  "metadata": {
    "patient_id": "XXXX",
    "study_date": "20260128",
    "modality": "DX"
  },
  "annotated_image": "data:image/png;base64,..."
}
```

**Error Response** (400 Bad Request)
```json
{
  "status": "error",
  "message": "Invalid DICOM file format"
}
```

#### `GET /health`

System health check endpoint.

**Response** (200 OK)
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda",
  "timestamp": "2026-01-28T10:30:00Z"
}
```

## Development

### Project Structure

```
Deploy-Model/
├── app.py                          # Main Flask application
├── gunicorn_config.py              # Production server config
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version specification
├── Dockerfile                      # Container definition
├── Procfile                        # Heroku process configuration
├── .gitignore                      # Git ignore patterns
│
├── templates/
│   └── index.html                  # Main UI template
│
├── static/                         # Static assets (CSS, JS)
│
├── uploads/                        # Temporary file storage
│   └── .gitkeep
│
├── ensemble output/                # Classification models
│   ├── densenet121_balanced/
│   │   └── model_best.pth
│   ├── resnet50_optimized/
│   │   └── model_best.pth
│   └── tf_efficientnetv2_s_optimized/
│       └── model_best.pth
│
├── detection output/               # Detection model
│   └── yolo11/
│       └── weights/
│           └── best.pt
│
├── Testing/                        # Sample DICOM files
│   ├── normal.dicom
│   └── abnormal.dicom
│
├── docs/                           # Documentation
│   ├── DEPLOYMENT_GUIDE.md
│   ├── DEPLOYMENT.md
│   └── CONTRIBUTING.md
│
└── vindr-spinexr-dataset-analysis.ipynb  # Dataset analysis notebook
```

### Development Workflow

**1. Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

**2. Make Changes**
- Write clean, documented code
- Follow PEP 8 style guidelines
- Add unit tests for new functionality

**3. Test Locally**
```bash
python app.py
# Run tests
pytest tests/
```

**4. Commit & Push**
```bash
git add .
git commit -m "feat: descriptive commit message"
git push origin feature/your-feature-name
```

**5. Create Pull Request**
- Describe changes clearly
- Reference any related issues
- Wait for code review

### Code Style

**Python Standards**
- Follow PEP 8 conventions
- Maximum line length: 100 characters
- Use type hints where applicable
- Document functions with docstrings

**Commit Messages**
- Use conventional commits format
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Example: `feat: add batch processing endpoint`

### Testing

**Run Unit Tests**
```bash
pytest tests/ -v
```

**Run Integration Tests**
```bash
pytest tests/integration/ -v
```

**Check Code Coverage**
```bash
pytest --cov=. --cov-report=html
```

## Security

### Input Validation
- **File Type Verification**: Strict DICOM format checking via PyDICOM
- **Size Limitations**: 16 MB maximum upload size
- **Extension Validation**: Only `.dcm` and `.dicom` extensions accepted
- **Content Scanning**: DICOM header validation before processing

### Data Protection
- **Temporary Storage**: Uploaded files automatically deleted after processing
- **No Persistence**: Patient data not stored permanently
- **Secure Transmission**: HTTPS enforced in production
- **Metadata Sanitization**: PHI (Protected Health Information) handling

### Deployment Security
- Regular dependency updates via Dependabot
- Container security scanning
- Environment variable management
- Rate limiting on API endpoints

### Compliance Considerations
- HIPAA compliance requires additional infrastructure
- GDPR-compliant data handling practices
- Audit logging for production deployments
- Access control and authentication (to be implemented)

## Medical Disclaimer

**⚠️ CRITICAL NOTICE FOR CLINICAL USE ⚠️**

This software is provided **FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY** and is **NOT** intended for clinical diagnosis or patient care.

**Important Limitations:**
- ❌ Not FDA cleared or CE marked for clinical use
- ❌ Not clinically validated in prospective studies
- ❌ Not intended to replace professional medical judgment
- ❌ Results may contain false positives and false negatives
- ❌ Performance may vary on different imaging equipment
- ❌ No warranties provided regarding accuracy or reliability

**Professional Responsibility:**
- ✅ All medical imaging must be reviewed by qualified radiologists
- ✅ This tool may be used only as a supplementary aid
- ✅ Clinical decisions must be made by licensed healthcare professionals
- ✅ Always consult appropriate medical experts for diagnosis and treatment

**Liability:**
The authors and contributors assume NO responsibility for any clinical decisions made based on this software's output.

## Contributing

We welcome contributions from the community! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute

1. **Fork the Repository**
2. **Create a Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit Your Changes** (`git commit -m 'feat: add AmazingFeature'`)
4. **Push to the Branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Areas
- 🐛 Bug fixes and issue resolution
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Test coverage expansion
- 🎨 UI/UX improvements
- ⚡ Performance optimizations

### Code Review Process
All submissions require review and approval from maintainers. Please be patient and responsive to feedback.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

```
MIT License

Copyright (c) 2026 Spinal Lesion Detection System Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## Acknowledgments

### Dataset
- **VinDr-SpineXR Dataset**: Nguyen et al., VinDr-SpineXR: A large annotated medical image dataset for spinal lesion detection and classification from radiographs

### Frameworks & Libraries
- **PyTorch**: Facebook AI Research (FAIR)
- **Ultralytics YOLO**: Ultralytics team and contributors
- **Flask**: Pallets Projects
- **PyDICOM**: PyDICOM contributors

### Research Community
- Medical Imaging and Computer Vision research communities
- Open-source contributors worldwide

## Citation

If you use this software in your research, please cite:

```bibtex
@software{spine_lesion_detection_2026,
  title={Spinal Lesion Detection System},
  author={Deploy-Model Contributors},
  year={2026},
  url={https://github.com/pronad1/Deploy-Model}
}
```

## Contact & Support

**Issues & Bug Reports**
- GitHub Issues: [https://github.com/pronad1/Deploy-Model/issues](https://github.com/pronad1/Deploy-Model/issues)

**Documentation**
- Deployment Guide: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- API Documentation: [DEPLOYMENT.md](DEPLOYMENT.md)
- Contributing Guidelines: [CONTRIBUTING.md](CONTRIBUTING.md)

---
**Maintained with commitment to advancing medical AI research and open-source collaboration.**
