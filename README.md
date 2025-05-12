# Sugar Beet and Weed Classification using YOLOv12s

<div align="center">
  <img src="https://stmaaprodfwsite.blob.core.windows.net/assets/sites/1/2024/02/Sugar-beet-plants-growing-on-black-fen-soil-Cambridgeshire-early-June-c-GNP-web.jpg" alt="Sugar Beet Classification Banner" width="800"/>
</div>

## 📋 Table of Contents
- [Project Overview](##project-overview)
- [Dataset](##dataset)
- [Features](##features)
- [Model Architecture](##model-architecture)
- [Performance Metrics](##performance-metrics)
- [Installation](##installation)
- [Usage](##usage)
- [Training Process](##training-process)
- [Results](##results)
- [Future Work](##future-work)
- [Acknowledgments](#acknowledgments)

## 🌱 Project Overview
This project implements a robust computer vision solution to classify sugar beets and weeds in agricultural fields. By leveraging YOLOv12s, a state-of-the-art object detection model, the system accurately identifies and differentiates between sugar beets and various weed types, enabling precision agriculture applications.

## 📊 Dataset
The dataset used in this project is the comprehensive Sugar Beets dataset from Roboflow:
- **Source**: [Sugar Beets Dataset on Roboflow](https://universe.roboflow.com/vision-3gxqu/sugarbeets-zg7nc/dataset/2)
- **Contents**: High-quality images of sugar beet fields with detailed annotations for sugar beets and various weed species
- **Split**: Train/Validation/Test sets for robust model evaluation

## ✨ Features
- Real-time detection and classification of sugar beets and weeds
- High accuracy object detection using YOLOv12s architecture
- Optimized for agricultural field conditions
- Support for various image input formats

## 🔧 Model Architecture
This project utilizes YOLOv12s (You Only Look Once), which offers:
- Fast inference times suitable for real-time applications
- Excellent accuracy in complex agricultural scenes
- Efficient architecture optimized for deployment

## 📈 Model Performance Metrics

The following charts represent the performance of the trained model, showcasing key evaluation metrics.

### F1-Score Curve
The F1-Score curve demonstrates the model's balance between precision and recall, providing insights into its ability to classify both positive and negative instances correctly.
<div align="center">
  <img src="https://github.com/mohamed-ehab415/Suger_beets/blob/main/runs/detect/train/F1_curve.png" alt="F1 Score Curve" width="700"/>
</div>

### Precision-Recall Curve
The Precision-Recall curve illustrates the trade-off between precision (positive prediction accuracy) and recall (true positive rate). This metric is especially useful when dealing with imbalanced datasets.
<div align="center">
  <img src="https://github.com/mohamed-ehab415/Suger_beets/blob/main/runs/detect/train/PR_curve.png" alt="Precision-Recall Curve" width="700"/>
</div>

### Model Training Results
The table below summarizes the final evaluation results of the model, including accuracy, loss, and other key metrics:
<div align="center">
  <img src="https://github.com/mohamed-ehab415/Suger_beets/blob/main/runs/detect/train/results.png" alt="Model Training Results" width="700"/>
</div>

## ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/mohamed-ehab415/Suger_beets.git
cd Suger_beets

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## 🚀 Usage

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('models/best.pt')

# Perform inference on an image
results = model('path_to_test_image.jpg')

# Display results
results[0].show()

# Save results
results[0].save(filename='prediction.jpg')
```

## 🔬 Training Process
The model training process involved:

1. **Data Preparation**: Downloaded and preprocessed the Sugar Beets dataset from Roboflow
2. **Model Configuration**: Set up the YOLOv12s model with optimized hyperparameters
3. **Training Environment**: Utilized Kaggle's GPU acceleration for efficient training
4. **Monitoring**: Tracked key metrics during training using TensorBoard
5. **Validation**: Regularly validated performance against a held-out validation set

The training was performed on Kaggle: [Sugar Beet Classification Notebook](https://www.kaggle.com/code/mohamedehab0122/using-kaggel/edit)

## 🏆 Results
The model achieved impressive results:
- High precision and recall for both sugar beet and weed classes
- Fast inference time suitable for real-time applications
- Robust performance across varying field conditions and lighting

## 🔮 Future Work
- Expand detection capabilities to identify specific weed species
- Implement edge deployment for in-field robotics applications
- Develop a user-friendly interface for agricultural technicians
- Integrate with automated weed removal systems



## 🙏 Acknowledgments
- Roboflow for providing the comprehensive Sugar Beets dataset
- Ultralytics for the powerful YOLOv12s implementation
- Kaggle for computing resources used during model training
