# SmartRecycleNet: Real-Time Recyclable Classification on Jetson Nano

This project deploys a lightweight machine learning model on an embedded Jetson Nano to classify recyclable waste items — **plastic, paper, glass, and metal** — in real-time using computer vision.

## 🔍 Problem Statement

Many countries lack the infrastructure for proper waste classification, especially at the point of disposal. Traditional recycling infrastructure is expensive and requires significant public behavior change. Our goal is to develop a **low-cost, behavior-independent** machine learning solution that automates recyclable classification.

## 💡 Solution Overview

- **Transfer learning** with MobileNetV2 and EfficientNetB0 (pretrained on ImageNet)
- Trained on two combined public datasets:
  - [TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet)
  - [Recycle_Classification_Dataset](https://www.kaggle.com/datasets/jinfree/recycle-classification-dataset)
- **MobileNetV2** chosen for deployment due to high accuracy (84%) and low parameter count
- Real-time classification on **Jetson Nano** with live camera integration

## 🛠️ Tech Stack

- Python, TensorFlow/Keras
- Transfer Learning (MobileNetV2, EfficientNetB0)
- ONNX Runtime
- NVIDIA Jetson Nano
- OpenCV + CSI Camera integration

## 🧪 Training Pipeline

1. **Feature Extraction Phase**: freeze pretrained base, train custom top layers
2. **Fine-Tuning Phase**: unfreeze base, reduce LR, adapt to recyclable data
3. **Data Augmentation**: horizontal flip, rotation, zoom, translation
4. Evaluated with **confusion matrix**, **per-class accuracy**, and **classification report**

## 📦 Deployment

- Converted Keras `.h5` model to `.onnx`
- Deployed to Jetson Nano for **on-device inference**
- Integrated CSI-Camera feed with real-time overlay of classification result

## 📊 Results

| Model         | Accuracy | Params     | Suitability for Embedded |
|---------------|----------|------------|---------------------------|
| MobileNetV2   | 84%      | ~2.2M      | ✅ Best Trade-off         |
| EfficientNetB0| 73%      | ~5.3M      | ❌ Too heavy              |

- Struggles with glass classification due to visual similarity and camera quality

## 🚀 Future Work

- 2-stage classification: Trash vs Recyclable → Type of recyclable
- Federated Learning for distributed model improvements
- Enhanced image quality & camera tuning for better accuracy

## 🧠 Authors

Jay Siegfried, Byung Woong Ko  
ECE 629 - UMass Amherst

