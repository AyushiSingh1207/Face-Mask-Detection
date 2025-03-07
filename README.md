# Face-Mask-Detection
Project Overview

This project focuses on building a deep learning model to detect face masks in real-time using OpenCV, TensorFlow/Keras, and a pre-trained face detection model. The goal is to enhance public safety by identifying individuals who are not wearing masks. The project involves data collection, preprocessing, model training, and real-time detection.

Project Workflow

1. Environment Setup

Tools Used: Python, OpenCV, TensorFlow, Keras, Matplotlib, NumPyObjective: Create a structured workspace and organize project files efficiently for seamless data processing.

2. Dataset Collection & Preprocessing

Source: Open-source datasets for masked and unmasked faces.Storage Structure: The dataset is stored in the dataset/ directory.Preprocessing Steps: Convert images to grayscale (optional). Resize images to a uniform size (e.g., 224x224). Normalize pixel values for better model convergence.

3. Installing Required Libraries

Install necessary dependencies:

pip install -r requirements.txt  

Key Libraries: OpenCV, TensorFlow, Keras, Matplotlib, NumPy, Scikit-learn

4. Data Augmentation (Current Stage)

Objective: Improve model generalization by applying transformations.Techniques Used: Rotation, flipping, zoom, brightness adjustments.Implementation: Using ImageDataGenerator from Keras to generate augmented images dynamically during training.

5. Model Training (Next Steps)

Train a Convolutional Neural Network (CNN) on augmented data. Evaluate model performance using accuracy, precision, recall, and F1-score. Save the trained model as mask_detector.model.

6. Project Requirements

Python Version: 3.7+Key Dependencies: TensorFlow, OpenCV, NumPy, Matplotlib, Scikit-learn

7. Key Findings & Insights

Data augmentation improves model robustness against variations in lighting and pose. CNN-based architectures effectively differentiate between masked and unmasked faces. OpenCV enables real-time mask detection with minimal computational overhead.

8. Future Enhancements

Implement real-time detection using detect_mask_video.py. Deploy the model as a web application using Flask. Optimize the model for mobile and embedded systems.

9. Acknowledgments

Dataset Sources: Kaggle, OpenCV face datasets.Inspiration: COVID-19 safety measures and real-world mask detection applications.
