Breast Cancer Detection Using Convolutional Neural Networks
Project Overview
This project explores the use of deep learning techniques to classify breast cancer using ultrasound images. A Convolutional Neural Network (CNN) model was developed to differentiate between benign and malignant breast tissue. The model was trained on a publicly available dataset of labeled ultrasound images and was evaluated using performance metrics such as accuracy, precision, recall, and F1-score.

The project aims to contribute to early detection methods for breast cancer, supporting radiologists in clinical decision-making by integrating artificial intelligence (AI) in diagnostic workflows.

Table of Contents
Project Overview
Data Source
Model Architecture
VGG16
VGG19
InceptionV3
ResNet50
EfficientNetB0
Evaluation Metrics
Accuracy
Precision
Recall
F1-score
Application
How to Run the Project
Future Work
Contributors
License
Data Source
The dataset used in this project was sourced from Kaggle. It contains ultrasound images categorized into two classes:

Benign: No breast cancer detected
Malignant: Breast cancer detected
Key features:

Grayscale ultrasound images.
Data split into training, validation, and testing sets.
Data Preprocessing
The dataset underwent several preprocessing steps:

Resizing images to 224x224 pixels.
Normalization of pixel values.
Data Augmentation: Techniques such as rotation and flipping to enhance robustness.
Model Architecture
This project experimented with various CNN architectures to classify breast cancer from ultrasound images. EfficientNetB0 emerged as the most accurate model, followed by VGG16 and ResNet50. Below are the details of the models:

VGG16
A 16-layer CNN, fine-tuned for the task of breast cancer detection. It demonstrated strong performance in the early stages of testing but was later surpassed by EfficientNetB0.

VGG19
An extended version of VGG16 with 19 layers. VGG19 provided a similar level of accuracy, but slightly lower generalization on unseen data.

InceptionV3
This model utilizes parallel convolutions with different kernel sizes, designed to improve classification performance by capturing different feature scales. However, it struggled to generalize effectively for this task.

ResNet50
A deeper architecture with residual connections that help avoid the vanishing gradient problem. ResNet50 performed well but showed some overfitting on the validation dataset.

EfficientNetB0
This model achieved the highest accuracy across all architectures, due to its efficient use of parameters and superior compound scaling technique. It provided a balanced performance in both precision and recall metrics.

Evaluation Metrics
The performance of the models was assessed using several evaluation metrics:

Accuracy
EfficientNetB0 achieved an accuracy of 97.27%, outperforming other models.
VGG16 reached 98.05%, while ResNet50 achieved 83.20%.
Precision
Precision measures how many selected items are relevant:

EfficientNetB0 achieved 98% precision for benign cases and 95% for malignant cases.
Recall
Recall measures how many relevant items are selected:

EfficientNetB0 had a recall of 96% for benign and 98% for malignant cases.
F1-score
The harmonic mean of precision and recall:

EfficientNetB0 F1-scores were 0.97 for both benign and malignant classes.
Application
A Streamlit web-based application was developed to classify ultrasound images as benign or malignant. Users can upload an ultrasound image, and the model will classify it based on a predefined threshold.

Key Features:
File Upload: Upload ultrasound images for classification.
Real-time Prediction: The app returns a probability score indicating whether the tissue is benign or malignant.
Streamlit Interface: Intuitive and easy to use for non-technical users, including lab assistants.
Future Work
External Validation: Test the model on datasets from different clinical settings to improve its generalizability.

Explainability: Integrate explainable AI techniques to visualize which parts of the image the model focuses on during prediction.

Multimodal Learning: Combine ultrasound imaging with other diagnostic data (e.g., mammography or MRI) for a more robust prediction system.

Clinical Integration: Explore real-world deployment options by integrating the model into clinical workflows.

Contributors
