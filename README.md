# Hybrid Models for Sign Language to Text Translation

This project explores the feasibility of translating American Sign Language (ASL) from video input into written text through various deep learning models. The work includes multiple approaches, detailed in the notebook and report. My contributions focused on implementing transfer learning techniques and developing a Convolutional Neural Network (CNN) using Mediapipe landmarks (Landmark CNN), which ultimately demonstrated superior performance.

## 📊 Dataset Overview

The dataset used in this work is a large-scale Word-Level American Sign Language (WLASL) video dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=nslt_2000.json), containing more than 2000 words performed by over 100 signers. It includes:

- 21,083 RGB-based videos performed by 119 signers
- Each video contains a single ASL sign, lasting 2.5 seconds on average
- Detailed descriptions provided in a `WLASL_v0.3.json` file, including attributes like `gloss`, `fps`, `split`, `frame_start`, `frame_end`, `url`, and `video_id`

Below a video example of our dataset.

https://github.com/user-attachments/assets/dd327847-ac4b-4681-8751-c68309893dc0

## 🚀 Outline of the project

### Transfer Learning

We utilized various pre-trained convolutional neural networks (CNNs) for video classification, adapting them for ASL recognition:

 - **InceptionV3**: Captures complex spatial hierarchies.
 - **ResNet50**: Learns intricate patterns without vanishing gradients.
 - **EfficientNetV2L**: Balances accuracy and computational efficiency.
 - **VGG16**: Strong baseline feature extraction.
 - **InceptionResNetV2**: High performance with lower computational cost.

### Landmarks CNN

In the context of computer vision and machine learning, a landmark refers to specific, predefined points on an object that are used to understand its structure and spatial configuration. Landmarks are typically chosen because they are stable, easily identifiable, and relevant. We used Google MediaPipe to extract hand landmarks from video frames and fed them into a 2D CNN for translation tasks.
  
  ![landmarks](https://github.com/user-attachments/assets/a2af7948-47b2-4745-b7cd-b20fbd5a5a90)

Despite experimenting with various architectures, some models underperformed due to the low quality of the dataset. However, our best-performing model, the 2D-CNN using landmarks, achieved promising results even with an increased label set. 

The robustness of our approach was evaluated on a set of **1000 labels**, achieving a maximum accuracy of 27%. The stabilization of validation loss and accuracy indicates the model's potential for handling large-scale classification tasks within a simplified architecture.

## References

[1] https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
