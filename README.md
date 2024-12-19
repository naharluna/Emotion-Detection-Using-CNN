# Emotion Detection Using CNN

Welcome to the Emotion Detection Using Convolutional Neural Networks (CNN) repository! This project leverages deep learning techniques to detect and classify emotions from facial expressions using the FER2013 dataset and a custom-trained CNN model.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Real-Time Emotion Detection](#real-time-emotion-detection)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## About the Project
This project aims to automatically recognize human emotions based on facial expressions using a deep learning model. The CNN model is trained on the FER2013 dataset and supports the following seven emotion categories:

1. Angry
2. Disgusted
3. Fearful
4. Happy
5. Neutral
6. Sad
7. Surprised

---

## Dataset
We use the FER2013 dataset, which consists of 48x48 grayscale images of faces labeled with one of the seven emotions mentioned above. The dataset includes both training and test sets, enabling robust model evaluation.

---

## Installation
Follow these steps to set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/naharluna/Emotion-Detection-Using-CNN.git
   cd Emotion-Detection-Using-CNN
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model
1. Place the FER2013 dataset in the `Dataset` directory as follows:
   ```
   Dataset/
       train/
       test/
   ```

2. Train the model using:
   ```bash
   python train_model.py
   ```

### Testing the Model
1. To evaluate the model's performance:
   ```bash
   python evaluate_model.py
   ```

### Real-Time Emotion Detection
1. Run the real-time emotion detection script:
   ```bash
   python real_time_detection.py
   ```

---

## Model Architecture
The CNN model consists of several convolutional, max-pooling, and dropout layers followed by fully connected layers. The final layer outputs a softmax probability distribution over the seven emotion classes.

---

## Training and Evaluation
- The model is trained using the Adam optimizer with a learning rate of 0.0001.
- The loss function used is categorical cross-entropy.
- Training includes data augmentation to improve generalization.

---

## Real-Time Emotion Detection
The real-time emotion detection script uses OpenCV to capture video frames, preprocess them, and classify emotions using the trained model. Detected emotions are displayed on the video feed.

---

## Results
- Achieved **X% accuracy** on the test set (update with your results).
- Confusion matrix and classification report provide detailed performance metrics.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

1. Fork the repository.
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)

---

Enjoy building and experimenting with the Emotion Detection Using CNN project! Feel free to reach out for questions or feedback.
