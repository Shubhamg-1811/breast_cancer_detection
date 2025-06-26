# 🧠 Breast Cancer Detection using Deep Learning and Fractal Geometry

This project combines Convolutional Neural Networks (CNNs) with fractal dimension features (using Differential Box Counting) to build a robust binary classifier for breast cancer detection.

## Algorithm Reference
Our algorithm is primarily based on the “Differential Box Counting Algorithm” proposed by B. B. Chaudhari and Nirupam Sarkar.

## The Differential Box Counting Algorithm
In 1992, Sarkar et al. proposed a straightforward approach for estimating the fractal dimension (FD) of a grayscale image. Their methodology involves partitioning a square image (I) of size M×M pixels into non-overlapping grids of size s×s pixels, where s is an integer ranging from 2 to M/2. The scale (r) of a grid with size s×s pixels relative to the image size M is defined as r = s/M. If s is not a divisor of M, then the non-image pixels on the boundary of the grids are treated as zero.

Within each grid, multiple boxes of size s×s×h are utilized to cover the rough gray-level image intensity surface. These boxes are assigned numerical values according to a specific scheme. Here, ⌊.⌋ denotes the floor function, and it is used to determine the number of boxes needed to cover the intensity surface, denoted by ⌊G/h⌋ = ⌊M/s⌋, which implies h = s × G/M.

This approach provides a systematic framework for computing the fractal dimension of an image, enabling the quantification of its complexity and texture. By iteratively adjusting the grid size and analyzing the distribution of intensity values within each grid, Sarkar et al.'s method facilitates the estimation of the fractal dimension, offering valuable insights into the structural properties of the image.

Reference: [MDPI Paper](https://www.mdpi.com/1099-4300/19/10/534)

## 📊 Overview

- **Input:** Breast cancer histopathology images
- **Dual Input Model:** 
  - CNN on image data
  - Fully connected layers on fractal dimension features
- **Output:** Binary classification (Benign / Malignant)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC
- **Dataset:** [BreaKHis](https://www.kaggle.com/datasets/ambarish/breakhis/code)
---

## 🛠️ Technologies Used

- Python 3.11+
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV
- Matplotlib, Seaborn
- scikit-learn

---

## 🧬 Project Pipeline

### 1. Data Preparation
- Load histopathology images and corresponding labels
- Compute **fractal dimension features** using Improved Differential Box Counting (DBC) method

### 2. Model Architecture
- **Image Branch:** CNN for feature extraction from RGB image
- **Fractal Branch:** Dense layers for scalar features (fractal dimensions)
- **Concatenation:** Merge both branches and apply final classification layer

### 3. Training
- EarlyStopping with patience of 5
- Augmented training using `ImageDataGenerator`
- Validation set used for monitoring

### 4. Evaluation
- Test set accuracy: **~89.47%**
- ROC AUC: **0.96**
- Classification report and confusion matrix shown
- **Model:** [Trained_model](https://drive.google.com/file/d/1GmSFnuqg0-az5Qw86_aq2W4qM9oX7peU/view?usp=drive_link)
---

## 📈 Results

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | 89.5%     |
| Precision    | 0.86–0.93 |
| Recall       | 0.89–0.91 |
| ROC-AUC      | 0.96      |

---

## How to Run
- Upload the model (Trained_model) and your test image to your working directory (Colab or local).
- Run the following Python script:

``` Python
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import math

# Load the trained dual-input model
model = load_model("breast_cancer_dual_input_model.h5")

# Differential Box Counting for Fractal Dimension
def differential_box_counting(grayscale_matrix):
    size_of_matrix = grayscale_matrix.shape[0]
    size_of_grid = 2
    num_of_boxes = []
    size_array = []

    while size_of_grid <= size_of_matrix // 2:
        ans = 0
        for i in range(0, size_of_matrix, size_of_grid):
            for j in range(0, size_of_matrix, size_of_grid):
                sub_mat = grayscale_matrix[i:i+size_of_grid, j:j+size_of_grid]
                if sub_mat.shape[0] != size_of_grid or sub_mat.shape[1] != size_of_grid:
                    continue
                max_gray = np.max(sub_mat)
                min_gray = np.min(sub_mat)
                delta = math.ceil(max_gray / size_of_grid) - math.ceil(min_gray / size_of_grid) + 1
                ans += delta
        num_of_boxes.append(ans)
        size_array.append(size_of_grid)
        size_of_grid *= 2

    if len(num_of_boxes) < 2:
        return 0.0

    x = np.log(np.array([size_of_matrix / s for s in size_array]))
    y = np.log(np.array(num_of_boxes))
    slope, _ = np.polyfit(x, y, 1)
    return -slope

# Prediction function
def predict_image(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))
        plt.imshow(img)
        plt.title("Test Image")
        plt.axis('off')
        plt.show()

        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        gray = cv2.cvtColor((img_arr[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        fd = differential_box_counting(gray)
        fd = np.array([[fd]])

        prediction = model.predict([img_arr, fd])
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]

        labels = ['Benign', 'Malignant']
        print(f"Prediction: {labels[class_index]} ({confidence*100:.2f}%)")
    except Exception as e:
        print(f"Error: {e}")

# Change the path to your test image
predict_image("your_image_path_here.png")
