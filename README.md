## **Traffic Sign Classification using CNN**

### 🚦 **Overview**
This project implements a Convolutional Neural Network (CNN) to classify traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model processes images, identifies patterns, and predicts the correct traffic sign class among 43 categories. It achieves high accuracy through rigorous training, optimization, and validation.

### 🎯 **Features**
- Multi-class classification of 43 traffic sign categories.
- Data preprocessing and augmentation for robust training.
- Configurable CNN architecture with customizable layers, filters, dropout rates, and optimizers.
- Performance visualization through accuracy plots, confusion matrices, and ROC curves.
- Evaluation metrics include Accuracy, Precision, Recall, and F1-Score.

### 📂 **Project Structure**
```
SignClassifier.ipynb       # Main notebook containing code and visualizations
README.md                  # Project documentation
```

### 🛠️ **Technologies Used**
- **Frameworks:** TensorFlow/Keras, Scikit-learn, Seaborn, Matplotlib
- **Languages:** Python
- **Tools:** Jupyter Notebook, Kaggle API

### 📊 **Dataset**
- **Source:** [Kaggle - GTSRB Dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Preprocessing:** Images resized to 30x30 pixels and normalized using MinMaxScaler.
- **Split:** Training (80%), Validation (20%), and Testing.

### 🔧 **Setup and Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/GururajaSR/Traffic-Sign-Classification-using-CNN.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Traffic-Sign-Classification
   ```
3. Run the notebook using Jupyter or Google Colab.

### 🚀 **How to Run**
1. Download the dataset using Kaggle API.
2. Preprocess the data for training and validation.
3. Configure and train the CNN model using various parameters.
4. Evaluate the model on the validation and test sets.
5. Visualize the results with confusion matrices, ROC curves, and performance metrics.

### 📈 **Results**
- **Best Model Accuracy:** 99.72% (Validation) and 97.20% (Test).
- **Confusion Matrix:** Displays the classification accuracy per class.
- **ROC Curve:** Highlights model performance with an AUC close to 1.
- **Metrics Summary:**
  - Precision: 95.54%
  - Recall: 95.61%
  - F1 Score: 95.49%

### 📘 **Model Insights**
The CNN architecture uses multiple convolutional layers, pooling, dropout regularization, and a final dense softmax layer. Hyperparameters like learning rate, optimizer type, and filter size were fine-tuned for optimal performance.

### 📜 **License**
This project is licensed under the MIT License. See the LICENSE file for details.
