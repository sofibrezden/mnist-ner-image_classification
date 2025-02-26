## MNIST Classification with OOP

This project leverages the **MNIST** dataset to classify handwritten digits using three distinct approaches: **Random Forest, Feed-Forward Neural Network (NN), and Convolutional Neural Network (CNN)**. Each model is built with object-oriented programming (**OOP**) principles to ensure a modular, maintainable, and scalable codebase.

## 📑Table of Contents
- 📖[Project Overview](#project-overview)
- 📊[Dataset](#dataset)
- 📁[Project Structure](#project-structure)
- 🛠️[Installation](#installation)
- 🚀[Usage](#usage)
- 📁[Files](#files)

## 📖Project Overview
In this project, I explore different **machine learning techniques for image classification**. The workflow begins with comprehensive data preprocessing of the MNIST dataset, followed by the implementation and evaluation of three models:

- **Random Forest:** An ensemble-based approach to improve prediction reliability.
- **Feed-Forward Neural Network (NN):** A simple neural architecture designed to capture complex patterns.
- **Convolutional Neural Network (CNN):** A deep learning model specifically tailored for image feature extraction.

By leveraging OOP, each model is encapsulated within its own class, making it easier to develop, test, and extend. This structured approach not only simplifies debugging and maintenance but also facilitates the integration of additional models and features in the future.
### 📊Dataset

The MNIST dataset is used for training and testing the models. It consists of 60,000 training images and 10,000 testing
images of handwritten digits (0-9).

### 📁Project Structure
``` task1/
   │── classifiers/
   │   ├── CNNClassifier.py              # CNN Model Implementation
   │   ├── Feed_forward_nn_Classifier.py # Feed-Forward NN Model
   │   ├── MnistClassifierInterface.py   # Interface for classifiers
   │   ├── RFClassifier.py               # Random Forest Model
   │   ├── MnistClassifier.py            # Base MNIST classifier class
   │── demo.ipynb                        # Jupyter notebook for testing models
   │── README.md                         # Project documentation
   │── requirements.txt                   # Required dependencies
```

### 🛠️Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```
2. Create a virtual environment :
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
### 🚀Usage

1. **🤖Training the Models**:
    - To train the Feed-Forward Neural Network:
        ```python
        from task1.classifiers.Feed_forward_nn_Classifier import FeedForwardNNClassifier

        classifier = FeedForwardNNClassifier()
        classifier.train(x_train, y_train)
        ```

    - To train the Random Forest classifier:
        ```python
        from task1.classifiers.RFClassifier import RFClassifier

        classifier = RFClassifier()
        classifier.train(x_train, y_train)
        ```

    - To train the Convolutional Neural Network:
        ```python
        from task1.classifiers.CNNClassifier import CNNClassifier

        classifier = CNNClassifier()
        classifier.train(x_train, y_train)
        ```

2. **✨Making Predictions**:
    - To make predictions with the Feed-Forward Neural Network:
        ```python
        predictions = classifier.predict(x_test)
        ```

    - To make predictions with the Random Forest classifier:
        ```python
        predictions = classifier.predict(x_test)
        ```

    - To make predictions with the Convolutional Neural Network:
        ```python
        predictions = classifier.predict(x_test)
        ```

3. **🤖Validating the Models**:
    - To validate the Feed-Forward Neural Network:
        ```python
        accuracy = classifier.validate(x_val, y_val)
        ```

    - To validate the Random Forest classifier:
        ```python
        metrics = classifier.validate(x_val, y_val)
        ```

    - To validate the Convolutional Neural Network:
        ```python
        accuracy = classifier.validate(x_val, y_val)
        ```

4. **📊Visualizing Results**:
    - To visualize results for the Feed-Forward Neural Network:
        ```python
        classifier.visualize_results(x_test, y_test)
        ```

    - To visualize results for the Random Forest classifier:
        ```python
        classifier.visualize_results(x_test, y_test)
        ```

    - To visualize results for the Convolutional Neural Network:
        ```python
        classifier.visualize_results(x_test, y_test)
        ```

5. **🌐Using the Demo Notebook**:
    - Open the `demo.ipynb` notebook to see examples of how to train, predict, validate, and visualize results for all models. The notebook provides a step-by-step guide and visualizations for better understanding.

### 📁Files
- **task1/classifiers/CNNClassifier.py:** Contains the implementation of the Convolutional Neural Network (CNN) classifier.
- **task1/classifiers/Feed_forward_nn_Classifier.py:** Contains the implementation of the Feed-Forward Neural Network (NN) classifier.
- **task1/classifiers/MnistClassifierInterface.py:** Defines the interface that all classifiers must implement.
- **task1/classifiers/RFClassifier.py:** Contains the implementation of the Random Forest classifier.
- **task1/classifiers/MnistClassifier.py:** Base class for MNIST classifiers.
- **task1/demo.ipynb:** Jupyter notebook for testing the models.
- **task1/README.md:** Project documentation.
- **task1/requirements.txt:** Lists the required dependencies for the project.

### 🌐Edge Cases
**1. Input Data Shape:**  
- Ensure correct input shape: flattened for Feed-Forward NN, 2D/3D for CNN.

**2. Missing/Corrupted Data:**  
- Ensure correct input shape: flattened for Feed-Forward NN, 2D/3D for CNN.

**3.Small Datasets:** 
- Ensure models perform well on small datasets.

**4. Class Imbalance:**
- Use techniques like oversampling, undersampling, or class weights.

**5. Large Batch Sizes or Memory Constraints**
- Be mindful of batch sizes if using GPU-accelerated training with large datasets.

### 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.