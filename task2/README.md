## Named Entity Recognition (NER) + Image Classification task

This project implements a pipeline that combines Named Entity Recognition (NER) for animal detection in text and Image
Classification for animal detection in images. The system verifies if the animal mentioned in the text matches the
animal present in the provided image.

## 📑Table of Contents

- 📖[Project Overview](#project-overview)
- 📊[Dataset](#dataset)
- 🌐[Models](#models)
- 📁[Project Structure](#project-structure)
- 🛠️[Installation](#installation)
- 🚀[Usage](#usage)

## 📖Project Overview

This project is a **Named Entity Recognition (NER)** system that identifies animal names in text using a custom-trained
model based on the `BERT` architecture. The NER model is then combined with an **Image Classification** system that
detects animals in images using a pre-trained model such as `EfficientNetB3`. The pipeline verifies if the animal
mentioned in the text matches the animal present in the image, providing a comprehensive solution for animal detection
across different modalities.

## 📊Dataset

#### Text Data (NER)

The NER dataset was created through the following steps:

- **Data Collection:** Retrieved Wikipedia articles for a list of animal names using API requests.
- **Data Cleaning:** Filtered and cleaned the paragraphs mentioning each animal.
- **Dataset Aggregation:** Combined the cleaned paragraphs into a dataset where each record includes the article text
  and the corresponding animal name as the label.
- **Preprocessing:**
    - Split the text into sentences and words.
    - Applied lemmatization.
    - Transformed the text into a tagged format by labeling words as "ANIMAL" if they matched the animal name or "O"
      otherwise.
      **Output:** Organized the tagged data into a DataFrame and exported it to a CSV file for further analysis and
      model training.

#### Image Data (Classification)

For image classification, the dataset was taken from Kaggle at
👉[link](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/data).
This dataset contains images of **90** different animal species, enabling multi-class classification. Each category
includes a substantial number of images, ensuring diversity for effective model training. The dataset is an excellent
resource for training and evaluating image recognition systems in animal classification tasks.

## 🌐Models

### Image Classification Model🖼️

The image classifier is built using the **EfficientNetB3** architecture pre-trained on ImageNet. It is fine-tuned on our
animal image dataset and employs additional layers such as Global Average Pooling, a dense ReLU layer, and dropout for
regularization, ending with a softmax output layer for multi-class classification.

### Named Entity Recognition (NER) Model📝

The **NER** model is based on the **BERT** architecture and is specifically trained for token classification to identify
animal names in text. This model is available on the Hugging Face Hub
👉[link](https://huggingface.co/sofibrezden/animal-ner), allowing for easy access and integration.

### Unified Pipeline🔄

Both models are integrated into a unified pipeline that processes text and image inputs simultaneously. The system
verifies whether the animal mentioned in the text corresponds to the animal present in the image, providing a
comprehensive solution for multi-modal animal detection.


## 📁Project Structure

```
├── dataset-for-testing-pipeline/    # Test dataset for verifying the pipeline  
├── image_classifier/                # Directory for image classification models  
│   ├── data/                        # Data for training the classification model  
│   │   ├── animals/                 # Folder containing animal-related photos  
│   │   │   ├── names.txt            # Text file with names of animals  
│   ├── models/                      # Directory for storing trained models  
│   │   ├── animal_classifier.h5     # Trained model in HDF5 format  
│   │   ├── label_encoder.npy        # Encoded labels for classification  
│   ├── __init__.py                  # Init file for the module  
│   ├── eda.ipynb                    # Notebook for Exploratory Data Analysis (EDA)  
│   ├── img_inference.py             # Script for running inference on the image classifier  
│   ├── img_training.py              # Script for training the image classification model  
├── ner/                              # Directory for the NER model  
│   ├── data/                         # Data for training the NER model  
│   │   ├── __init__.py  
│   │   ├── ner_dataset_creation&processing.ipynb  # Dataset creation & processing for NER  
│   ├── ner_inference.py              # Script for running inference on the NER model  
│   ├── ner_training.py               # Script for training the NER model  
│   ├── dataset_gen_for_demo.py       # Script for generating a demo dataset  
│   ├── demo.ipynb                    # Jupyter Notebook for pipeline demonstration  
├── pipeline.py                        # Main script that runs the entire pipeline  
├── README.md                          # Project documentation  
├── requirements.txt                    # List of dependencies  
```

## 🛠️Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/sofibrezden/mnist-ner-image_classification.git
    cd task2
    ```
2. Create a virtual environment :
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```


## 🚀Usage

### Image Classification

#### Training

To train the image classification model, use the img_training.py script. The script loads the dataset, preprocesses the
images, and fine-tunes the EfficientNetB3 model for image classification. You can adjust any parameters as needed.

```
cd image_classifier
python img_training.py \
  --dataset_dir data/animals \
  --labels_file data/names.txt \
  --img_size 224 \
  --test_size 0.2 \
  --batch_size 32 \
  --epochs 30 \
  --learning_rate 1e-4 \
  --fine_tune_layers 50 \
  --model_output animal_classifier.h5 \
  --label_output label_encoder.npy
```

#### Inference

To run inference on the trained image classification model, use the img_inference.py script. This script loads the
trained model and label encoder, processes the input image, and predicts the animal class.

```
cd image_classifier
python img_inference.py /path/to/image.jpg
```

### Named Entity Recognition (NER)

#### Training

To train the NER model for animal detection in text, use the ner_training.py script. This script loads the NER dataset,
fine-tunes the BERT model, and saves the trained model. As with the image classification script, you can modify parameters to suit your needs.

```
cd ner
python ner_training.py \
  --dataset_path tagged_dataset.csv \
  --model_name bert-base-cased \
  --output_dir ./models \
  --logging_dir ./logs \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --epochs 10 \
  --test_size 0.2 \
  --early_stopping_patience 3
```

#### Inference

To run inference on the trained NER model, use the ner_inference.py script. This script loads the trained model and
processes the input text to detect animal names.

```
cd ner
python ner_inference.py "your sentence here"
```

##### Running the Pipeline

To run the pipeline that processes the downloaded images and matches objects in the images with names mentioned in
sentences, use the pipeline.py script.
```python pipeline.py "A sentence describing the image" "path/to/image.jpg"```

For example:

```python pipeline.py "A brown dog is running in the park" "dataset-for-testing-pipeline/dog.jpg"```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.