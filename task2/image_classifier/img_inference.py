import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "animal_classifier.h5")
LE_PATH = os.path.join(BASE_DIR, "models", "label_encoder.npy")

model = tf.keras.models.load_model(MODEL_PATH)
le_classes = np.load(LE_PATH, allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = le_classes


def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    image = cv2.resize(image, target_size)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb / 255.0  # Normalize pixel values
    return image_rgb, image_norm


def classify_image(image_path):
    """
    Classifies the image and returns the predicted label and the original image.
    """
    orig_image, processed_image = preprocess_image(image_path)
    img_batch = np.expand_dims(processed_image, axis=0)
    pred = model.predict(img_batch)
    pred_class = np.argmax(pred, axis=-1)[0]
    predicted_label = label_encoder.inverse_transform([pred_class])[0]
    return predicted_label, orig_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal image classification.")
    parser.add_argument("image_path", type=str, help="Path to the image for classification")
    args = parser.parse_args()
    image_path = args.image_path

    try:
        predicted_label, orig_image = classify_image(image_path)
    except Exception as e:
        print(e)
        exit(1)

    print(f"Predicted class: {predicted_label}")

    plt.imshow(orig_image)
    plt.title(f"Predicted class: {predicted_label}")
    plt.axis("off")
    plt.show()
