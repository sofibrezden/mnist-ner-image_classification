import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import plotly.graph_objects as go


# Function to preprocess an image
def preprocess_image(image_path, img_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize pixel values
    return image


def load_dataset(dataset_dir, labels_file, img_size):
    with open(labels_file, 'r') as f:
        class_names = f.read().strip().split('\n')

    data, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            data.append(preprocess_image(img_path, img_size))
            labels.append(class_name)

    data, labels = np.array(data), np.array(labels)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return data, labels, label_encoder


# Function to build the model
def build_model(input_shape, num_classes, fine_tune_layers=50):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[-fine_tune_layers:]:  # Fine-tune last N layers
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


# Function to plot training history
def plot_history(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(history.history['accuracy']) + 1)), y=history.history['accuracy'],
                             mode='lines+markers', name='Training Accuracy'))
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history.history['val_accuracy']) + 1)), y=history.history['val_accuracy'],
                   mode='lines+markers', name='Validation Accuracy'))
    fig.update_layout(title='Model Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy', template='plotly_dark')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(history.history['loss']) + 1)), y=history.history['loss'],
                             mode='lines+markers', name='Training Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, len(history.history['val_loss']) + 1)), y=history.history['val_loss'],
                             mode='lines+markers', name='Validation Loss'))
    fig.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss', template='plotly_dark')
    fig.show()


def main(args):
    data, labels, label_encoder = load_dataset(args.dataset_dir, args.labels_file, args.img_size)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=args.test_size, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    model = build_model((args.img_size, args.img_size, 3), len(label_encoder.classes_), args.fine_tune_layers)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    # Train model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=args.batch_size),
                        epochs=args.epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[lr_scheduler, early_stopping])

    # Save model and label encoder
    model.save(args.model_output)
    np.save(args.label_output, label_encoder.classes_)

    # Plot training history
    plot_history(history)

    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/animals", help="Dataset directory")
    parser.add_argument("--labels_file", type=str, default="data/names.txt", help="File containing class names")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (width and height)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--fine_tune_layers", type=int, default=50, help="Number of layers to fine-tune")
    parser.add_argument("--model_output", type=str, default="animal_classifier.h5",
                        help="Path to save the trained model")
    parser.add_argument("--label_output", type=str, default="label_encoder.npy", help="Path to save the label encoder")

    args = parser.parse_args()
    main(args)
