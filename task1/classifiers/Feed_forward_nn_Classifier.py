import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
from MnistClassifierInterface import MnistClassifierInterface
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Feed_Forward_nn_Classifier(MnistClassifierInterface):
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def preprocess(self, x, y):
        x = x.astype('float32') / 255.0
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=-1)
        if y is not None:
            y = to_categorical(y, num_classes=10)
        return x, y

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        x_train, y_train = self.preprocess(x_train, y_train)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ]

        # Fit the model and store the history
        history = self.model.fit(x_train, y_train,
                                 validation_split=0.2,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=callbacks,
                                 verbose=1)

        plt.figure(figsize=(12, 6))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def predict(self, x_test):
        x_test, _ = self.preprocess(x_test, None)
        return tf.argmax(self.model.predict(x_test), axis=1).numpy()

    def validate(self, x_val, y_val):
        x_val, y_val = self.preprocess(x_val, y_val)
        loss, accuracy = self.model.evaluate(x_val, y_val, verbose=0)
        return accuracy

    def visualize_results(self, x_test, y_test, num_samples=10):
        X_test_preprocessed, y_test = self.preprocess(x_test, y_test)
        predictions_prob = self.model.predict(X_test_preprocessed)
        predictions = np.argmax(predictions_prob, axis=-1)

        cm = confusion_matrix(np.argmax(y_test, axis=1), predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Accuracy: {accuracy_score(np.argmax(y_test, axis=1), predictions):.2%})')
        plt.show()

        sample_indices = np.random.choice(len(x_test), size=num_samples, replace=False)
        plt.figure(figsize=(num_samples * 1.5, 2))
        for i, idx in enumerate(sample_indices):
            plt.subplot(1, num_samples, i + 1)
            img = x_test[idx]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze()
            plt.imshow(img, cmap='gray')
            true_label = np.argmax(y_test[idx])
            plt.title(f"T:{true_label}\nP:{predictions[idx]}", fontsize=8)
            plt.axis('off')
        plt.suptitle("Sample Predictions")
        plt.tight_layout()
        plt.show()
