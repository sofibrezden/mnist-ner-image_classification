import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from MnistClassifierInterface import MnistClassifierInterface
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns


class CNNClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adamax',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def preprocess(self, x, y):
        x = x.astype('float32') / 255.0
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=-1)
        if y is not None:
            y = to_categorical(y, num_classes=10)
        return x, y

    def train(self, x_train, y_train):
        X_train_preprocessed, y_train_preprocessed = self.preprocess(x_train, y_train)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        history = self.model.fit(
            X_train_preprocessed,
            y_train_preprocessed,
            epochs=10,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()

    def validate(self, x_val, y_val):
        x_val_preprocessed, y_val_preprocessed = self.preprocess(x_val, y_val)
        return self.model.evaluate(x_val_preprocessed, y_val_preprocessed)

    def predict(self, x_test):
        x_test_preprocessed, _ = self.preprocess(x_test, np.zeros(x_test.shape[0]))
        return np.argmax(self.model.predict(x_test_preprocessed), axis=-1)

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
