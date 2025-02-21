from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from MnistClassifierInterface import MnistClassifierInterface
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class RFClassifier(MnistClassifierInterface):
    def __init__(self):
        self._predictor = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    def preprocess(self, x, y):
        x = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
        return x, y

    def train(self, x_train, y_train):
        X_train_preprocessed, y_train_preprocessed = self.preprocess(x_train, y_train)
        self._predictor.fit(X_train_preprocessed, y_train_preprocessed)

    def predict(self, x_test):
        X_test_preprocessed, _ = self.preprocess(x_test, None)
        return self._predictor.predict(X_test_preprocessed)

    def validate(self, x_val, y_val):
        X_val_preprocessed, y_val_preprocessed = self.preprocess(x_val, y_val)
        predictions = self._predictor.predict(X_val_preprocessed)
        accuracy = accuracy_score(y_val_preprocessed, predictions)
        precision = precision_score(y_val_preprocessed, predictions, average='weighted')
        recall = recall_score(y_val_preprocessed, predictions, average='weighted')
        f1 = f1_score(y_val_preprocessed, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def visualize_results(self, x_test, y_test, num_samples=10):
        X_test_preprocessed, y_test = self.preprocess(x_test, y_test)
        predictions = self._predictor.predict(X_test_preprocessed)

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Accuracy: {accuracy_score(y_test, predictions):.2%})')
        plt.show()

        # Sample Predictions
        sample_indices = np.random.choice(len(x_test), size=num_samples, replace=False)
        plt.figure(figsize=(num_samples * 1.5, 2))
        for i, idx in enumerate(sample_indices):
            plt.subplot(1, num_samples, i + 1)
            img = x_test[idx].squeeze() if x_test[idx].ndim == 3 else x_test[idx]
            plt.imshow(img, cmap='gray')
            plt.title(f"T:{y_test[idx]}\nP:{predictions[idx]}", fontsize=8)
            plt.axis('off')
        plt.suptitle("Sample Predictions")
        plt.tight_layout()
        plt.show()
