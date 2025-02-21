from task1.classifiers.CNNClassifier import CNNClassifier
from task1.classifiers.RFClassifier import RFClassifier
from task1.classifiers.Feed_forward_nn_Classifier import Feed_Forward_nn_Classifier


class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.model = RFClassifier()
        elif algorithm == 'cnn':
            self.model = CNNClassifier()
        else:
            self.model = Feed_Forward_nn_Classifier()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def validate(self, X_val, y_val):
        return self.model.validate(X_val, y_val)

    def visualize_results(self, x_test, y_test):
        return self.model.visualize_results(x_test, y_test, num_samples=10)
