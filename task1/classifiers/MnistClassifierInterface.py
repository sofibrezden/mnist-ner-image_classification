from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    @abstractmethod
    def preprocess(self, x_test, y_test):
        pass

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def validate(self, x_test, y_test):
        pass

    @abstractmethod
    def visualize_results(self, x_test, y_test):
        pass
