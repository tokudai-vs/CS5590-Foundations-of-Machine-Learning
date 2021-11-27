import numpy as np
from numpy.random import rand
from sklearn.metrics import accuracy_score, precision_score, recall_score


class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, epochs=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.loss = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, y_pred, y):
        return -np.mean(y * (np.log(y_pred)) - (1 - y) * np.log(1 - y_pred))

    def gradient_descent(self, x, y, y_pred):
        m = x.shape[0]
        dw = (1 / m) * np.dot(x.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        return dw, db

    def fit(self, x, y):
        m, n = x.shape
        self.weights = np.zeros((n, 1))
        b = 0
        y = y.reshape(m, 1)
        for epoch in range(self.epochs):
            y_hat = self.sigmoid(np.dot(x, self.weights) + b)
            dw, db = self.gradient_descent(x, y, y_hat)
            self.weights -= self.learning_rate * dw
            b -= self.learning_rate * db
            l = self.cross_entropy(y, self.sigmoid(np.dot(x, self.weights) + b))
            self.loss.append(l)

    def predict(self, x, b=0):
        preds = self.sigmoid(np.dot(x, self.weights) + b)
        pred_classification = [1 if i > 0.5 else 0 for i in preds]
        return np.array(pred_classification)


if __name__ == "__main__":
    train_data = [
        [0.346, 0.780, 0],
        [0.303, 0.439, 0],
        [0.358, 0.729, 0],
        [0.602, 0.863, 1],
        [0.790, 0.753, 1],
        [0.611, 0.965, 1],
    ]
    test_data = [
        [0.959, 0.382, 0],
        [0.750, 0.306, 0],
        [0.395, 0.760, 0],
        [0.823, 0.764, 1],
        [0.761, 0.874, 1],
        [0.844, 0.435, 1],
    ]

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    model = LogisticRegressionClassifier(0.01, 5)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print(accuracy_score(y_test, preds))
    print(precision_score(y_test, preds))
    print(recall_score(y_test, preds))
