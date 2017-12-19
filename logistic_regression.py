import time
import numpy as np 
import matplotlib.pyplot as plt
import math

# label is converted in to one-hot label with 10 outputs
# using softmax as activation function

def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T


class DigitClassifier:

    def __init__(self):

        self.trainSet = []
        self.testSet = []

        self.weights_zeros = np.zeros((28*28, 10))
        self.b = np.zeros(10)

        self.confusionMatrix = [
            [0 for i in range(10)]
            for i in range(10)
        ]

        self.begin = None

    def load(self, X_train, y_train, X_test, y_test):
        for label in y_train:
            digit = int(label.strip())
            sample = []

            for i in range(28):
                line = X_train.readline()
                for j in range(28):
                    f = None
                    if line[j] == ' ':
                        f = 0
                    else:
                        f = 1
                    sample.append(f)
            sample = np.array(sample)
            label = np.array([0 for i in range(10)])
            label[digit] = 1
            self.trainSet.append([sample, label])


        for label in y_test:
            digit = int(label.strip())
            sample = []

            for i in range(28):
                line = X_test.readline()
                for j in range(28):
                    f = None
                    if line[j] == ' ':
                        f = 0
                    else:
                        f = 1
                    sample.append(f)
            sample = np.array(sample)
            label = np.array([0 for i in range(10)])
            label[digit] = 1
            self.testSet.append([sample, label])

    def predict(self, sample):

        predicted_class = None

        softmax_activation = softmax(np.dot(sample, self.weights_zeros) + self.b)
        predicted_class = np.argmax(softmax_activation)
        
        return predicted_class

    def learning_rate_decay(self):

        return 1000.0 / (1000 + time.time() - self.begin)

    def train(self, learning_rate, epochs):
        idx = 0
        self.begin = time.time()
        for _ in range(epochs):

            for sample, label in self.trainSet:
                print(idx)
                
                softmax_activation = softmax(np.dot(sample, self.weights_zeros) + self.b)
                delta_y = label - softmax_activation

                # reshape
                reshape_delta_y = []
                for i in range(10):
                    row = delta_y[i]
                    reshape_delta_y.append([row])
                reshape_delta_y = np.array(reshape_delta_y)

                reshape_sample = []
                for i in range(28*28):
                    row = sample[i]
                    reshape_sample.append([row])
                reshape_sample = np.array(reshape_sample)

                self.weights_zeros += learning_rate * np.dot(reshape_sample, reshape_delta_y.T)
                self.b += learning_rate * np.mean(delta_y, axis=0)

                idx += 1

    def evaluation(self):
        correct = 0
        total = 0
        for sample, label in self.testSet:
            for i in range(10):
                if label[i] == 1:
                    label = i
                    break
            
            y_predicted = self.predict(sample)
            total += 1
            if y_predicted == label:
                correct += 1

            self.confusionMatrix[label][y_predicted] += 1

        accuracy = float(correct) / total

        print("Accuracy %f" % accuracy)
        print("Confusion Matrix:")
        for i in range(10):
            print(self.confusionMatrix[i])



if __name__ == '__main__':

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')

    classifier = DigitClassifier()
    classifier.load(X_train, y_train, X_test, y_test)

    classifier.train(0.01, 100)
    classifier.evaluation()

# Accuracy 0.825000, learning rate = 0.01, epoches = 100


