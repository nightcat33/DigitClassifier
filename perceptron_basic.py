import time
import numpy as np 
import matplotlib.pyplot as plt
import math
import random

class DigitClassifier:

    def __init__(self):

        self.trainSet = []
        self.testSet = []

        self.weights_zeros = []
        for i in range(10):
            curr_weight = [0.0 for i in range(28*28)]
            curr_weight = np.array(curr_weight)
            self.weights_zeros.append(curr_weight)

        self.confusionMatrix = [
            [0 for i in range(10)]
            for i in range(10)
        ]

        self.begin = None

        self.b = random.random()

        self.train_epochs = []
        self.train_accuracy = []
        self.weights_random = []
        mean = 0
        std = 1
        np.random.seed(0)
        for i in range(10):
            weight = np.random.normal(mean, std, 28*28)
            self.weights_random.append(weight)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


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
            self.trainSet.append([sample, digit])


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
            self.testSet.append([sample, digit])

    def predict(self, sample):

        arg_max = -math.inf
        predicted_class = None

        for i in range(10):
            # activation = np.dot(self.weights_zeros[i], sample)
            # activation = self.sigmoid(np.dot(self.weights_random[i], sample) + self.b)
            activation = np.dot(self.weights_random[i], sample) + self.b
            if activation >= arg_max:
                arg_max = activation
                predicted_class = i

        return predicted_class

    def learning_rate_decay(self):

        return 1000.0 / (1000 + time.time() - self.begin)

    def train(self, epochs):
        idx = 0
        self.begin = time.time()
        for each in range(epochs):

            for sample, label in self.trainSet:
                print(idx)
                y_predicted = self.predict(sample)

                if y_predicted != label:
                    # alpha = min((np.dot((self.weights_zeros[y_predicted] - self.weights_zeros[label]), sample) + 1) / np.dot(2*sample, sample), 1.0)
                    # alpha = self.learning_rate_decay()
                    # alpha = min((np.dot((self.weights_random[y_predicted] - self.weights_random[label]), sample) + 1) / np.dot(2*sample, sample), 0.1)
                    alpha = 0.35
                    # self.weights_zeros[label] += alpha * sample
                    # self.weights_zeros[y_predicted] -= alpha * sample
                    self.weights_random[label] += alpha * sample
                    self.weights_random[y_predicted] -= alpha * sample


                idx += 1
            correct = 0
            total = 0
            for sample, label in self.trainSet:
                y_predicted = self.predict(sample)
                total += 1
                if y_predicted == label:
                    correct += 1
            accuracy = float(correct) / total

            self.train_accuracy.append(accuracy)
            self.train_epochs.append(each)



    def evaluation(self):
        correct = 0
        total = 0
        for sample, label in self.testSet:
                        
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

    def draw_curve(self):

        plt.plot(self.train_epochs, self.train_accuracy, 'r')
        plt.show()

    def visualize_weights(self):
        self.weights_2d = [
            [
                [None for i in range(28)]
                for i in range(28)
            ]   
            for i in range(10)
        ]

        for i in range(10):
            for j in range(28):
                for k in range(28):
                    self.weights_2d[i][j][k] = self.weights_random[i][j*28+k]

        figures =  [ None for i in range(10) ]
        axes =  [ None for i in range(10) ]
        colormaps = [None for i in range(10)]

        for i in range(10):
            figures[i], axes[i] = plt.subplots()  
            axes[i].invert_yaxis()
            colormaps[i] = axes[i].pcolor(self.weights_2d[i], cmap='gist_ncar')
            plt.colorbar(colormaps[i])
            plt.show()

        for i in range(10):
            print(self.weights_2d[i])
            print()


if __name__ == '__main__':

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')

    classifier = DigitClassifier()
    classifier.load(X_train, y_train, X_test, y_test)

    classifier.train(100)
    classifier.evaluation()
    classifier.draw_curve()

    classifier.visualize_weights()

# dacay learning rate, zeros weights, no bias, 100 epochs, accuracy = 0.806
# learning rate = 0.35, zeros weights, no bias, 100 epochs, accuracy = 0.808
# dacay learning rate, random weights, with bias, 100 epochs, accuracy = 0.812
# learning rate = 0.35, random weights, with bias, 100 epochs, accuracy = 0.819

# confusion matrix
# [85, 0, 1, 0, 0, 0, 1, 0, 2, 1]
# [0, 106, 0, 0, 1, 0, 1, 0, 0, 0]
# [0, 3, 81, 4, 2, 1, 3, 3, 6, 0]
# [0, 0, 3, 78, 0, 6, 2, 6, 5, 0]
# [0, 0, 1, 1, 89, 0, 3, 3, 2, 8]
# [2, 0, 2, 3, 1, 66, 1, 4, 11, 2]
# [1, 1, 2, 0, 2, 2, 78, 2, 3, 0]
# [0, 4, 4, 1, 3, 0, 0, 83, 1, 10]
# [0, 2, 6, 6, 2, 3, 2, 1, 78, 3]
# [0, 1, 0, 5, 10, 1, 0, 6, 2, 75]

# with sigmoid, accuracy = 0.707

