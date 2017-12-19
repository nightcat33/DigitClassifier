
import numpy as np
from collections import Counter
import csv
import time

class nearestNeighborClassifier(object):

    def __init__(self):

        self.trainSet = []
        self.testSet = []

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

    def similarity(self, x_train, x_test):
        diff = 0
        for i in range(28*28):
            if x_train[i] != x_test[i]:
                diff += 1
        return diff

    def predict(self, x_test, k):

        distances = []
        targets = []
        idx = 0
        for sample, _ in self.trainSet:
            diff = self.similarity(sample, x_test)
            distances.append([diff, idx])
            idx += 1

        distances = sorted(distances)

        for i in range(k):
            idx = distances[i][1]
            targets.append(self.trainSet[idx][1])

        return Counter(targets).most_common(1)[0][0]

    def evaluation(self, k):
        correct = 0
        total = 0
        idx = 0
        start = time.time()
        for sample, label in self.testSet:
            y_predict = self.predict(sample, k)

            if y_predict == label:
                correct += 1
            total += 1
            print(y_predict, label)

        accuracy = float(correct) / total
        print("accuracy %f." % accuracy)
        end = time.time() - start
        return accuracy, end



if __name__ == '__main__':

    X_train = open('./digitdata/trainingimages', 'r') 
    y_train = open('./digitdata/traininglabels', 'r')
    X_test = open('./digitdata/testimages', 'r')
    y_test = open('./digitdata/testlabels', 'r')

    classifier = nearestNeighborClassifier()
    classifier.load(X_train, y_train, X_test, y_test)

    for k in range(20, 50):
        accuracy, time_cost = classifier.evaluation(k)
        row = [accuracy, time_cost, k]
        with open('nearest_neighbor.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)




