import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score)

# encoding class
def encode_class(mydata):
    classes = []
    for i in range(len(mydata)):
        if mydata[i][-1] not in classes:
            classes.append(mydata[i][-1])
    for i in range(len(classes)):
        for j in range(len(mydata)):
            if mydata[j][-1] == classes[i]:
                mydata[j][-1] = i
    return mydata

# splitting the data
def splitting(mydata, ratio):
    train_num = int(len(mydata) * ratio)
    train = []
    test = list(mydata)
    
    while len(train) < train_num:
        index = random.randrange(len(test))
        train.append(test.pop(index))
    return train, test

# group classing
def groupUnderClass(mydata):
    data_dict = {}
    for i in range(len(mydata)):
        if mydata[i][-1] not in data_dict:
            data_dict[mydata[i][-1]] = []
        data_dict[mydata[i][-1]].append(mydata[i])
    return data_dict

# calculate mean and standard deviation
def MeanAndStdDev(numbers):
    avg = np.mean(numbers)
    stddev = np.std(numbers)
    return avg, stddev

# calculate mean and std dev for each class
def MeanAndStdDevForClass(mydata):
    info = {}
    data_dict = groupUnderClass(mydata)
    for classValue, instances in data_dict.items():
        info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
    return info

# calculate gaussian probability
def calculateGaussianProbability(x, mean, stdev):
    epsilon = 1e-10
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
    return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo

# calculate class probabilities
def calculateClassProbabilities(info, test):
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1.0
        for i in range(len(classSummaries) - 1): # Exclude the class label column
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
    return probabilities

# predicting for test set
def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel = max(probabilities, key=probabilities.get)
    return bestLabel

def getPredictions(info, test):
    return [predict(info, instance) for instance in test]

# calculate accuracy
def accuracy_rate(test, predictions):
    correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
    return (correct / float(len(test))) * 100.0


# loading data and preprocessing data
if __name__ == '__main__':
    filename = 'diabetes.csv' 
    df = pd.read_csv(filename, comment='#')
    mydata = df.values.tolist()

    mydata = encode_class(mydata)
    for i in range(len(mydata)):
        for j in range(len(mydata[i]) - 1):
            mydata[i][j] = float(mydata[i][j])

    # splits the data into training and testing sets
    ratio = 0.7
    train_data, test_data = splitting(mydata, ratio)

    print('Total number of examples:', len(mydata))
    print('Training examples:', len(train_data))
    print('Test examples:', len(test_data))

    # training and testing model
    info = MeanAndStdDevForClass(train_data)
    predictions = getPredictions(info, test_data)
    accuracy = accuracy_rate(test_data, predictions)
    print(f'Accuracy of the model: {accuracy:.2f}%')

    # evaluating model with confusion matrix
    y_true = [row[-1] for row in test_data]
    y_pred = predictions

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # precision, recall and f1 score
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # plotting metrics bar chart
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]

    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1)
    plt.title('Precision, Recall, and F1 Score')
    plt.ylabel('Score')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()
