import numpy as np
from sklearn.model_selection import train_test_split
import math
import operator


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    data = np.loadtxt(fname='Tictactoe/tic-tac-toe.data.txt',dtype='string', delimiter=',')
    print len(data)

    dataXY = np.zeros((958, 10))

    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 'x':
                dataXY[i][j] = 1
            if data[i][j] == 'o':
                dataXY[i][j] = -1
            if data[i][j] == 'b':
                dataXY[i][j] = 0
            if data[i][j] == 'positive':
                dataXY[i][j] = 1
            if data[i][j] == 'negative':
                dataXY[i][j] = -1

    trainingSet, testSet = train_test_split(dataXY, test_size=0.2)

    foldNum = 5
    bestC = -1.0
    bestAcc = 0.0

    for i in [1, 3, 5, 7]:
        print "For k = " + repr(i)

        sum_trainAcc = 0.0
        sum_Acc = 0.0

        # foldNum-fold cross validation
        for k in range(foldNum):

            if k == 0:
                # print k
                trainStack = trainingSet[len(trainingSet) / foldNum:, :]
                validationStack = trainingSet[:len(trainingSet) / foldNum, :]
                # print 'Train set: ' + repr(len(trainStack))
                # print 'Test set: ' + repr(len(validationStack))
            else:
                if k == foldNum - 1:
                    trainStack = trainingSet[: k * (len(trainingSet) / foldNum), :]
                    validationStack = trainingSet[k * (len(trainingSet) / foldNum):, :]
                    # print 'Train set: ' + repr(len(trainStack))
                    # print 'Test set: ' + repr(len(validationStack))
                else:
                    # print k
                    trainStack = np.vstack((trainingSet[0:k * (len(trainingSet) / foldNum), 0:10],
                                            trainingSet[(k + 1) * (len(trainingSet) / foldNum):, 0:10]))
                    validationStack = trainingSet[
                                      k * (len(trainingSet) / foldNum):(k + 1) * (len(trainingSet) / foldNum), 0:10]
                    # print 'Train set: ' + repr(len(trainStack))
                    # print 'Test set: ' + repr(len(validationStack))

            # Training Error
            predictions = []
            for x in range(len(trainStack)):
                neighbors = getNeighbors(trainStack, trainStack[x], i)
                result = getResponse(neighbors)
                predictions.append(result)
                # print('> predicted=' + repr(result) + ', actual=' + repr(trainStack[x][-1]))
            accuracy = getAccuracy(trainStack, predictions)
            # print('Training Accuracy: ' + repr(accuracy) + '%')
            trainAcc = accuracy
            sum_trainAcc += trainAcc

            # Validation
            predictions = []
            for x in range(len(validationStack)):
                neighbors = getNeighbors(trainStack, validationStack[x], i)
                result = getResponse(neighbors)
                predictions.append(result)
                # print('> predicted=' + repr(result) + ', actual=' + repr(validationStack[x][-1]))
            accuracy = getAccuracy(validationStack, predictions)
            # print('Validation Accuracy: ' + repr(accuracy) + '%')
            crossAcc = accuracy
            sum_Acc += crossAcc

        avg_trainAcc = sum_trainAcc / foldNum
        avg_Acc = sum_Acc / foldNum

        print "avg training acc = " + repr(avg_trainAcc)
        print "avg validation acc = " + repr(avg_Acc)

        if avg_Acc > bestAcc:
            bestAcc = avg_Acc
            asso_trainAcc = avg_trainAcc
            bestC = i

    print "Best k = " + repr(bestC)
    print "Best associated training acc = " + repr(asso_trainAcc)
    print "Best validation acc = " + repr(bestAcc)

    # generate predictions
    predictions = []
    k = bestC
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Test Accuracy: ' + repr(accuracy) + '%')


main()

