import numpy as np
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
    data = np.loadtxt(fname='PokerHand/poker-hand-training-true.data.txt', dtype='float', delimiter=',')
    test = np.loadtxt(fname='PokerHand/poker-hand-testing.data.txt', dtype='float', delimiter=',')

    trainingSet = np.zeros((25010, 26))
    testSet = np.zeros((1000000, 26))

    for i in range(len(data)):
        for j in range(len(data[0])):
            if j == 0 or j == 2 or j == 4 or j == 6 or j == 8:
                if data[i][j] == 1:
                    trainingSet[i][5 * j / 2] = 1
                if data[i][j] == 2:
                    trainingSet[i][5 * j / 2 + 1] = 1
                if data[i][j] == 3:
                    trainingSet[i][5 * j / 2 + 2] = 1
                if data[i][j] == 4:
                    trainingSet[i][5 * j / 2 + 3] = 1
            else:
                if j == 10:
                    if data[i][j] > 0:
                        trainingSet[i][25] = 1
                    else:
                        trainingSet[i][25] = -1
                else:
                    trainingSet[i][5 * (j + 1) / 2 - 1] = data[i][j]

    for i in range(len(test)):
        for j in range(len(test[0])):
            if j == 0 or j == 2 or j == 4 or j == 6 or j == 8:
                if test[i][j] == 1:
                    testSet[i][5 * j / 2] = 1
                if test[i][j] == 2:
                    testSet[i][5 * j / 2 + 1] = 1
                if test[i][j] == 3:
                    testSet[i][5 * j / 2 + 2] = 1
                if test[i][j] == 4:
                    testSet[i][5 * j / 2 + 3] = 1
            else:
                if j == 10:
                    if test[i][j] > 0:
                        testSet[i][25] = 1
                    else:
                        testSet[i][25] = -1
                else:
                    testSet[i][5 * (j + 1) / 2 - 1] = test[i][j]

    # print len(trainingSet)
    # print len(trainingSet[0])
    # print len(testSet)
    # print len(testSet[0])
    #
    # for i in range(len(trainingSet)):
    #     if trainingSet[i][10] < 1:
    #         trainingSet[i][10] = -1
    #     else:
    #         trainingSet[i][10] = 1
    #
    # for i in range(len(testSet)):
    #     if testSet[i][10] < 1:
    #         testSet[i][10] = -1
    #     else:
    #         testSet[i][10] = 1
    #
    # np.random.shuffle(trainingSet)
    # np.random.shuffle(testSet)
    #
    # trainingSet = trainingSet[:5000, :]
    # testSet = testSet[:1000, :]
    #
    # print len(trainingSet)
    # print len(trainingSet[0])
    # print len(testSet)
    # print len(testSet[0])

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
                    # print k
                    trainStack = trainingSet[: k * (len(trainingSet) / foldNum), :]
                    validationStack = trainingSet[k * (len(trainingSet) / foldNum):, :]
                    # print 'Train set: ' + repr(len(trainStack))
                    # print 'Test set: ' + repr(len(validationStack))
                else:
                    # print k
                    trainStack = np.vstack((trainingSet[0:k * (len(trainingSet) / foldNum), 0:26],
                                            trainingSet[(k + 1) * (len(trainingSet) / foldNum):, 0:26]))
                    validationStack = trainingSet[
                                        k * (len(trainingSet) / foldNum):(k + 1) * (len(trainingSet) / foldNum), 0:26]
                    # print 'Train set: ' + repr(len(trainStack))
                    # print 'Test set: ' + repr(len(validationStack))

            # Training Error
            # predictions = []
            # for x in range(len(trainStack)):
            #     neighbors = getNeighbors(trainStack, trainStack[x], i)
            #     result = getResponse(neighbors)
            #     predictions.append(result)
            #     print('> predicted=' + repr(result) + ', actual=' + repr(trainStack[x][-1]))
            # accuracy = getAccuracy(trainStack, predictions)
            # print('Training Accuracy: ' + repr(accuracy) + '%')
            # trainAcc = accuracy
            # sum_trainAcc += trainAcc

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

        # avg_trainAcc = sum_trainAcc / foldNum
        avg_Acc = sum_Acc / foldNum

        # print "avg training acc = " + repr(avg_trainAcc)
        print "avg validation acc = " + repr(avg_Acc)

        if avg_Acc > bestAcc:
            bestAcc = avg_Acc
            # asso_trainAcc = avg_trainAcc
            bestC = i

    print "Best k = " + repr(bestC)
    # print "Best associated training acc = " + repr(asso_trainAcc)
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
