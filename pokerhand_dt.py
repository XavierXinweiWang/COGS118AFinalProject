import numpy as np
from sklearn import tree

data = np.loadtxt(fname='PokerHand/poker-hand-training-true.data.txt', dtype='float', delimiter=',')
test = np.loadtxt(fname='PokerHand/poker-hand-testing.data.txt', dtype='float', delimiter=',')

print len(data)
print len(data[0])
print len(test)
print len(test[0])
print data[100]
print test[100]

trainingSet = np.zeros((25010, 26))
testSet = np.zeros((1000000, 26))


for i in range(len(data)):
    for j in range(len(data[0])):
        if j == 0 or j == 2 or j == 4 or j == 6 or j == 8:
            if data[i][j] == 1:
                trainingSet[i][5*j/2] = 1
            if data[i][j] == 2:
                trainingSet[i][5*j/2+1] = 1
            if data[i][j] == 3:
                trainingSet[i][5*j/2+2] = 1
            if data[i][j] == 4:
                trainingSet[i][5*j/2+3] = 1
        else:
            if j == 10:
                if data[i][j] > 0:
                    trainingSet[i][25] = 1
                else:
                    trainingSet[i][25] = -1
            else:
                trainingSet[i][5*(j+1)/2-1] = data[i][j]

print trainingSet[100]

for i in range(len(test)):
    for j in range(len(test[0])):
        if j == 0 or j == 2 or j == 4 or j == 6 or j == 8:
            if test[i][j] == 1:
                testSet[i][5*j/2] = 1
            if test[i][j] == 2:
                testSet[i][5*j/2+1] = 1
            if test[i][j] == 3:
                testSet[i][5*j/2+2] = 1
            if test[i][j] == 4:
                testSet[i][5*j/2+3] = 1
        else:
            if j == 10:
                if test[i][j] > 0:
                    testSet[i][25] = 1
                else:
                    testSet[i][25] = -1
            else:
                testSet[i][5*(j+1)/2-1] = test[i][j]

print testSet[100]

foldNum = 5
i = 1
bestC = -1.0
bestAcc = 0.0

while i < 15:
    # print "For D = "
    # print i
    clf = tree.DecisionTreeClassifier(max_depth=i)
    sum_trainAcc = 0.0
    sum_Acc = 0.0

    # foldNum-fold cross validation
    for k in range(foldNum):

        if k == 0:
            # print k
            clf.fit(trainingSet[len(trainingSet)/foldNum:, 0:25], trainingSet[len(trainingSet)/foldNum:, 25])
            # print trainingSet[len(trainingSet)/foldNum:, 0:25]

            trainResult = clf.predict(trainingSet[len(trainingSet) / foldNum:, 0:25])
            correct_num = 0.0
            for j in range(len(trainResult)):
                if trainResult[j] == trainingSet[len(trainingSet)/foldNum + j, 25]:
                    correct_num += 1.0
            trainAcc = correct_num / len(trainResult)
            sum_trainAcc += trainAcc

            crossResult = clf.predict(trainingSet[:len(trainingSet)/foldNum, 0:25])
            # print crossResult
            correct_num = 0.0
            for j in range(len(crossResult)):
                if crossResult[j] == trainingSet[j, 25]:
                    correct_num += 1.0
            crossAcc = correct_num / len(crossResult)
            sum_Acc += crossAcc
            # print crossResult
            # print crossAcc
            # print trainAcc
        else:
            if k == foldNum - 1:
                # print k
                clf.fit(trainingSet[: k * (len(trainingSet) / foldNum), 0:25], trainingSet[: k * (len(trainingSet) / foldNum), 25])
                # print len(trainingSet[: k * (len(trainingSet) / foldNum), 0:25])

                trainResult = clf.predict(trainingSet[: k * (len(trainingSet) / foldNum), 0:25])
                correct_num = 0.0
                for j in range(len(trainResult)):
                    if trainResult[j] == trainingSet[j, 25]:
                        correct_num += 1.0
                trainAcc = correct_num / len(trainResult)
                sum_trainAcc += trainAcc

                crossResult = clf.predict(trainingSet[k * (len(trainingSet) / foldNum):, 0:25])
                correct_num = 0.0
                for j in range(len(crossResult)):
                    if crossResult[j] == trainingSet[j + k * (len(trainingSet) / foldNum), 25]:
                        correct_num += 1.0
                crossAcc = correct_num / len(crossResult)
                sum_Acc += crossAcc
                # print crossResult
                # print crossAcc
                # print trainAcc
            else:
                # print k
                trainStack = np.vstack((trainingSet[0:k*(len(trainingSet)/foldNum), 0:26], trainingSet[(k+1)*(len(trainingSet)/foldNum):, 0:26]))
                validationStack = trainingSet[k*(len(trainingSet)/foldNum):(k+1)*(len(trainingSet)/foldNum), 0:26]
                # print len(trainStack)
                clf.fit(trainStack[:, 0:25], trainStack[:, 25])

                trainResult = clf.predict(trainStack[:, 0:25])
                correct_num = 0.0
                for j in range(len(trainResult)):
                    if trainResult[j] == trainStack[j, 25]:
                        correct_num += 1.0
                trainAcc = correct_num / len(trainResult)
                sum_trainAcc += trainAcc

                crossResult = clf.predict(validationStack[:, 0:25])
                correct_num = 0.0
                for j in range(len(crossResult)):
                    if crossResult[j] == trainingSet[j + k * (len(trainingSet) / foldNum), 25]:
                        correct_num += 1.0
                crossAcc = correct_num/len(crossResult)
                sum_Acc += crossAcc
                # print crossResult
                # print crossAcc
                # print trainAcc

    avg_Acc = sum_Acc/foldNum
    avg_trainAcc = sum_trainAcc / foldNum
    # print "acc = "
    # print avg_Acc

    if avg_Acc > bestAcc:
        bestAcc = avg_Acc
        asso_trainAcc = avg_trainAcc
        bestC = i

    i += 1

print "The best D is"
print bestC
print "The best acc is"
print bestAcc
print "The associated training acc is"
print asso_trainAcc

clf = tree.DecisionTreeClassifier(max_depth=bestC)
clf.fit(trainingSet[:, 0:25], trainingSet[:, 25])
testResult = clf.predict(testSet[:, 0:25])

correct_num = 0.0
for j in range(len(testResult)):
    if testResult[j] == testSet[j, 25]:
        correct_num += 1.0
testAcc = correct_num / len(testResult)
print "Test Result"
print testAcc
