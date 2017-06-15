import numpy as np
from sklearn import svm

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
# trainingSet = trainingSet[:500, :]
# testSet = testSet[:100, :]
#
# print len(trainingSet)
# print len(trainingSet[0])
# print len(testSet)
# print len(testSet[0])

foldNum = 5
i = 1.0
bestC = -1.0
bestGamma = -1.0
bestAcc = 0.0
gamma = 0.1
len_attr = 26


while gamma < 1.0:

    while i <= 10.0:
        # print "For C = "
        # print i
        clf = svm.SVC(C=i, kernel='rbf', gamma=gamma)
        sum_Acc = 0.0

        # foldNum-fold cross validation
        for k in range(foldNum):

            if k == 0:
                print k
                clf.fit(trainingSet[len(trainingSet)/foldNum:, 0:len_attr-1], trainingSet[len(trainingSet)/foldNum:, len_attr-1])
                # print trainingSet[len(trainingSet)/foldNum:, 0:len_attr-1]
                crossResult = clf.predict(trainingSet[:len(trainingSet)/foldNum:, 0:len_attr-1])
                correct_num = 0.0
                for j in range(len(crossResult)):
                    if crossResult[j] == trainingSet[j, len_attr-1]:
                        correct_num += 1.0
                crossAcc = correct_num / len(crossResult)
                sum_Acc += crossAcc
                # print crossResult
                print crossAcc
            else:
                if k == foldNum - 1:
                    print k
                    clf.fit(trainingSet[: k * (len(trainingSet) / foldNum), 0:len_attr-1], trainingSet[: k * (len(trainingSet) / foldNum), len_attr-1])
                    # print len(trainingSet[: k * (len(trainingSet) / foldNum), 0:len_attr-1])

                    crossResult = clf.predict(trainingSet[k * (len(trainingSet) / foldNum):, 0:len_attr-1])
                    correct_num = 0.0
                    for j in range(len(crossResult)):
                        if crossResult[j] == trainingSet[j + k * (len(trainingSet) / foldNum), len_attr-1]:
                            correct_num += 1.0
                    crossAcc = correct_num / len(crossResult)
                    sum_Acc += crossAcc
                    # print crossResult
                    print crossAcc
                else:
                    print k
                    trainStack = np.vstack((trainingSet[0:k*(len(trainingSet)/foldNum), 0:len_attr], trainingSet[(k+1)*(len(trainingSet)/foldNum):, 0:len_attr]))
                    validationStack = trainingSet[k*(len(trainingSet)/foldNum):(k+1)*(len(trainingSet)/foldNum), 0:len_attr]
                    # print len(trainStack)

                    clf.fit(trainStack[:, 0:len_attr-1], trainStack[:, len_attr-1])
                    crossResult = clf.predict(validationStack[:, 0:len_attr-1])
                    correct_num = 0.0
                    for j in range(len(crossResult)):
                        if crossResult[j] == trainingSet[j + k * (len(trainingSet) / foldNum), len_attr-1]:
                            correct_num += 1.0
                    crossAcc = correct_num/len(crossResult)
                    sum_Acc += crossAcc
                    # print crossResult
                    print crossAcc

        avg_Acc = sum_Acc/foldNum
        print "acc = "
        print avg_Acc

        if avg_Acc > bestAcc:
            bestAcc = avg_Acc
            bestC = i
            bestGamma = gamma

        i += 1.0
    gamma += 0.1
    i = 1.0

print "\nThe best C is"
print bestC
print "The best gamma is"
print bestGamma
print "The best acc is"
print bestAcc

clf = svm.SVC(C=bestC, kernel='rbf', gamma=gamma)
clf.fit(trainingSet[:, 0:len_attr-1], trainingSet[:, len_attr-1])
testResult = clf.predict(testSet[:, 0:len_attr-1])

correct_num = 0.0
for j in range(len(testResult)):
    if testResult[j] == testSet[j, len_attr-1]:
        correct_num += 1.0
testAcc = correct_num / len(testResult)
print "Test Accuracy"
print testAcc
