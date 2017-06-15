import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree


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
i = 1
bestC = -1.0
bestAcc = 0.0

while i <= 9:
    # print "For D = "
    # print i
    clf = tree.DecisionTreeClassifier(max_depth=i)
    sum_trainAcc = 0.0
    sum_Acc = 0.0

    # foldNum-fold cross validation
    for k in range(foldNum):

        if k == 0:
            # print k
            clf.fit(trainingSet[len(trainingSet)/foldNum:, 0:9], trainingSet[len(trainingSet)/foldNum:, 9])
            # print trainingSet[len(trainingSet)/foldNum:, 0:9]

            trainResult = clf.predict(trainingSet[len(trainingSet) / foldNum:, 0:9])
            correct_num = 0.0
            for j in range(len(trainResult)):
                if trainResult[j] == trainingSet[len(trainingSet)/foldNum + j, 9]:
                    correct_num += 1.0
            trainAcc = correct_num / len(trainResult)
            sum_trainAcc += trainAcc

            crossResult = clf.predict(trainingSet[:len(trainingSet)/foldNum, 0:9])
            # print crossResult
            correct_num = 0.0
            for j in range(len(crossResult)):
                if crossResult[j] == trainingSet[j, 9]:
                    correct_num += 1.0
            crossAcc = correct_num / len(crossResult)
            sum_Acc += crossAcc
            # print crossResult
            # print crossAcc
            # print trainAcc
        else:
            if k == foldNum - 1:
                # print k
                clf.fit(trainingSet[: k * (len(trainingSet) / foldNum), 0:9], trainingSet[: k * (len(trainingSet) / foldNum), 9])
                # print len(trainingSet[: k * (len(trainingSet) / foldNum), 0:10])

                trainResult = clf.predict(trainingSet[: k * (len(trainingSet) / foldNum), 0:9])
                correct_num = 0.0
                for j in range(len(trainResult)):
                    if trainResult[j] == trainingSet[j, 9]:
                        correct_num += 1.0
                trainAcc = correct_num / len(trainResult)
                sum_trainAcc += trainAcc

                crossResult = clf.predict(trainingSet[k * (len(trainingSet) / foldNum):, 0:9])
                correct_num = 0.0
                for j in range(len(crossResult)):
                    if crossResult[j] == trainingSet[j + k * (len(trainingSet) / foldNum), 9]:
                        correct_num += 1.0
                crossAcc = correct_num / len(crossResult)
                sum_Acc += crossAcc
                # print crossResult
                # print crossAcc
                # print trainAcc
            else:
                # print k
                trainStack = np.vstack((trainingSet[0:k*(len(trainingSet)/foldNum), 0:10], trainingSet[(k+1)*(len(trainingSet)/foldNum):, 0:10]))
                validationStack = trainingSet[k*(len(trainingSet)/foldNum):(k+1)*(len(trainingSet)/foldNum), 0:10]
                # print len(trainStack)
                clf.fit(trainStack[:, 0:9], trainStack[:, 9])

                trainResult = clf.predict(trainStack[:, 0:9])
                correct_num = 0.0
                for j in range(len(trainResult)):
                    if trainResult[j] == trainStack[j, 9]:
                        correct_num += 1.0
                trainAcc = correct_num / len(trainResult)
                sum_trainAcc += trainAcc

                crossResult = clf.predict(validationStack[:, 0:9])
                correct_num = 0.0
                for j in range(len(crossResult)):
                    if crossResult[j] == trainingSet[j + k * (len(trainingSet) / foldNum), 9]:
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
clf.fit(trainingSet[:, 0:9], trainingSet[:, 9])
testResult = clf.predict(testSet[:, 0:9])

correct_num = 0.0
for j in range(len(testResult)):
    if testResult[j] == testSet[j, 9]:
        correct_num += 1.0
testAcc = correct_num / len(testResult)
print "Test Result"
print testAcc
