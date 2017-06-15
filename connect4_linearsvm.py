import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

data = np.loadtxt(fname='connect-4.data', dtype='string', delimiter=',')
print len(data)
print len(data[0])

dataXY = np.zeros((67557, 43))

for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i][j] == 'x':
            dataXY[i][j] = 1
        if data[i][j] == 'o':
            dataXY[i][j] = -1
        if data[i][j] == 'b':
            dataXY[i][j] = 0
        if data[i][j] == 'win':
            dataXY[i][j] = 1
        if data[i][j] == 'loss' or data[i][j] == 'draw':
            dataXY[i][j] = -1

trainingSet, testSet = train_test_split(dataXY, test_size=0.2)

foldNum = 5
i = 1.0
bestC = -1.0
bestAcc = 0.0

while i <= 100.0:
    # print "For C = "
    # print i
    clf = svm.SVC(C=i, kernel='linear')
    sum_Acc = 0.0

    # foldNum-fold cross validation
    for k in range(foldNum):

        if k == 0:
            # print k
            clf.fit(trainingSet[len(trainingSet)/foldNum:, 0:42], trainingSet[len(trainingSet)/foldNum:, 42])
            # print trainingSet[len(trainingSet)/foldNum:, 0:42]
            crossResult = clf.predict(trainingSet[:len(trainingSet)/foldNum:, 0:42])
            correct_num = 0.0
            for j in range(len(crossResult)):
                if crossResult[j] == trainingSet[j, 42]:
                    correct_num += 1.0
            crossAcc = correct_num / len(crossResult)
            sum_Acc += crossAcc
            # print crossResult
            # print crossAcc
        else:
            if k == foldNum - 1:
                # print k
                clf.fit(trainingSet[: k * (len(trainingSet) / foldNum), 0:42], trainingSet[: k * (len(trainingSet) / foldNum), 42])
                # print len(trainingSet[: k * (len(trainingSet) / foldNum), 0:42])

                crossResult = clf.predict(trainingSet[k * (len(trainingSet) / foldNum):, 0:42])
                correct_num = 0.0
                for j in range(len(crossResult)):
                    if crossResult[j] == trainingSet[j + k * (len(trainingSet) / foldNum), 42]:
                        correct_num += 1.0
                crossAcc = correct_num / len(crossResult)
                sum_Acc += crossAcc
                # print crossResult
                # print crossAcc
            else:
                # print k
                trainStack = np.vstack((trainingSet[0:k*(len(trainingSet)/foldNum), 0:43], trainingSet[(k+1)*(len(trainingSet)/foldNum):, 0:43]))
                validationStack = trainingSet[k*(len(trainingSet)/foldNum):(k+1)*(len(trainingSet)/foldNum), 0:43]
                # print len(trainStack)

                clf.fit(trainStack[:, 0:42], trainStack[:, 42])
                crossResult = clf.predict(validationStack[:, 0:42])
                correct_num = 0.0
                for j in range(len(crossResult)):
                    if crossResult[j] == trainingSet[j + k * (len(trainingSet) / foldNum), 42]:
                        correct_num += 1.0
                crossAcc = correct_num/len(crossResult)
                sum_Acc += crossAcc
                # print crossResult
                # print crossAcc

    avg_Acc = sum_Acc/foldNum
    # print "acc = "
    # print avg_Acc

    if avg_Acc > bestAcc:
        bestAcc = avg_Acc
        bestC = i

    i += 1.0

print "\nThe best C is"
print bestC
print "The best acc is"
print bestAcc

clf = svm.SVC(C=bestC, kernel='linear')
clf.fit(trainingSet[:, 0:42], trainingSet[:, 42])
testResult = clf.predict(testSet[:, 0:42])

correct_num = 0.0
for j in range(len(testResult)):
    if testResult[j] == testSet[j, 42]:
        correct_num += 1.0
testAcc = correct_num / len(testResult)
print "Test Accuracy"
print testAcc
