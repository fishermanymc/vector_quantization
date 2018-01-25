import pca
import numpy as np
import matplotlib.pyplot as plt
import pickle


def extractData(pid=0):
    # this hard-coded function extract all the data about a given pid
    # the result is an n * m numpy matrix, where n is the number of gradients
    # and m is the number of iterations
    X = []
    fileHeader = 'phase1/iter000'
    pidName = '_pid0' + str(pid) + '.bin'  # hard coded file name
    for i in range(1, 301):
        fileName = fileHeader + '0' * (3 - len(str(i))) + str(i) + pidName
        temp = np.fromfile(fileName, dtype=np.float32)
        X.append(temp)
    return np.transpose(np.matrix(X))


def coreTest(X, sampleSize=150, lossThres=0.05):
    # This function train a PCA using every 'sampleSize' samples, and use the
    # resulted compressor to reduce the dimension of the next 'samplesSize'
    # samples.
    # Inputs:
    #    samplesSize: the training and testing size
    #    lossThres: the variance loss rate, higher thres will yield higher
    #               compression.
    # Outputs:
    #    lossLog: the square error of each decompressed sample set v.s. origin
    #    normLog: the sum square of each original sample set
    #    dLog: the compressed dimension of each sample set
    XTrain = X[:, :sampleSize]  # the first set of samples
    lossLog = []
    normLog = []
    dLog = []
    ULog = []
    for i in range(int(np.floor(X.shape[1] / sampleSize)) - 1):
        # centralize the training set
        XTrainCentered, meanGrad = pca.centerilze(XTrain)
        # SVD the centralized training set
        U, S = pca.svd(XTrainCentered)
        # generate compressor
        Uc, d = pca.generateCompressor(U, S, thres=lossThres)

        # obtain the testing set
        XTest = X[:, (i + 1) * sampleSize: (i + 2) * sampleSize]
        # compress it
        XTestComp = pca.compress(XTest, meanGrad, Uc)
        # decompress
        XTestDecomp = pca.decompress(XTestComp, meanGrad, Uc)

        # performance metrices
        loss, norm = pca.absLoss(XTest, XTestDecomp)
        lossLog.append(loss)
        normLog.append(norm)
        dLog.append(d)
        ULog.append(U)

        # prepare for the next iteration
        # Xtrain = XTestDecomp  # Youjie, please try both.
        XTrain = XTest
    return lossLog, normLog, dLog, ULog


def doubleplot(y1, y2, sampleSize):
    # plot two y-axis in one graph.
    fig, ax1 = plt.subplots()
    x = (np.array(range(len(y1))) + 1) * sampleSize
    ax1.plot(x, y1, 'b-', linewidth=3)
    ax1.set_xlabel('Iteration index', fontsize=16)
    ax1.set_ylabel('compression ratio', color='b', fontsize=16)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(x, y2 * 100, 'r--', marker='s', linewidth=3)
    ax2.set_ylabel('testing loss (%)', color='r', fontsize=16)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.grid()
    plt.show()


def mainTest(pid=0, gradSize=1000, sampleSize=50, lossThres=0.05):
    # This main test function slices gradients, run coreTest() on each slice,
    # and aggregate the compression ratio and loss rate.
    # extract the data
    X = extractData(pid=0)
    totalGrad = X.shape[0]
    numSlice = int(np.ceil((totalGrad / gradSize)))
    totalLossLog = []  # the square error of decompression
    totalNormLog = []  # the sum square of original data
    totalDLog = []  # the number of dimensions after compression
    totalULog = {}
    for i in range(numSlice):
        # obtain one slice of the gradients
        XSlice = X[i * gradSize: (i + 1) * gradSize, :]
        # test the performance of this slice
        lossLog, normLog, dLog, ULog = coreTest(XSlice, sampleSize=sampleSize,
                                                lossThres=lossThres)
        if totalLossLog == []:  # initialize the logs if it's the first slice
            totalLossLog = np.array(lossLog)
            totalNormLog = np.array(normLog)
            totalDLog = np.array(dLog)
        else:  # add results to the log
            totalLossLog += lossLog
            totalNormLog += normLog
            totalDLog += dLog
        totalULog[i] = ULog
    totalLossRatio = totalLossLog / totalNormLog
    compRatio = totalGrad / totalDLog
    doubleplot(compRatio, totalLossRatio, sampleSize)
    result = {}
    result['compRatio'] = compRatio
    result['lossRatio'] = totalLossRatio
    result['ULog'] = totalULog
    with open('result_pid0' + str(pid) + '.pickle', 'wb') as handle:
        pickle.dump(result, handle)


# below is an example
pid = 0
gradSize = 1000  # the number of gradients to be compressed together
sampleSize = 50  # the training and testing sample size
lossThres = 0.05  # the variance loss rate. Set higher for higher compression.
mainTest(pid, gradSize, sampleSize, lossThres)
