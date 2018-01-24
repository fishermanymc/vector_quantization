import numpy as np
import matplotlib.pyplot as plt


def centerilze(X):
    # This function substracts the mean of each row of X
    # from the entries of the corresponding row.
    # If X is an n * m matrix, where each column is a sample,
    # then this function effectively centralizes the samples.
    meanCol = np.matrix(np.mean(X, axis=1))
    meanRepeat = np.repeat(meanCol, X.shape[1], 1)
    X = np.subtract(X, meanRepeat)
    return X, meanCol


def svd(X):
    # This function calculates the covariance matrix of X
    # and then decompose it
    # into singular vectors and signular values.
    _, c = X.shape
    covMatrix = np.dot(X, np.transpose(X)) / (c - 1)
    U, s, V = np.linalg.svd(covMatrix)
    return U, s, V


def singluarAnalysis(s, thres=0.05):
    # Given a signluar value vector and a threshold, this function finds the
    # number of largest singular values whose sum is more than $1-thres$
    # of the total singular values.
    d = len(s)
    cumu = 0
    sumThres = sum(s) * (1 - thres)
    for i in range(d):
        cumu += s[i]
        if cumu > sumThres:
            break
    return i + 1, d / (i + 1)


def getRatio(pid=0, numIter=300, maxPara=1000, thres=0.05):
    # This function is currently hard coded.
    # It reads a set of data samples, use the first half to train PCA,
    # and then use the second half for testing.

    # read the samples
    x = []
    pidName = '_pid0' + str(pid) + '.bin'  # hard coded file name
    for i in range(1, numIter + 1):
        fileName = 'phase1/iter' + '0' * (6 - len(str(i))) + str(i) + pidName
        temp = np.fromfile(fileName, dtype=np.float32)
        x.append(temp[:maxPara])
    X = np.transpose(np.matrix(x))

    # extract the training set
    XTrain = X[:, 0:150]

    # perform PCA
    Xc, meanCol = centerilze(XTrain)
    U, s, V = svd(Xc)

    # analysis compression ratio
    d, ratio = singluarAnalysis(s, thres=thres)
    print("compression ratio:", ratio)

    # compress then recover the training set, then analyse the loss
    XTrainC = compress(XTrain, meanCol, U, d)
    XTrainD = decompress(XTrainC, meanCol, U, d)
    print("training set recovery loss (normalized square error) is: ",
          offset(XTrain, XTrainD))

    # extract the testing set
    XTest = X[:, 150:300]

    # compress then recover the training set, then analyse the loss
    XTestC = compress(XTest, meanCol, U, d)
    XTestD = decompress(XTestC, meanCol, U, d)
    print("training set recovery loss (normalized square error) is: ",
          offset(XTest, XTestD))
    return ratio, offset(XTest, XTestD)


def compress(X, meanCol, U, d):
    # compress a sample set by projecting them to the first d dimension
    # subspace spanned by U.
    X = np.subtract(X, np.repeat(meanCol, X.shape[1], 1))
    return np.dot(np.transpose(U[:, :d]), X)


def decompress(Xd, meanCol, U, d):
    # decompress the compressed sample set by projecting them to the original
    # n-dimensional space.
    Xr = np.add(np.dot(U[:, :d], Xd), np.repeat(meanCol, Xd.shape[1], 1))
    return Xr


def offset(X, Xr):
    # calculate the normalized square error of ground truth X
    # and the recovered Xr.
    diff = np.subtract(X, Xr)
    diff = np.sum(np.multiply(diff, diff))
    base = np.sum(np.multiply(X, X))
    return diff / base


def doubleplot(x, y1, y2):
    # plot two y-axis in one graph.
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'b-', linewidth=3)
    ax1.set_xlabel('variance loss (%)', fontsize=16)
    ax1.set_ylabel('compression ratio', color='b', fontsize=16)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r--', marker='s', linewidth=3)
    ax2.set_ylabel('testing loss (%)', color='r', fontsize=16)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.grid()
    plt.show()


for pid in [0, 2, 4, 6, 8]:
    print("\ncurrently working on pid=", pid)
    r = []
    loss = []
    varloss = [20, 15, 10, 5, 1]
    for var in varloss:
        print("when var loss is ", var, "%:")
        rr, ll = getRatio(pid=pid, thres=(var / 100))
        r.append(rr)
        loss.append(ll)
    loss = [l * 100 for l in loss]
    doubleplot(varloss, r, loss)
    print("\n")
