import numpy as np

# M = np.array([[7, 4, 6], [1, 2, 4], [0, 0, 0], [-1, -2, -4], [-2, -4, -6]])

# B = np.dot(np.transpose(M), M)
# # print(B)

# U, s, V = np.linalg.svd(B, full_matrices=True)


def centerilze(X):
    meanCol = np.matrix(np.mean(X, axis=1))
    meanRepeat = np.repeat(meanCol, X.shape[1], 1)
    X = np.subtract(X, meanRepeat)
    return X, meanRepeat


def svd(X):
    _, c = X.shape
    covMatrix = np.dot(X, np.transpose(X))
    U, s, V = np.linalg.svd(covMatrix)
    return U, s, V


def singluarAnalysis(s, thres=0.95):
    d = len(s)
    cumu = 0
    sumThres = sum(s) * thres
    for i in range(d):
        cumu += s[i]
        if cumu > sumThres:
            break
    return i + 1, d / (i + 1)


def getRatio(pid=0, numIter=300, maxPara=1000, thres=0.95):
    x = []
    cumu = 0
    pidName = '_pid0' + str(pid) + '.bin'
    for i in range(1, numIter + 1):
        fileName = 'phase1/iter' + '0' * (6 - len(str(i))) + str(i) + pidName
        temp = np.fromfile(fileName, dtype=np.float32)
        x.append(temp[:maxPara])
        cumu += temp[1]
    X = np.transpose(np.matrix(x))
    XTrain = X[:, :300]
    Xc, meanRepeat = centerilze(XTrain)
    U, s, V = svd(Xc)
    d, ratio = singluarAnalysis(s, thres=thres)
    print("compression ratio:", ratio)
    Ud = U[:, :d]
    Xd = np.dot(np.transpose(Ud), XTrain)
    Xr = np.add(np.dot(Ud, Xd), meanRepeat)
    offset(XTrain, Xr)
    return Xr


def offset(X, Xr):
    diff = np.subtract(X, Xr)
    diff = np.sum(np.multiply(diff, diff))
    base = np.sum(np.multiply(X, X))
    print("square offset is: ", diff / base)


getRatio(thres=0.95)
