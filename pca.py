import numpy as np


def centerilze(X):
    # This function substracts the mean of each row of X
    # from the entries of the corresponding row.
    # If X is an n * m matrix, where each column is a sample,
    # then this function effectively centralizes the samples.
    meanPara = np.matrix(np.mean(X, axis=1))
    meanRepeat = np.repeat(meanPara, X.shape[1], 1)
    X = np.subtract(X, meanRepeat)
    return X, meanPara


def svd(X):
    # This function calculates the covariance matrix of X
    # and then decompose it
    # into singular vectors and signular values.
    _, c = X.shape
    covMatrix = np.dot(X, np.transpose(X)) / (c - 1)
    U, s, V = np.linalg.svd(covMatrix)
    return U, s


def generateCompressor(U, s, thres=0.05):
    # Given a signluar value vector and a threshold, this function finds the
    # number of largest singular values whose sum is more than $1-thres$
    # of the total singular values. It then returns the corresponding
    # eigen vectors
    dmax = len(s)
    cumu = 0
    sumThres = sum(s) * (1 - thres)
    for d in range(dmax):
        cumu += s[d]
        if cumu > sumThres:
            break
    return U[:, :d + 1], d + 1


def compress(X, meanPara, Ud):
    # compress a sample set by projecting them to the first d dimension
    # subspace spanned by U.
    # First centralize the data
    Xc = np.subtract(X, np.repeat(meanPara, X.shape[1], 1))
    # Then reduce its dimension
    return np.dot(np.transpose(Ud), Xc)


def decompress(Xc, meanPara, Ud):
    # decompress the compressed data by projecting them to the original
    # n-dimensional space.
    Xd = np.add(np.dot(Ud, Xc), np.repeat(meanPara, Xc.shape[1], 1))
    return Xd


def squareLoss(X, Xd):
    # calculate the normalized square error of the recovered Xr compared with
    # the ground truth X
    diff = np.subtract(X, Xd)
    diff = np.sum(np.multiply(diff, diff))
    base = np.sum(np.multiply(X, X))
    return diff / base


def absLoss(X, Xd):
    # calculate the square error of the recovered Xr compared with
    # the ground truth X, and the 2-norm of X.
    diff = np.subtract(X, Xd)
    diff = np.sum(np.multiply(diff, diff))
    norm = np.sum(np.multiply(X, X))
    return diff, norm
