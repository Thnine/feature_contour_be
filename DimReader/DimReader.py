import csv
import json

import numpy as np
import torch as t
from sklearn.manifold import TSNE
from . import Grid


class DimReader:
    def __init__(self, X, Y, perp=30):
        '''类初始化

        Arguments:
            X {Tensor} -- 欲降维数据(n,nfeature)

        Keyword Arguments:
            perp {int} -- 困惑度 (default: {30})
        '''

        self.X = t.tensor(X, requires_grad=True)
        self.N = self.X.shape[0]
        self.M = self.X.shape[1]
        self.Y = t.tensor(Y)
        self.perp = perp

    def Hbeta(self, D, beta=1.0):
        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sumP = sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def x2p(self, beta):
        # Compute P-value

        (n, d) = self.X.shape
        sum_X = t.sum(t.square(self.X), 1)
        D = t.add(t.add(-2 * t.mm(self.X, self.X.T), sum_X).T, sum_X)
        P = t.exp(-D * beta.reshape(1, n))
        # P[t.eye(self.N, self.N, dtype=t.long) == 1] = 0
        P = P - t.diag_embed(t.diag(P))
        P = P / t.sum(P)

        return P

    def getBeta(self, perplexity=30, tol=1e-5):
        # compute beta

        print("计算高斯分布参数beta")
        X = self.X.detach().numpy()
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        # Loop over all datapoints
        for i in range(n):

            # Print progress
            if i % 500 == 0:
                print("Computing P-values for point %d of %d..." % (i, n))

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 50:

                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        # Return final P-matrix
        print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return t.from_numpy(beta.astype(np.float32))

    def getGrad(self):
        '''训练

        Keyword Arguments:
            epoch {int} -- 迭代次数 (default: {1000})
            lr {int} -- 学习率，典型10-100 (default: {10})
            weight_decay {int} -- L2正则系数 (default: {0})
            momentum {float} -- 动量 (default: {0.9})
            show {bool} -- 是否显示训练信息 (default: {False})

        Returns:
            Tensor -- 降维结果(n,2)
        '''

        # 先算出原分布的相似矩阵
        beta = self.getBeta(self.perp)
        P = self.x2p(beta)  # 计算原分布概率矩阵
        P = P + P.T
        sumP = t.max(t.sum(P), t.tensor(1e-12))
        P = P / sumP
        P = t.max(P, t.tensor(1e-12))

        sum_Y = t.sum(t.square(self.Y), 1)
        num = -2. * t.mm(self.Y, self.Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[t.eye(self.N, self.N, dtype=t.long) == 1] = 0
        Q = num / t.sum(num)
        Q = t.max(Q, t.tensor(1e-12))

        PQ = P - Q
        grad = t.zeros(self.N, 2, self.M)
        for i in range(self.N):
            dY = t.sum(t.reshape(PQ[:, i] * num[:, i], (1, -1)) * (self.Y[i, 0] - self.Y[:, 0]))
            dY.backward(retain_graph=True)
            grad[i, 0, :] = -self.X.grad[i]
            self.X.grad.zero_()

            dY = t.sum(t.reshape(PQ[:, i] * num[:, i], (1, -1)) * (self.Y[i, 1] - self.Y[:, 1]))
            dY.backward(retain_graph=True)
            grad[i, 1, :] = -self.X.grad[i]
            self.X.grad.zero_()
        return grad

    def run(self, perturbationIndex):
        grad = self.getGrad().numpy()

        pert = np.zeros(self.M)
        pert[perturbationIndex] = 1

        data = []
        for i in range(self.N):
            data.append({"domain": self.X[i].detach().numpy().tolist(),
                         "range": self.Y[i].numpy().tolist(),
                         "inputPert": pert.tolist(),
                         "outputPert": [float(grad[i, 0, perturbationIndex]), float(grad[i, 1, perturbationIndex])]
                         })

        g = Grid.Grid(self.Y.numpy().tolist(), grad[:, :, perturbationIndex].reshape((2 * self.N)).tolist())
        # 计算出各个顶点的值
        grid = g.calcGridVertices().tolist()

        result = {"points": data,
                  "grad": grad.tolist(),
                  "scalarField": grid}

        return result

    def getContour(self, grad, perturbationIndex):
        grad = np.array(grad)
        g = Grid.Grid(self.Y.numpy().tolist(), grad[:, :, perturbationIndex].reshape((2 * self.N)).tolist())
        # 计算出各个顶点的值
        grid = g.calcGridVertices().tolist()
        return grid


def runDimReader(X, Y, perturbationIndex=0, grad=[], perplexity=30):
    # 输入：
    # X：list二维数组
    # Y：list二维数组
    # perturbationIndex： int数字，小于X的维度
    # grad：list三维数组（如果有的话）
    # perplexity：超参数困惑度，数字
    # 输出：
    # 如果未输入梯度，返回对象数组，包括等高线scalarField，梯度grad，点数据points
    # 如果输入梯度，返回scalarField,点数据points
    dimReader = DimReader(X, Y, perplexity)
    if len(grad) != 0:
        res_countour = dimReader.getContour(grad, perturbationIndex)
        res = {
            'scalarField':res_countour,
            'points':Y.tolist(),
        }
    else:
        res = dimReader.run(perturbationIndex)
        res['points'] = Y.tolist()
    return res
