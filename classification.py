from sklearn.datasets import make_moons

from value import Value
from network import MLP
from model import Model

import random
import matplotlib.pyplot as plt
import numpy as np


class Classifier(Model):
    def __init__(self, nin=2, nouts=[4, 4, 1]):
        np.random.seed(9999)
        random.seed(9999)
        self.mlp = MLP(nin, nouts)

    def create_dataset(self):
        self.X, self.y = make_moons(n_samples=100, noise=0.1)
        self.y = self.y*2 - 1

    def plot_points(self):
        plt.figure(figsize=(5, 5))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=20)

    def calc_loss(self):
        scores = list(map(self.mlp, self.X))

        # svm max margin loss
        losses = [(1 - yi*scorei).relu() for yi, scorei in zip(self.y, scores)]
        data_loss = sum(losses)/len(losses)

        # l2 regularization
        alpha = 1e-4
        l2_regularization_loss = alpha * \
            sum(p*p for p in self.mlp.parameters())

        loss = data_loss + l2_regularization_loss

        correct = [(yi == 1 and scorei.data > 0) or (
            yi == -1 and scorei.data < 0) for yi, scorei in zip(self.y, scores)]

        accuracy = sum(correct)/len(scores)
        return loss, accuracy

    def gradient_descent(self, iterations=100, printiter=1):
        assert self.mlp
        assert self.X is not None
        assert self.y is not None

        for k in range(iterations):
            loss, acc = self.calc_loss()

            self.mlp.zero_grad()
            loss.backward()

            alpha = 1.0 - .9*k/iterations
            for p in self.mlp.parameters():
                p.data -= alpha * p.grad

            if k % printiter == 0:
                print(loss, acc)

    def plot_decision_boundary(self):
        h = 0.25
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Xmesh = np.c_[xx.ravel(), yy.ravel()]
        inputs = [list(map(Value, xrow)) for xrow in Xmesh]
        scores = list(map(self.mlp, inputs))
        Z = np.array([s.data > 0 for s in scores])
        Z = Z.reshape(xx.shape)

        fig = plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(self.X[:, 0], self.X[:, 1],
                    c=self.y, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()
