
from value import Value
from network import MLP
from model import Model


import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


class Regressor(Model):
    def __init__(self, nin=2, nouts=[8, 8, 1], seed=9999):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.nin = nin
        self.mlp = MLP(nin, nouts)

    def create_dataset(self, n_samples=200, noise=0.1, kind="sincos"):
        rng = np.random.default_rng(self.seed)
        if kind == "sincos":
            X = rng.uniform(-3, 3, size=(n_samples, self.nin))
            y = np.sin(X[:, 0])
            if self.nin > 1:
                y = y + 0.5*np.cos(X[:, 1])
            y = y + noise*rng.standard_normal(n_samples)
        elif kind == "linear":
            X, y = make_regression(
                n_samples=n_samples, n_features=self.nin, noise=noise, random_state=self.seed)
        else:
            raise ValueError(f"unknown dataset kind: {kind}")

        # 60 / 20 / 20 train / val / test
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=self.seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=.5, random_state=self.seed)

        # standardise
        self.x_mean, self.x_std = X_train.mean(0), X_train.std(0) + 1e-8
        self.y_mean, self.y_std = y_train.mean(), y_train.std() + 1e-8

        self.splits = {
            name: (((Xs - self.x_mean)/self.x_std).tolist(),
                   ((ys - self.y_mean)/self.y_std).tolist())
            for name, Xs, ys in [("train", X_train, y_train),
                                 ("val", X_val, y_val),
                                 ("test", X_test, y_test)]
        }

    def plot_points(self):
        assert self.splits is not None
        X, y = self.splits["train"]
        X = np.array(X)
        plt.figure(figsize=(5, 5))
        if self.nin == 1:
            plt.scatter(X[:, 0], y, s=20)
            plt.xlabel("x1")
            plt.ylabel("y")
        else:
            plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="viridis")
            plt.colorbar(label="y")
            plt.xlabel("x1")
            plt.ylabel("x2")

    def calc_loss(self, split="train", alpha=1e-4):
        data_loss = self.evaluate(split)

        # l2 regularization
        l2_regularization_loss = alpha * \
            sum(p*p for p in self.mlp.parameters())

        loss = data_loss + l2_regularization_loss

        return loss, data_loss.data

    def predict(self, X):
        return np.array([self.mlp(x).data for x in X])

    def evaluate(self, split = "test"):
        X, y = self.splits[split]
        scores = [self.mlp(x) for x in X]

        # Mean squared error loss
        losses = [(yi - scorei)**2 for yi, scorei in zip(y, scores)]
        data_loss = sum(losses)/len(losses)
        return data_loss

    def gradient_descent(self, iterations=800, lr=0.05, printiter=10):
        assert self.mlp
        assert self.splits is not None, "call create_dataset() first."

        params = self.mlp.parameters()
        best_val, best_params = float('inf'), None

        for k in range(iterations):
            loss, train_mse = self.calc_loss()

            self.mlp.zero_grad()
            loss.backward()

            alpha = lr * (1.0 - .9*k/iterations)
            for p in params:
                p.data -= alpha * p.grad

            # validation
            val_mse = self.evaluate("val").data
            if val_mse < best_val:
                best_val, best_params = val_mse, [p.data for p in params]

            if k % printiter == 0 or k == iterations - 1:
                print(f"iter {k:4d}  loss {loss.data:.4f}  "
                      f"train MSE {train_mse:.4f}  "
                      f"val MSE {val_mse:.4f}")

        # keep the best params
        for p, d in zip(params, best_params):
            p.data = d

        print(f"best val MSE {best_val:.4f}")


    def plot_predictions(self, split="test"):
        X, y = self.splits[split]
        y = np.array(y)
        preds = self.predict(X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        lo, hi = min(y.min(), preds.min()), max(y.max(), preds.max())
        ax1.scatter(y, preds, s=20)
        ax1.plot([lo, hi], [lo, hi], "k--")
        ax1.set_xlabel("actual")
        ax1.set_ylabel("predicted")
        ax1.set_title(f"{split}: predicted vs actual")

        ax2.scatter(preds, preds - y, s=20)
        ax2.axhline(0, color="k", linestyle="--")
        ax2.set_xlabel("predicted")
        ax2.set_ylabel("residual")
        ax2.set_title(f"{split}: residuals")

    def plot_surface(self, h=0.1):
        assert self.nin == 2, "surface plot only for 2 input dimensions"
        X, y = self.splits["train"]
        X = np.array(X)
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Xmesh = np.c_[xx.ravel(), yy.ravel()].tolist()
        Z = self.predict(Xmesh).reshape(xx.shape)

        plt.figure()
        cs = plt.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap="viridis",
                    edgecolors="k", vmin=cs.get_clim()[0], vmax=cs.get_clim()[1])
        plt.colorbar(cs, label="prediction")
        plt.xlabel("x1")
        plt.ylabel("x2")

if __name__ == "__main__":
    r = Regressor()
    r.create_dataset()
    r.gradient_descent()

    r.plot_surface()
    r.plot_predictions()
    plt.show()