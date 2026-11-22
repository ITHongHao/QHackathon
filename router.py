"""The learned router: given a shot where BP did not converge, decide whether to trust the
OSD-0 answer or escalate the shot to relay-BP.

Two models with the same interface:
  LogisticRouter  9 weights + 1 bias. A single prediction is a few arithmetic operations.
  GbmRouter       a small gradient-boosted tree ensemble (scikit-learn). More flexible, but a
                  single prediction goes through scikit-learn and takes far longer.

score(features) is the predicted probability that the OSD-0 answer is wrong.
escalate(features) is True when that probability is above the router's threshold.
"""

import pickle

import numpy as np


def signed_log(x):
    """Compress heavy-tailed features (counts, costs) while keeping their sign."""
    return np.sign(x) * np.log1p(np.abs(x))


def choose_threshold(scores, wrong, n_all, budget):
    """Highest score threshold such that shots accepted below it, yet wrong, are at most
    `budget` x n_all. n_all counts every basis-shot, including ones where BP converged.

    Returns the threshold; shots with score <= threshold are accepted. -inf means escalate all.
    """
    order = np.argsort(scores)
    cumulative_wrong = np.cumsum(wrong[order])
    ok = np.flatnonzero(cumulative_wrong <= budget * n_all)
    if ok.size == 0:
        return -np.inf
    last = ok[-1]
    # Don't split ties: if the next score equals this one, step back to a strict boundary.
    while last + 1 < len(order) and scores[order[last + 1]] == scores[order[last]]:
        last -= 1
        if last < 0:
            return -np.inf
    return float(scores[order[last]])


class LogisticRouter:
    kind = "logistic"

    def __init__(self, mean, std, weights, bias, threshold):
        self.mean, self.std = np.asarray(mean), np.asarray(std)
        self.weights, self.bias = np.asarray(weights), float(bias)
        self.threshold = float(threshold)
        # Compare logits instead of probabilities: same decision, no exp() per shot.
        t = self.threshold
        self.logit_threshold = -np.inf if t <= 0 else np.inf if t >= 1 else np.log(t / (1 - t))

    @classmethod
    def fit(cls, X, y):
        from sklearn.linear_model import LogisticRegression
        Z = signed_log(X)
        mean, std = Z.mean(axis=0), Z.std(axis=0) + 1e-9
        model = LogisticRegression(max_iter=2000).fit((Z - mean) / std, y)
        return cls(mean, std, model.coef_[0], model.intercept_[0], threshold=0.5)

    def scores(self, X):
        return 1 / (1 + np.exp(-(((signed_log(X) - self.mean) / self.std) @ self.weights + self.bias)))

    def escalate(self, x):
        return ((signed_log(x) - self.mean) / self.std) @ self.weights + self.bias > self.logit_threshold

    def with_threshold(self, t):
        return LogisticRouter(self.mean, self.std, self.weights, self.bias, t)

    @property
    def n_parameters(self):
        return self.weights.size + 1

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, weights=self.weights, bias=self.bias, threshold=self.threshold)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(f["mean"], f["std"], f["weights"], f["bias"], f["threshold"])


class GbmRouter:
    kind = "gbm"

    def __init__(self, model, threshold):
        self.model, self.threshold = model, float(threshold)
        # scikit-learn predicts with a pool of OpenMP threads. For one shot at a time, starting
        # and syncing those threads costs far more than the prediction (tens of ms vs. well
        # under one), so the router runs single-threaded. This applies process-wide.
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)

    @classmethod
    def fit(cls, X, y, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=seed)
        return cls(model.fit(signed_log(X), y), threshold=0.5)

    def scores(self, X):
        return self.model.predict_proba(signed_log(X))[:, 1]

    def escalate(self, x):
        return self.model.predict_proba(signed_log(x)[None, :])[0, 1] > self.threshold

    def with_threshold(self, t):
        return GbmRouter(self.model, t)

    @property
    def n_parameters(self):
        # every tree node stores a feature/threshold or a leaf value
        return int(sum(tree[0].nodes.shape[0] for tree in self.model._predictors))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "threshold": self.threshold}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["model"], d["threshold"])
