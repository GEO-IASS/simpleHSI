import numpy as np


class BaseEstimator(object):
    # Object atttributes
    X = None  # no X
    y = None  # no y
    y_required = True  # y required for base estimator
    fit_required = True  # fit required for base estimator

    def _setup_input(self, X, y=None):
        """BaseEstimator ensures the inputs to an estimator are in the
        expected format.



        Parameters
        ----------

        X : array
            Feature dataset ( samples x dimensions)
        y : array
            Target values. By default is required, but if y_required = false
            then they can be omitted.
        """
        # check if X input is an array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # check if X is size 0 (number of features == 0)
        if X.size == 0:
            raise ValueError('Number of features must be > 0')

        # check if if number of dimensions is 1
        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        # assign X as an attribute
        self.X = X

        # check y requirements
        if self.y_required:
            if y is None:
                raise ValueError('Missed required argument y')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('Number of targets must be > 0')

        # assign y as an attribute
        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        # check if X is array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError('You must call `fit` before `predict`')

    def _predict(self, X=None):
        raise NotImplementedError()

