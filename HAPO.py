import numpy as np
from scipy.optimize import minimize
from sklearn import linear_model


def powerSum(freqs):
    if len(freqs) % 2 == 1:  # odd number
        print('Length of the list must be even number!')
        return None
    return np.sum([np.sqrt((freqs[i] ** 2 + freqs[i + 1] ** 2)) for i in range(0, len(freqs), 2)])


class HAPOLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with regularization
    """

    def __init__(self, alpha=1.0, sample_weights=None, coef_init=None, max_iter=1000, deTrend=False, normalize=True, time_x=None, aprox_max=None, aprox_min=None):
        self.alpha = alpha
        self.coef_ = None
        self.max_iter = max_iter
        self.sample_weights = sample_weights
        self.coef_init = coef_init
        self.deTrend = deTrend
        self.normalize = normalize
        self.aprox_max = aprox_max
        self.aprox_min = aprox_min
        self.max_freq = 0
        self.freq_list = None

        if self.deTrend:
            if time_x is None:
                print('Please provide time of observations.')
            self.time_x = time_x

        self.objective = self.d1_regularized_loss


    def predict_harm(self, X):
        return np.matmul(X, self.coef_)

    def predict(self, X, time_pred=None):
        prediction = self.predict_harm(X)

        if self.normalize:
            prediction = prediction * (self.aprox_max - self.aprox_min) + self.aprox_min

        if self.deTrend:
            if time_pred is None:
                print('Please provide time of observations for prediction')
                return None
            return self.trend_model.predict(time_pred[:, np.newaxis]) - np.mean(self.trend_model.predict(self.time_x[:, np.newaxis])) + prediction
        return prediction

    def model_sse(self):
        error = sum((self.predict_harm(self.X) - self.Y) ** 2)
        return error

    def model_error(self):
        error = self.loss_function(
            self.predict_harm(self.X), self.Y, sample_weights=self.sample_weights
        )
        return error


    def d1_regularized_loss(self, coef):  #
        self.coef_ = coef

        if self.normalize:
            loss = (1 / (2 * self.Y.shape[0])) * self.model_sse() * \
                   (np.abs(1 - powerSum(self.coef_[1:])) + self.alpha * powerSum(
                    self.coef_[1:] * self.freq_list)) # small overall range, small overall length
        return loss


    def est_trend(self):
        self.trend_model = linear_model.LinearRegression()
        self.trend_model.fit(self.time_x[:, np.newaxis], self.Y)
        self.Y = self.Y - self.trend_model.predict(self.time_x[:, np.newaxis]) + np.mean(self.trend_model.predict(self.time_x[:, np.newaxis]))


    def est_norm(self):
        self.Y = (self.Y - self.aprox_min) / (self.aprox_max - self.aprox_min)


    def fit(self, X, y):
        # Initialize coef estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)

        self.X = X
        self.Y = y
        self.max_freq = int((self.X.shape[1] - 1) / 2)
        self.freq_list = np.repeat(list(range(self.max_freq)), 2) + 1

        if self.deTrend:
            self.est_trend()
        else:
            self.trend_model = linear_model.LinearRegression()
            self.trend_model.fit(self.Y, np.zeros_like(self.Y))

        if self.aprox_max is None:
            self.aprox_max = np.percentile(self.Y, 99)
        if self.aprox_min is None:
            self.aprox_min = np.percentile(self.Y, 1)

        if self.aprox_max == self.aprox_min:
            self.coef_ = np.zeros(self.X.shape[1]) + self.aprox_max
            self.coef_init = self.coef_
            return

        if self.normalize:
            self.est_norm()

        if type(self.coef_init) == type(None):
            self.coef_init = np.append(np.mean(self.Y)/2, np.random.random_sample(self.X.shape[1] - 1))

        else:
            # Use provided initial values
            pass

        if self.coef_ != None and all(self.coef_init == self.coef_):
            print("Model already fit once; continuing fit with more itrations.")

        res = minimize(self.objective, self.coef_init,
                       method='BFGS', options={'maxiter': self.max_iter})
        self.coef_ = res.x
        self.coef_init = self.coef_

