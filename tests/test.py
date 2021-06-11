import numpy as np
from matplotlib import pyplot as plt
from datetime import date
from HAPO import HAPOLinearModel


def coefficient_matrix(dates, avg_days_yr=365.25, max_freq=3):
    """
    Fourier transform function to be used for the matrix of inputs for
    model fitting
    Args:
        dates: list of ordinal dates
        num_coefficients: how many coefficients to use to build the matrix
    Returns:
        Populated numpy array with coefficient values
    """
    w = 2 * np.pi / avg_days_yr

    matrix = np.zeros(shape=(len(dates), max_freq * 2 + 1), order='F')

    cos = np.cos
    sin = np.sin

    w1 = w * dates
    matrix[:, 0] = np.ones(len(dates))

    for i in range(max_freq):
        matrix[:, 2 * i + 1] = cos((i+1) * w1)
        matrix[:, 2 * i + 2] = sin((i+1) * w1)

    return matrix

avg_days_yr = 365.25 # The constant T in the manuscript

# Load the test data
# The data are 2 by 13 matrix that record date and NIR surface reflectance in 2017 from a Landsat pixel
data = np.load(r'K:\90daytemp\QZhou\transfer\tile_valid_results\Bands\GA\ts_demo\data.npy')
data_year = 2017

pred_dates = np.asarray(range(date(data_year, 1, 1).toordinal(), date(data_year, 12, 31).toordinal())) # dates to predict
max_freq = 3 # we use annual, bi-annual, and tri-annual frequencies in the harminic regression

# Construct the harmonic components for regression and prediction
coef_matrix = coefficient_matrix(data[0, :], avg_days_yr, max_freq=max_freq)
pred_coef = coefficient_matrix(pred_dates, avg_days_yr, max_freq=max_freq)

# Run the HAPO model
model = HAPOLinearModel(deTrend=True, time_x=data[0, :], alpha=1)
model.fit(coef_matrix, data[1, :])
pred = model.predict(pred_coef, time_pred=pred_dates)

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(data[0, :], data[1, :], 'ko', label='Observations')
plt.plot(pred_dates, pred, '-', label='HAPO')
plt.legend()
plt.ylim((0.8 * np.min(pred), 1.2 * np.max(pred)))
x_tick = [date(data_year, m, 1) for m in range(1, 13)]
plt.xticks(x_tick, [x.strftime("%Y/%m/%d") for x in x_tick], rotation=45)
plt.xlabel('Date')
plt.ylabel('NIR')
plt.tight_layout()
plt.show()
