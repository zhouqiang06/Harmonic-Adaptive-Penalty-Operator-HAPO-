# A copy of github:
https://code.usgs.gov/lcmap/research/Harmonic-Adaptive-Penalty-Operator-HAPO-

# This is the source code of HAPO regression method for harnomic analysis
HAPO introduced a new penalty function to minimize unnecessary fluctuations in the harmonic regression model, reducing the over-fitting issue in time series (https://doi.org/10.1016/j.isprsjprs.2022.01.006). 

![alt text](https://github.com/zhouqiang06/Harmonic-Adaptive-Penalty-Operator-HAPO-/blob/main/tests/test_plot.png?raw=true)

The figure demonstrates the different regression models derived from the four methods for a Landsat NIR surface reflectance time series with a large gap between March and August, in which HAPO is the only method that is not impacted by this data gap (∼ 4 months) (HAPO: Harmonic Adaptive Penalty Operator; LASSO: least absolute shrinkage and selection operator; OLS: ordinary least squares; and Ridge: Ridge regression). The observation data are acquired from Landsat Analysis Ready Data (ARD) products (https://earthexplorer.usgs.gov/). The location is grassland near Montgomery, Alabama, USA (Latitude: 32.408041°, Longitude −86.272600°).


# Install
## System requirements
It's highly recommended to do all your development & testing in anaconda virtual environment.

* python3-dev

* python-virtualenv

Required modules

* numpy>=1.18.1

* scikit-learn>=0.22.1

* scipy>=1.4.1

# Usage

Please run the test.py to get the following plot
![alt text](https://github.com/zhouqiang06/Harmonic-Adaptive-Penalty-Operator-HAPO-/blob/main/tests/test_plot.png?raw=true)
