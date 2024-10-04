# Features
## Analytical error propagation
Used to calculate the mean, mean value, standard deviation and standard deviation value of a physical quantity with known dependency on other physical quantities.
### Parameters
- function: The connection between the physical quantity and the other physical quantities. For example: p = m * v. All variables must be sympy symbols.
- values: The values of the used physical quantities. Must be a dictionary. For example: {"m": 10, "v": 15}. The units are arbitrary and must be calculated seperately.
- variables: A list of sympy symbols to be substituted in the function. For example: \[m, v\]. If you use physical constants in the function, they need to be in this list. Fill in the covariance matrix accordingly (correlation to other quantities is 0, variance must be found in literature). 
- correlation: The covariance matrix of the physical quantities. This matrix must be symmetric. For example: \[\[sm, rsm\], \[rsm, sv\]\]. The order of the variables must be the same as the order in the variables list.
Example covariance matrix:

|                       	| $\boldsymbol{X_1}$ 	| $\boldsymbol{X_2}$ 	| $\boldsymbol{\cdots}$ 	| $\boldsymbol{X_n}$ 	|
|-----------------------	|--------------------	|--------------------	|-----------------------	|--------------------	|
| $\boldsymbol{X_1}$    	| $\sigma_1^2$      	| $\rho_{12}$        	| $\cdots$              	| $\rho_{1n}$         	|
| $\boldsymbol{X_2}$    	| $\rho_{12}$        	| $\sigma_2^2$      	| $\cdots$              	| $\rho_{2n}$        	|
| $\boldsymbol{\vdots}$ 	| $\vdots$           	| $\vdots$           	| $\ddots$              	| $\vdots$           	|
| $\boldsymbol{X_n}$    	| $\rho_{1n}$        	| $\rho_{2n}$        	| $\cdots$              	| $\sigma_n^2$      	|

### Returns
- mean: The mean of the physical quantity. Is usually the same as the function.
- mean_val: The mean value of the physical quantity given the values parameter.
- std: The standard deviation of the physical quantity, derived by the quaratic sum of all partial derivatives and the covariance matrix.
- std_val: The standard deviation value of the physical quantity given the values parameter.
- partials: All partial derivatives.

### Example
```python
import sympy as sp
from physicum.analytical_error_propagation import analytical_error_propagation

m, v = sp.symbols("m v")
sv, sm, rsm = sp.symbols("sigma_v sigma_m rho_vm")
data: dict[str, int | float] = {
    "m": 10, # kg
    "v": 15, # m/s
    "sigma_m": 1, # kg
    "sigma_v": 1, # m/s
    "rho_vm": 0, # no correlation
}
E = 1 / 2 * m * v**2
var = [m, v]
cor: list[list[sp.Symbol]] = [[sm, rsm], [rsm, sv]]
sp.pprint(
    analytical_error_propagation(
        function=E,
        values=data,
        correlation=cor,
        variables=var,
    )
)
```
Output:
```
1/2*m*v**2, 1125, sqrt((m*v*sigma_v)**2+(1/2*v**2*sigma_m**2)), 187.5, [1/2*v^2, m*v]
```
## Fit with pull distribution
Performs a least-sqaure optimization on the pull function a function given a data set. Plots the fit with the data and the resulting pull distribution. To evaluate the quality of the fit, the residuals are plotted against the z-distribution. If the fit is good, the histogram should look similar to the curve. 

### Parameters
- data: The data to be plotted. The x, y and y-error values must be given in the first three columns. If a fourth column is given, it is interpreted as the error of the x-values, else the x-error is set to 0 for all data points.
- params: The parameters of the function to be fitted.
- fitfunc: The function to be fitted.
- x_label: The label for the x-axis. (There is only one x-axis, as the two plots share the same x-axis.)
- plot_y_label: The label for the y-axis.
- pull_y_label: The label for the y-axis of the pull distribution.
- normal_x_label: The label for the x-axis of the z-distribution and the residuals plot.
- plot_title: The title for the plot.
- normal_title: The title for the z-distribution and the residuals plot.
- filename_plot: The filename for the saved plot. Must end with .jpg or .png.
- filename_normal: The filename for the saved z-distribution and residuals plot. Must end with .jpg or .png.
- fit_label: The label for the fitted curve.
- datapoint_label: The label for the data.
- pull_point_label: The label for the pull distribution.
- normal_y_label: The label for the y-axis of the z-distribution and the residuals plot.
- fit_color: The color for the fitted curve.
- zero_axis_color: The color for the zero axis which is added for emphasis.
- normal_color: The color for the z-distribution in the residuals plot.
- histogram_color: The color for the histogram in the residuals plot.
- pull_error_color: The color for the error of the pull distribution.
- data_error_color: The color for the error of the data.
- dpi: The dpi for the plot.
- maxfev: The maximum number of function evaluations performed by scipy.optimize.leastsq.

### Returns
Nothing

### Example
```python
from physicum.fit_with_pull_dist import plot_fit_with_pull
data = np.genfromtxt("/PATH/TO/THE/DATA")
params = [1,1,1]
fitfunc = lambda x, p: p[0] + p[1] * x + p[2] * x**2
plot_fit_with_pull(
    data,
    params,
    fitfunc
)
```

## Fourier transform
Plots the fourier series and its coefficients for a given function.
### Parameters
- function: The function to be plotted.
- x_min: The function to be plotted.
- x_max: The maximum value of the x-axis.
- period: The period of the function.
- plts: List of orders of the fourier series to be plotted.
- x_label: The label for the x-axis (used in the plot of the function and the fourier series.
- y_label: The label for the y-axis (used in the plot of the function and the fourier series.
- filename: The filename for the plot. Must end with .jpg or .png.
- x_label_index: The label for the x-axis of the coefficients.
- y_label_index_a: The label for the y-axis of the a_k coefficients.
- y_label_index_b: The label for the y-axis of the b_k coefficients.
- coefficient_color: The color for the coefficients.
- coefficient_error_color: The color for the coefficient error (error returned by scipy.integrate.quad).
- data_color: The color in which the function is plotted.
- period_color: The color in which the vertial line for the period is plotted.
- period_label: The label for the period.
- signal_label: The label for the signal.
- coefficient_error_label: The label for the coefficient error.
- coefficient_label: The label for the coefficients.
- fourrier_colors: The colors for the fourier series. If empty, the colors are set by matplotlib.
- colors_passed: Must be set to True to use the colors in fourrier_colors.
- points: The number of points sampled from the function to draw the plots.
- padding: Padding factor for the y-axis.

### Returns
- ak_values: The values of the a_k coefficients.
- bk_values: The values of the b_k coefficients.
- ak_errors: The errors of the a_k coefficients as returned by scipy.integrate.quad.
- bk_errors: The errors of the b_k coefficients as returned by scipy.integrate.quad.

### Example
```python
from physicum.fourier import fourier
T = 8.0
x_min = -4
x_max = 20.0
draw_series_element = [10,20]

fourier(
    lambda t: signal.square(2 * np.pi * t / T),
    x_min,
    x_max,
    T,
    draw_series_element,
    "Time [s]",
    "Meassurement [UNIT]",
)
```
## Negative log likelihood 
Used to find the mean and standard deviation of a single physical quantity (that is assumed to be a distributed normally) given the meassured data.
### Parameters
- data: A list of data points.
### Returns
- data_mean_fit: The mean of the data after the fit.
- error_data_mean: The uncertainty of the mean of the data after the fit.
- data_std_fit: The standard deviation of the data after the fit.
- error_data_std: The uncertainty of the standard deviation of the data after the fit.
### Example
```python
from physicum.nll import nll
data = np.genfromtxt("/PATH/TO/THE/DATA")
nll(list(data))
```
Output:
```
µ, σ_µ, σ, σ_σ
```
## Plot negative log likelihood
Uses [negative log likelihood](#Negative-log-lokelihood) to find the mean, standard deviation and their uncertainty of a single physical quantity (that is assumed to be a distributed normally) given the meassured data.
Plots a histogram of the data with errorbars and a normalized histogram of the data with the fit.
### Parameters
- data: A list of data points.
- x_label: The label for the x-axis.
- y_label: The label for the y-axis.
- y_label_density: The label for the y-axis of the normalized plot.
- datalabel: The label for the data.
- errorlabel: The label for the errorbars.
- fitlabel: The label for the fit.
- meanlabel: The label for the vertical line at the mean.
- mean_unit: The unit of the mean. Used in the legend.
- suptitle: The title for the plot.
- subtitle1: The subtitle for the plot.
- subtitle2: The subtitle for the normalized plot.
- size: The number of points for the x-axis.
- capsize: The capsize for the error bars.
- hist_bins: The number of bins for the histogram.
- hist_color: The color for the histogram.
- error_color: The color for the error bars.
- fit_color: The color for the fit.
- mean_color: The color for the mean.
### Returns
- data_mean_fit: The mean of the data after the fit.
- error_data_mean: The uncertainty of the mean of the data after the fit.
- data_std_fit: The standard deviation of the data after the fit.
- error_data_std: The uncertainty of the standard deviation of the data after the fit.
### Example
```python
from physicum.nll import plot_nll
data = np.genfromtxt("/PATH/TO/THE/DATA")
plot_nll(
    list(data),
    x_label="x",
    y_label="Counts",
    y_label_density="Density",
)
```
## Confidence bands
Plots a function and its confidence bands.
### Parameters
- function: The function to plot the confidence bands for.
- values: The values to substitute in the function.
- variables_mean: The mean of the non-x-axis variables.
- std: The standard deviations of the variables.
- variables: The variables, the function is dependent on. Must end with the x-axis variable (the time t in the example below).
- time_interval: The time interval to plot the function over. Has to have 3 entries: start, end, number of points.
- confidence_level: The confidence levels to plot. May be percentages or sigmas.
- is_sigma: Set True if confidence_level is filled with sigma-values.
- function_label: The label of the function.
- function_color: The color of the function.
- colors: The colors of the confidence bands.
- filename: The filename of the saved plot.
### Returns
Nothing
### Example
```python
from physicum.confidence_bands import conficende_bands
x0, y0, theta, v, yf, t, g = sp.symbols("x0 y0 theta v yf t g")
s_time, s_theta = sp.symbols("sigma_t sigma_theta")
data = {x0: 100, y0: 50, v: 200, g: -9.81}
theta_val = 45 * np.pi / 180

timeInt = [0, 35, 200]
variable = [theta, t]
stds = [s_theta, s_time]
std_values = [np.pi / 180, 0]
h = y0 + v * sp.sin(theta) * t + sp.Rational(1, 2) * g * t**2
conficende_bands(
        function=h,
        values=data,
        variables_mean=[theta_val],
        std=std_values,
        variables=variable,
        time_interval=timeInt,
)
```
# Dependencies
- sympy
- matplotlib
- numpy
- scipy
- (typing)
# WIP Features
- Discrete Fourier transform
- Time synced plots (plot mulitplie measurements synchronized at a characteristic feature (e.g. slope reaches threshold for the first time))
- Negative-log-lokelihood for multiple physical quantities (maybe even not distributed normally)
- Plot of confidence levels for a quantity dependent on 2 (graphic) or more (???) quantities
