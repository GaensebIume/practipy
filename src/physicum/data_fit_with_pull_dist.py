from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq as leastsq
from scipy.stats import norm


def plot_fit_with_pull(
    data: np.ndarray,
    params: list,
    fitfunc: Callable,
    x_label: str,
    plot_y_label: str,
    pull_y_label: str,
    x_label: str,
    plot_title: str,
    normal_title: str,
    filename_plot: str = "dataFit.jpg",
    filename_normal: str = "dataFitNormal.jpg",
    fit_label: str = "Fit",
    datapoint_label: str = "Data",
    pull_point_label: str = "Pull",
    normal_y_label: str = "Frequency",
    fit_color: str = "red",
    zero_axis_color: str = "black",
    normal_color: str = "blue",
    histogram_color: str = "orange",
    pull_error_color: str = "green",
    data_error_color: str = "blue",
    dpi: int = 1000,
    maxfev: int = 10000,
) -> None:
    """
    Plots the data and the fit with the pull distribution. Plots the normal distribution of the pulls.
    If the fit is good, the pulls should be normal distributed with mean 0 and variance 1.
    Saves both plots as file.

    Parameters
    ----------
    data : np.ndarray
        The data to be plotted.
    params : list
        The parameters of the fit.
    fitfunc : Callable
        The function to be used for the fit.
    x_label : str
        The label for the shared x-axis.
    plot_y_label : str
        The label for the y-axis.
    pull_y_label : str
        The label for the pull y-axis.
    normal_x_label : str
        The label for the normal x-axis.
    plot_title : str
        The title for the main plot.
    normal_x_label : str
        The title for the normal-distribution.
    filename_plot : str
        The filename for the saved plot. Must end with .jpg or .png.
    filename_normal : str
        The filename for the normal distribution. Must end with .jpg or .png.
    normal_title : str
        The title for the normal distribution.
    fit_label : str, optional
        The label for the fit, by default "Fit"
    datapoint_label : str, optional
        The label for the datapoint, by default "Data"
    pull_point_label : str, optional
        The label for the pull point, by default "Pull"
    normal_y_label : str, optional
        The label for the normal y-axis, by default "Frequency"
    fit_color : str, optional
        The color for the fit, by default "red"
    zero_axis_color : str, optional
        The color for the zero axis which is added for emphasis, by default "black"
    normal_color : str, optional
        The color for the normal distribution, by default "blue"
    histogram_color : str, optional
        The color for the histogram, by default "orange"
    pull_error_color : str, optional
        The color for the pull error, by default "green"
    data_error_color : str, optional
        The color for the data error, by default "blue"
    plot_fmt : str, optional
        The format for the plot, by default "bo" for blue circles.
        See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot for reference (Table: Matkers)
    pull_fmt : str, optional
        The format for the pull, by default "go" for green circles
        See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot for reference (Table: Matkers)
    dpi:int, optional
        The dpi for the plot, by default 1000
    maxfev : int, optional
        The maximum number of function evaluations, by default 10000
    """
    x = data[:, 0]
    y = data[:, 1]
    errory = data[:, 2]
    errorx = np.zeros(len(data[:, 0])) if len(data[0]) < 4 else data[:, 3]

    def pull_function(parameter, x, y, errory, errorx):
        total_error = np.sqrt(errory**2 + errorx**2)
        return (y - fitfunc(x, parameter)) / total_error

    par_f, _, info, errmsg, ierr = leastsq(
        pull_function,
        params,
        args=(x, y, errory, errorx),
        full_output=True,
        maxfev=maxfev,
    )
    if ierr not in [1, 2, 3, 4]:
        msg = "Optimal parameters not found: " + errmsg
        raise RuntimeError(msg)
    print(f"Convergence after {info['nfev']} function calls")
    chi2 = (info["fvec"] ** 2).sum()
    ndof = len(data[:, 0]) - len(par_f)
    print(f"Chi2/NDF = {chi2:.2f}/{ndof}={chi2/ndof:.4f}")

    xx = np.linspace(data[:, 0][0], data[:, 0][-1], 100)
    yy = fitfunc(xx, par_f)

    pulls = info["fvec"]
    e_pull = np.ones(len(pulls))
    msize = 3.0
    lwidth = 1.5

    fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
    ax[0].errorbar(
        x,
        y,
        yerr=errory,
        xerr=errorx,
        fmt="bo",
        markersize=msize,
        linewidth=lwidth,
        zorder=3.0,
        label=datapoint_label,
        ecolor=data_error_color,
    )
    ax[0].plot(xx, yy, color=fit_color, lw=lwidth, label=fit_label)
    ax[0].set_ylabel(plot_y_label)
    ax[0].grid()
    ax[0].legend()
    ax[1].errorbar(
        data[:, 0],
        pulls,
        yerr=e_pull,
        xerr=errorx,
        fmt="go",
        markersize=msize,
        linewidth=lwidth,
        zorder=3.0,
        label=pull_point_label,
        ecolor=pull_error_color,
    )
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(pull_y_label)
    ax[1].set_ylim(-5.0, 5.0)
    ax[1].grid()
    ax[1].legend()
    fig.suptitle(plot_title)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.gca().axhline(y=0, color=zero_axis_color, linewidth=2)
    plt.savefig(filename_plot, dpi=dpi)
    plt.show()

    # Normal distribution of pulls, if fit is good.
    # Pulls should be normal distributed with mean 0 and variance 1
    plt.figure()
    plt.title(normal_title)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, norm.pdf(x, 0, 1), color=normal_color)
    plt.hist(pulls, density=True, bins=len(pulls), color=histogram_color)
    plt.xlabel(normal_x_label)
    plt.ylabel(normal_y_label)
    plt.savefig(filename_normal, dpi=dpi)
    plt.show()


if __name__ == "__main__":
    CONST1 = 1.6
    CONST2 = 1.67
    plot_fit_with_pull(
        np.genfromtxt("./CmsHiggs2GammaGamma.csv", delimiter=";", skip_header=1),
        [1, 1, 1, 1, 1, 1, 100.0, 125.0],
        lambda x, p: p[0]
        + p[1] * x
        + p[2] * x**2
        + p[3] * x**3
        + p[4] * x**4
        + p[5] * x**5
        + p[6]
        * CONST2
        / np.sqrt(2 * np.pi * CONST1**2)
        * np.exp(-0.5 * (x - p[7]) ** 2 / CONST1**2),
        x_label="x",
        plot_y_label="y",
        pull_y_label="pull",
        normal_x_label="x",
        plot_title="Plot title",
        normal_title="Normal title",
    )
