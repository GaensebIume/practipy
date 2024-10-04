import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from scipy.optimize import fsolve


def nll(
    data: list[int | float],
) -> tuple[float, float, float, float]:
    """
    Finds the mean and standard deviation of a (single) physical quantity
    (that is assumed to be a distrubted normally)
    given a function and a list of variables.

    Parameters
    ----------
    data : list[int | float]
        A list of data points.

    Returns
    -------
    data_mean_fit : float
        The mean of the data after the fit.
    error_data_mean : float
        The uncertainty of the mean of the data after the fit.
    data_std_fit : float
        The standard deviation of the data after the fit.
    error_data_std : float
        The uncertainty of the standard deviation of the data after the fit.
    """
    x, mu, sigma = sp.symbols("x mu sigma", real=True)
    f = 1 / sp.sqrt(2 * sp.pi * sigma**2) * sp.exp(-((x - mu) ** 2) / (2 * sigma**2))
    data_mean = np.mean(data)
    data_std = np.std(data, ddof=1)
    f = sp.simplify(-sp.log(f))

    df_dmu = sp.simplify(sp.diff(f, mu))
    f_df_dmu = sp.lambdify((x, mu, sigma), df_dmu)
    df_dsigma = sp.simplify(sp.diff(f, sigma))
    f_df_dsigma = sp.lambdify((x, mu, sigma), df_dsigma)
    d2f_dmu2 = sp.simplify(sp.diff(df_dmu, mu))
    f_d2f_dmu2 = sp.lambdify((x, mu, sigma), d2f_dmu2)
    d2f_dsigma2 = sp.simplify(sp.diff(df_dsigma, sigma))
    f_d2f_dsigma2 = sp.lambdify((x, mu, sigma), d2f_dsigma2)

    def d_f_loglike(p) -> tuple[float, float]:
        mu, sigma = p
        df_dmu = np.array([f_df_dmu(x, mu, sigma) for x in data]).sum()
        df_dsigma = np.array([f_df_dsigma(x, mu, sigma) for x in data]).sum()
        return df_dmu, df_dsigma

    def f_uncertainty(mu, sigma) -> tuple[float, float]:
        d2f_dmu2 = np.array([f_d2f_dmu2(x, mu, sigma) for x in data]).sum()
        d2f_dsigma2 = np.array([f_d2f_dsigma2(x, mu, sigma) for x in data]).sum()
        e_mu = 1 / np.sqrt(d2f_dmu2)
        e_sigma = 1 / np.sqrt(d2f_dsigma2)
        return e_mu, e_sigma

    data_mean_fit, data_std_fit = fsolve(d_f_loglike, (data_mean, data_std))
    error_data_mean, error_data_std = f_uncertainty(data_mean_fit, data_std_fit)

    return data_mean_fit, error_data_mean, data_std_fit, error_data_std


def plot_nll(
    data: list[int | float],
    x_label: str,
    y_label: str,
    y_label_density: str,
    datalabel: str = "Data",
    errorlabel: str = "Uncertainty",
    fitlabel: str = "Fit",
    meanlabel: str = "Mean",
    mean_unit: str = "",
    suptitle: str = "",
    subtitle1: str = "",
    subtitle2: str = "",
    size: int = 1000,
    capsize: int = 5,
    hist_bins: int = 100,
    hist_color: str = "blue",
    error_color: str = "red",
    fit_color: str = "green",
    mean_color: str = "yellow",
) -> tuple[float, float, float, float]:
    """
    Finds the mean and standard deviation of a (single) physical quantity
    (that is assumed to be a distrubted normally)
    given a function and a list of variables.
    Plots a histogram of the data and the fit.
    Parameters
    ----------
    data : list[int | float]
        A list of data points.

    Returns
    -------
    data_mean_fit : float
        The mean of the data after the fit.
    error_data_mean : float
        The uncertainty of the mean of the data after the fit.
    data_std_fit : float
        The standard deviation of the data after the fit.
    error_data_std : float
        The uncertainty of the standard deviation of the data after the fit.
    """

    def normal(x, mu, sigma):
        return (
            1
            / np.sqrt(2 * np.pi * sigma**2)
            * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        )

    mu, e_mu, sigma, e_sigma = nll(data)
    plt.figure()
    plt.subplot(1, 2, 1)
    counts, bins, _ = plt.hist(data, hist_bins, label=datalabel, color=hist_color)
    plt.errorbar(
        x=bins[:-1] + (bins[1] - bins[0]) / 2,
        y=counts,
        yerr=np.sqrt(counts),
        fmt="",
        capsize=capsize,
        linewidth=0,
        elinewidth=1,
        label=errorlabel,
        ecolor=error_color,
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(subtitle1)
    plt.subplot(1, 2, 2)
    plt.hist(
        data,
        hist_bins,
        density=True,
        label=datalabel + " (normaized)",
        color=hist_color,
    )
    x = np.linspace(bins[0], bins[-1], size)
    plt.plot(
        x,
        normal(x, mu, sigma),
        label=fitlabel,
        color=fit_color,
    )
    plt.xlabel("x")
    plt.ylabel(y_label_density)
    plt.axvline(
        x=mu,
        color=mean_color,
        linestyle="--",
        label=meanlabel + r"$\approx$" + f"{mu:.2f} {mean_unit}",
    )
    plt.legend()
    plt.title(subtitle2)
    plt.suptitle(suptitle)
    plt.show()
    return mu, e_mu, sigma, e_sigma


if __name__ == "__main__":
    data = np.genfromtxt("./TODO/Magnetfeld.csv", delimiter=",", skip_header=1)
    data = np.sqrt(data[:, 1] ** 2 + data[:, 2] ** 2 + data[:, 3] ** 2)
    plot_nll(
        list(data),
        x_label="x",
        y_label="Counts",
        y_label_density="Density",
        errorlabel="Uncertainty",
        fitlabel="Fit",
        size=1000,
        capsize=5,
        mean_unit="mT",
        suptitle="a",
        subtitle1="b",
        subtitle2="c",
    )
