from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import integrate, signal


def fourier_series(
    function: Callable,
    x_min: float | int,
    x_max: float | int,
    period: int | float,
    plts: list[int],
    x_label: str,
    y_label: str,
    filename: str = "fourier.png",
    x_label_index: str = "Index k",
    y_label_index_a: str = "Fourier coefficients $A_k$",
    y_label_index_b: str = "Fourier coefficients $B_k$",
    coefficient_color: str = "blue",
    coefficient_error_color: str = "red",
    data_color: str = "blue",
    period_color: str = "red",
    period_label: str = "Period",
    signal_label: str = "Signal",
    coefficient_error_label: str = "Coefficient error",
    coefficient_label: str = "Coefficient",
    fourrier_colors: list[str] = [],
    colors_passed: bool = False,
    points: int = 1000,
    padding: float = 1.1,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Plots the fourier series (after multiple steps (use plts t odefine how many))
    and its coefficients for a given function.

    Args:
        function (Callable):
            The function to be plotted.
        x_min (float | int):
            The function to be plotted.
        x_max (float | int):
            The maximum value of the x-axis.
        period (int | float):
            The period of the function.
        plts (list[int]):
            The orders of the fourier series to be plotted.
        x_label (str):
            The label for the x-axis.
        y_label (str):
            The label for the y-axis.
        filename (str, optional):
            The filename for the plot. Defaults to "fourier.png".
        x_label_index (str, optional):
            The label for the x-axis of the coefficients. Defaults to "Index k".
        y_label_index_a (str, optional):
            The label for the y-axis of the a_k coefficients. Defaults to "Fourier coefficients $A_k$".
        y_label_index_b (str, optional):
            The label for the y-axis of the b_k coefficients. Defaults to "Fourier coefficients $B_k$".
        coefficient_color (str, optional):
            The color for the coefficients. Defaults to "blue".
        coefficient_error_color (str, optional):
            The color for the coefficient error. Defaults to "red".
        data_color (str, optional):
            The color for the data. Defaults to "blue".
        period_color (str, optional):
            The color for the period. Defaults to "red".
        period_label (str, optional):
            The label for the period. Defaults to "Period".
        signal_label (str, optional):
            The label for the signal. Defaults to "Signal".
        coefficient_error_label (str, optional):
            The label for the coefficient error. Defaults to "Coefficient error".
        coefficient_label (str, optional):
            The label for the coefficients. Defaults to "Coefficient".
        fourrier_colors (list[str], optional):
        The label for the coefficients. Defaults to "Coefficient".
        colors_passed (bool, optional):
            Whether the colors for the fourier series have been passed. Defaults to False.
            Must be True if colorargs are passed, otherwise the colors are ignored.
        points (int, optional):
            The number of points for the x-axis. Defaults to 1000.
        padding (float, optional):
            The padding for the y-axis. Defaults to 1.1.

        Returns:
            tuple[list[float], list[float], list[float], list[float]]:
                The fourier coefficients for the different orders.
                1. a_k Values
                2. b_k Values
                3. a_k Errors
                4. b_k Errors
    """
    if colors_passed:
        assert len(fourrier_colors) == len(plts)

    def cos_function(t: float | int, k: int) -> float:
        return function(t) * np.cos(2 * np.pi * t * k / period)

    def sin_function(t: float | int, k: int) -> float:
        return function(t) * np.sin(2 * np.pi * t * k / period)

    def ak(k: int) -> tuple[float, float]:
        return (
            2.0 / period * np.array(integrate.quad(cos_function, 0, period, args=(k)))
        )

    def bk(k: int) -> tuple[float, float]:
        return (
            2.0 / period * np.array(integrate.quad(sin_function, 0, period, args=(k)))
        )

    def split_index(function, index: int) -> tuple[list[float], list[float]]:
        value = []
        error = []
        for k in range(index):
            pair = function(k)
            value.append(pair[0])
            error.append(pair[1])
        return value, error

    def plot(plot, xdata, ydata, xlabel, ylabel, label, color, vlines: bool = False):
        plt.subplot(2, 2, plot)
        if vlines:
            plt.vlines(xdata, [0], ydata, lw=3, color=color, label=label)
        else:
            if color == "":
                plt.plot(xdata, ydata, lw=2, label=label)
            else:
                plt.plot(xdata, ydata, lw=2, color=color, label=label)
            plt.ylim(padding * min(ydata), padding * max(ydata))
            plt.xlim(x_min, x_max)
            plt.legend()
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    x_values = np.linspace(x_min, x_max, points)
    data = function(x_values)
    max_plt = max(plts)
    ak_values, ak_errors = split_index(ak, max_plt)
    bk_values, bk_error = split_index(bk, max_plt)

    fouries_values = np.zeros_like(data)

    xk = range(max_plt)
    # Plot signal
    plot(1, x_values, data, x_label, y_label, signal_label, data_color)

    # Plot fourier coefficients Ak
    plot(
        2,
        xk,
        ak_values,
        x_label_index,
        y_label_index_a,
        coefficient_label,
        coefficient_color,
        vlines=True,
    )
    plt.errorbar(
        xk,
        ak_values,
        label=coefficient_error_label,
        yerr=ak_errors,
        color=coefficient_error_color,
        linewidth=0,
        elinewidth=1,
    )
    plt.legend()

    # Plot fourier coefficients Bk
    plot(
        3,
        xk,
        bk_values,
        x_label_index,
        y_label_index_b,
        coefficient_label,
        coefficient_color,
        vlines=True,
    )
    plt.errorbar(
        xk,
        bk_values,
        label=coefficient_error_label,
        yerr=bk_error,
        color=coefficient_error_color,
        linewidth=0,
        elinewidth=1,
    )
    plt.legend()

    # Plot fourier
    plot(4, x_values, data, x_label, y_label, signal_label, data_color)

    plt.axvline(x=period, color=period_color, label=period_label, linestyle="--")
    plt.axvline(x=0, color=period_color, linestyle="--")
    for k in range(max_plt):
        fouries_values += ak_values[k] * np.cos(
            2 * np.pi * k * x_values / period
        ) + bk_values[k] * np.sin(2 * np.pi * k * x_values / period)
        if k in plts:
            if colors_passed:
                plot(
                    4,
                    x_values,
                    fouries_values,
                    x_label,
                    y_label,
                    f"k = {k}",
                    fourrier_colors[k],
                )
                # plt.plot(
                #     x_values, fouries_values, label=f"k = {k}", color=fourrier_colors[k]
                # )
            else:
                plot(
                    4, x_values, fouries_values, x_label, y_label, f"k = {k}", color=""
                )
                # plt.plot(x_values, fouries_values, label=f"k = {k}")
    plt.savefig(filename)
    plt.show()

    return ak_values, bk_values, ak_errors, bk_error


def complex_fourier(
    function: sp.Expr,
    var: sp.Symbol,
    k: int | float | None = None,
) -> tuple[sp.Expr, sp.Expr] | sp.Expr:
    zeta: sp.Symbol = sp.symbols("zeta")
    integrand: Callable = function * sp.exp(-1j * zeta * var)
    f_hat = 1 / sp.sqrt(2 * sp.pi) * sp.integrate(integrand, (var, -sp.oo, sp.oo))
    return f_hat, zeta if k is None else f_hat.subs(zeta, k)


def complex_fourier_inv(
    function: sp.Expr,
    var: sp.Symbol,
    k: int | float | None = None,
) -> tuple[sp.Expr, sp.Expr] | sp.Expr:
    zeta: sp.Symbol = sp.symbols("zeta")
    integrand: Callable = function * sp.exp(1j * zeta * var)
    f_hat = 1 / sp.sqrt(2 * sp.pi) * sp.integrate(integrand, (var, -sp.oo, sp.oo))
    return f_hat, zeta if k is None else f_hat.subs(zeta, k)


if __name__ == "__main__":
    T = 8.0

    fourier_series(
        lambda t: signal.square(2 * np.pi * t / T),
        -4,
        20.0,
        T,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        "",
        "",
    )
