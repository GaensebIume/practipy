import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.special import erfinv


def conficende_bands(
    function: sp.core.function.Function,
    values: dict[str, float | int],
    variables_mean: float | int,
    std: list[int | float],
    variables: list[sp.Symbol],
    interval: list[int],
    function_label: str = "Function",
    function_color: str = "black",
    confidence_levels: list[float | int] = [0.6827, 0.9545, 0.9973],
    colors: list[str] = ["blue", "green", "red"],
    is_sigma: bool = False,
) -> None:
    assert len(confidence_levels) == len(colors)
    assert len(interval) == 3
    if not is_sigma:
        assert all(0 < confidence_level < 1 for confidence_level in confidence_levels)

    def confidence_to_sigma(confidence: float | int) -> float:
        return erfinv(confidence) * np.sqrt(2)

    if not is_sigma:
        confidence_levels = [
            confidence_to_sigma(confidence) for confidence in confidence_levels
        ]
    colors = sorted(colors, key=lambda i: confidence_levels[colors.index(i)])
    confidence_levels.sort()

    function = function.subs(values)  # substitute values in function
    variance: sp.Expr = 0
    for var, std_val in zip(variables, std):
        variance += sp.diff(function, var) ** 2 * std_val**2
    variance = variance.subs(values)
    sp.pprint(variance)
    f_func = sp.lambdify(variables[:-1], function, modules="numpy")
    variance_func = sp.lambdify(variables[:-1], variance, modules="numpy")
    x = np.linspace(interval[0], interval[1], interval[2])
    y_function = f_func(variables_mean)[
        0
    ]  # index 0 needed because variables_mean is a list
    y_values = np.array([y_function.subs(t, t_val) for t_val in x], dtype=float)

    y_variance = variance_func(np.array(std))[0]
    y_std_values = np.array(
        [sp.sqrt(y_variance.subs(t, t_val)) for t_val in x], dtype=float
    )
    plt.figure()
    plt.plot(x, y_values, label=function_label, color=function_color)
    for n_sigma, color in zip(confidence_levels, colors):
        tmp = np.array(n_sigma * y_std_values, dtype=float)
        plt.fill_between(
            x,
            y_values - tmp,
            y_values + tmp,
            label=f"{round(n_sigma,3)}" + r"$\sigma$",
            zorder=-n_sigma,
            color=color,
        )
    plt.legend()
    plt.ylim(0, 1300)
    plt.show()


if __name__ == "__main__":
    x0, y0, theta, v, yf, t, g = sp.symbols("x0 y0 theta v yf t g")
    s_time, s_theta = sp.symbols("sigma_t sigma_theta")
    data = {x0: 100, y0: 50, v: 200, g: -9.81}
    # Define constants
    thetaVal = 45 * np.pi / 180
    stdThetaVal = 1 * np.pi / 180

    timeInt = [0, 35, 200]  # last entry is the number of points generated
    levels = [1, 2, 3, 4, 5, 6, 7, 8, 1 / 4, 1 / 2]
    variable = [theta, t]
    stds = [s_theta, s_time]
    std_values = [stdThetaVal, 0]
    f = y0 + v * sp.sin(theta) * t + sp.Rational(1, 2) * g * t**2

    conficende_bands(
        function=f,
        values=data,
        std=std_values,
        variables=variable,
        interval=timeInt,
        confidence_levels=levels,
        variables_mean=[thetaVal],
        is_sigma=True,
        colors=[
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "black",
        ],
    )
