import sympy as sp


def analytical_error_propagation(
    function,
    values: dict[str, int | float],
    variables: list[sp.Symbol],
    correlation: list[list[sp.Symbol]],
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, list[sp.Expr]]:
    """
    Calculates the mean, mean value, standard deviation and standard deviation value of a physical
    quantity with known dependency on other physical quantities
    :param function:
        The function of the physical quantity.
    :param values:
        The values of the other physical quantities.
    :param variables:
        The other physical quantities.
    :param correlation:
        The correlation between the other physical quantities.
        Is a list of lists, where the diagonal elements are the variances (sigma**2)
        of the pyhsical quantity at the index.
        The off-diagonal elements are the correlation between the physical quantities.
        The physical quantities must be in the same order as the variables.
        The "Matrix" must be symmetric.
    :return:
        The mean, mean value, standard deviation, standard deviation value of
        the physical quantity and the partial derivatives.
    """
    mean = function
    mean_val = function.subs(values)
    partials = []
    for variable in variables:
        partials.append(sp.diff(function, variable))

    variance = 0
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i == j:
                variance += (
                    partials[i] ** 2 * correlation[i][i]
                )  # diagonal elements in matrix are variances
            else:
                assert correlation[i][j] == correlation[j][i]
                variance += (
                    partials[i]
                    * partials[j]
                    * correlation[i][j]
                    * sp.sqrt(correlation[i][i] * correlation[j][j])
                )

    std = sp.sqrt(variance).simplify()
    std_val = std.subs(values).simplify()
    return mean, mean_val, std, std_val, partials


if __name__ == "__main__":
    m, v = sp.symbols("m v")
    sv, sm, rsm = sp.symbols("sigma_v sigma_m rho_vm")
    # rms = 0, because mass and velocity should not be correlated.
    data: dict[str, int | float] = {
        "m": 10,
        "v": 15,
        "sigma_m": 1,
        "sigma_v": 1,
        "rho_vm": 0,
    }
    f = 1 / 2 * m * v**2
    var = [m, v]
    cor: list[list[sp.Symbol]] = [[sm, rsm], [rsm, sv]]
    sp.pprint(
        analytical_error_propagation(
            function=f,
            values=data,
            correlation=cor,
            variables=var,
        )
    )
