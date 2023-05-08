import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Union, List, Any, Tuple, Callable
from math import *
from scipy.stats import norm
from scipy import optimize
from scipy.optimize import shgo

########################################################################################################################
#
#                                   STATIC METHODS
#
########################################################################################################################


def G(tau: ArrayLike,
      kappa: ArrayLike
      ) -> float:
    """
    function used to compute ZC prices
    Parameters
    ----------
    tau : np.ndarray
        time steps.
    kappa : ArrayLike
        parameters of the model.
    Returns
    -------
    np.ndarray.
    """
    if np.isscalar(tau):
        return tau if kappa*tau < 1e-10 else (1 - exp(-kappa * tau)) / kappa
    else:
        return np.where(kappa*tau < 1e-10, tau, (1 - np.exp(-kappa * tau)) / kappa)


def mean(t: ArrayLike,
         x: ArrayLike,
         kappa: ArrayLike,
         sigma: ArrayLike
         ) -> ArrayLike:
    """
    Mean of the stochastic process x
    Parameters
    ----------
    t : ArrayLike
        current time.
    x : ArrayLike
        stochastic process x ( x = r - f )
    kappa : ArrayLike
        parameters of the model.
    sigma : ArrayLike
        volatility of the short rate
    Returns
    -------
    np.ndarray.
    """
    return x * np.exp(-kappa * t) + 0.5 * sigma ** 2 * G(t, kappa) ** 2

def variance(t: ArrayLike,
             kappa: ArrayLike,
             sigma: ArrayLike
             ) -> ArrayLike:
    """
    The variance of the stochastic process x
    Parameters
    ----------
    t : ArrayLike
        current time.
    kappa : ArrayLike
        parameters of the model.
    sigma : ArrayLike
        volatility of the short rate
    Returns
    -------
    ArrayLike.
    """
    return 0.5 * sigma ** 2 * G(2 * t, kappa)


def zc_bond_volatility(t: ArrayLike,
                       maturities: ArrayLike,
                       tenor: ArrayLike,
                       kappa: ArrayLike,
                       sigma: ArrayLike) -> ArrayLike:
    """
    Volatility of ZC

    Parameters
    ----------
    t : ArrayLike
        current time.
    maturities : np.ndarray
        expiry of the ZC bond
    tenor : ArrayLike
        tenor of the underlying
    kappa : ArrayLike
        parameters of the model.
    sigma : ArrayLike
        volatility of the short rate
    Returns
    -------
    ArrayLike
    """
    return sigma ** 2 * G(tenor, kappa) ** 2 * G(maturities - t, 2*kappa)


def price_zc(x: ArrayLike,
             t: ArrayLike,
             maturity: ArrayLike,
             func_spot_zc,
             kappa: ArrayLike,
             sigma: ArrayLike) -> ArrayLike:
    """
    ZC price
    Parameters
    ----------
    x : ArrayLike
        stochastic process x ( x = r - f )
    t : ArrayLike
        current time.
    maturity : np.ndarray
        expiry of the ZC bond
    func_spot_zc : function that spot ZC prices
    kappa : ArrayLike
        parameters of the model.
    sigma : ArrayLike
        volatility of the ZC
    Returns
    -------
    np.ndarray.
    """
    return (func_spot_zc(maturity) / func_spot_zc(t)) * \
           np.exp(-x * G(maturity - t, kappa) - 0.5 * variance(t, kappa, sigma) * G(maturity - t, kappa) ** 2)


def black_tools(x: ArrayLike,
                t: ArrayLike,
                maturity: ArrayLike,
                tenor: ArrayLike,
                strike: ArrayLike,
                func_spot_zc,
                kappa: ArrayLike,
                sigma: ArrayLike
                ) -> Tuple:
    """
    black tools
    Parameters
    ----------
    x : ArrayLike
        stochastic process x ( x = r - f )
    t : ArrayLike
        current time.
    maturity: np.ndarray
        expiry of the option
    tenor: np.ndarray
        expiry of the zc underlying
    strike : np.ndarray
        strike of the option
    func_spot_zc : function that spot ZC prices
    kappa : ArrayLike
        parameters of the model.
    sigma : ArrayLike
        volatility of the short rate
    Returns
    -------
    np.ndarray.
    """
    v = zc_bond_volatility(t, maturity, tenor, kappa, sigma)
    p_up = price_zc(x, t, maturity + tenor, func_spot_zc, kappa, sigma)
    p_down = price_zc(x, t, maturity, func_spot_zc, kappa, sigma)
    d_positif = (np.log(p_up / (strike * p_down)) + v / 2) / np.sqrt(v)
    d_negatif = (np.log(p_up / (strike * p_down)) - v / 2) / np.sqrt(v)
    return p_up, p_down, d_positif, d_negatif

def price_zc_call(x: ArrayLike,
                  t: ArrayLike,
                  maturity: ArrayLike,
                  tenor: ArrayLike,
                  strike: ArrayLike,
                  func_spot_zc,
                  kappa: ArrayLike,
                  sigma: ArrayLike
                  ) -> np.ndarray:
    """
    price of call option on ZC

    Parameters
    ----------
    x : ArrayLike
        stochastic process x ( x = r - f )
    t : ArrayLike
        current time.
    maturity: np.ndarray
        expiry of the option
    tenor: np.ndarray
        expiry of the zc underlying
    strike : np.ndarray
        strike of the option
    func_spot_zc : function that spot ZC prices
    kappa : ArrayLike
        parameters of the model.
    sigma : ArrayLike
        volatility of the short rate
    Returns
    -------
    np.ndaaray.
    """
    p_up, p_down, d_positif, d_negatif = black_tools(x, t, maturity, tenor, strike, func_spot_zc, kappa, sigma)
    return p_up * norm.cdf(d_positif) - p_down * strike * norm.cdf(d_negatif)


def price_zc_put(x: ArrayLike,
                 t: ArrayLike,
                 maturity: ArrayLike,
                 tenor: ArrayLike,
                 strike: ArrayLike,
                 func_spot_zc,
                 kappa: ArrayLike,
                 sigma: ArrayLike
                 ) -> np.ndarray:
    """
    price of put option on ZC

    Parameters
    ----------
    x : ArrayLike
        stochastic process x ( x = r - f )
    t : ArrayLike
        current time.@
    maturity: np.ndarray
        expiry of the option
    tenor: np.ndarray
        expiry of the zc underlying
    strike : np.ndarray
        strike of the option
    func_spot_zc : function that spot ZC prices
    kappa : ArrayLike
        parameters of the model.
    sigma : ArrayLike
        volatility of the short rate
    Returns
    -------
    np.ndarray.
    """
    p_up, p_down, d_positif, d_negatif = black_tools(x, t, maturity, tenor, strike, func_spot_zc,kappa, sigma)
    return p_down * strike * norm.cdf(-d_negatif) - p_up * norm.cdf(-d_positif)


def price_swaption(x: ArrayLike,
                   t: ArrayLike,
                   maturity: ArrayLike,
                   tenor:ArrayLike,
                   strike: ArrayLike,
                   func_spot_zc,
                   kappa: ArrayLike,
                   sigma: ArrayLike,
                   dt: ArrayLike = 0.5
                   ) -> ArrayLike:
    """
    price of swaption

    Parameters
    ----------
    x : ArrayLike
        state process x at current time t
    t : ArrayLike
        current time.
    array_of_tuple: np.ndarray
        list of tuple (maturity, tenor).
    strike: np.ndarray
        list of strike of the option.
    func_spot_zc : function that spot ZC prices
    kappa : ArrayLike
        parameters of the model.
    sigma : ArrayLike
        volatility of the short rate
    dt : ArrayLike
        time step convention for the swap
    Returns
    -------
    np.ndarray.
    """
    # schedule = (t0,t1,...,tN)
    schedule = np.arange(t+maturity, t+maturity+tenor+dt, dt)
    solution = optimize.root(x_root_function, 0.05, args=(strike, schedule, func_spot_zc, dt, kappa, sigma))
    optimal_x = solution.x
    vec_k = price_zc(optimal_x, schedule[0], schedule[1:], func_spot_zc, kappa, sigma)
    return strike * np.sum(dt * price_zc_put(x, t, maturity, schedule[1:] - schedule[0], vec_k, func_spot_zc, kappa, sigma)) \
           + price_zc_put(x, t, maturity, tenor, vec_k[-1], func_spot_zc, kappa, sigma)

price_swaption = np.vectorize(price_swaption, excluded=['func_spot_zc'])

def x_root_function(x, *args):
    strike, schedule, func_spot_zc, vec_dt, kappa, sigma = args
    return -1 + price_zc(x, schedule[0], schedule[-1], func_spot_zc, kappa, sigma) + \
            strike * np.sum(vec_dt * price_zc(x, schedule[0], schedule[1:], func_spot_zc, kappa, sigma))

def loss_function(params, *args):
    """
    Compute the mean square relative erros between swaptions market price and model price
    Parameters
    ----------
    params
    args

    Returns
    -------

    """

    kappa, sigma = params
    ircurve, swaptionsurface, dt = args
    premiums_market = swaptionsurface.prices
    vect_strike_atm = ircurve.forward_swap_rate(swaptionsurface.expiries, swaptionsurface.tenors, dt)
    premiums_model = price_swaption(0, 0, swaptionsurface.expiries, swaptionsurface.tenors, vect_strike_atm,
                                    ircurve.func_zc_prices, kappa, sigma, dt)
    return np.mean((premiums_market/premiums_model - 1) ** 2)

def constraint_func_max_error(params, *args):
    """
    Test if the calibration verifies the constraint
    Parameters
    ----------
    params
    args

    Returns
    -------

    """
    kappa, sigma = params
    max_error, ircurve, swaptionsurface, dt = args
    premiums_market = swaptionsurface.prices
    vect_strike_atm = ircurve.forward_swap_rate(swaptionsurface.expiries, swaptionsurface.tenors, dt)
    premiums_model = price_swaption(0, 0, swaptionsurface.expiries, swaptionsurface.tenors, vect_strike_atm,
                                    ircurve.func_zc_prices, kappa, sigma, dt)
    return max_error - np.nanmax((premiums_market/premiums_model - 1) ** 2)

########################################################################################################################
#
#                                   HULL WHITE
#
########################################################################################################################


class HullWhite(object):

    def __init__(self, kappa: ArrayLike=0.1, sigma: ArrayLike=0.02) -> None:
        """
        Interest Rates Gaussian 1 Factor model with constant parameters
        Parameters
        ----------
        kappa : TYPE
            DESCRIPTION.
        sigma : TYPE
            DESCRIPTION.
        func_instant_forwards : TYPE, optional
            DESCRIPTION. The default is lambda x:0.
        Returns
        -------
        None.
        """
        self.kappa = kappa
        self.sigma = sigma
        self.func_instant_forwards = lambda x: 0
        self.func_spot_zc = lambda x: 1

    def price_zc_spot(self, maturity: ArrayLike):
        """

        Parameters
        ----------
        maturity : ArrayLike
            maturity of the zero coupon

        Returns
        -------
            price of a zero coupon
        """
        return self.func_spot_zc(maturity)

    def price_zc(self, x: ArrayLike, t: ArrayLike, maturity: ArrayLike) -> np.ndarray:
        """
        ZC price
        Parameters
        ----------
        x : ArrayLike
            stochastic process x ( x = r - f )
        t : ArrayLike
            current time.
        maturity : np.ndarray
            expiry of the ZC bond
        Returns
        -------
        np.ndarray.
        """

        return price_zc(x, t, maturity, self.func_spot_zc, self.kappa, self.sigma)

    def price_zc_call(self, x: ArrayLike, t: ArrayLike, maturity: ArrayLike, tenor: ArrayLike,
                      strike: ArrayLike) -> np.ndarray:
        """
        price of call option on ZC

        Parameters
        ----------
        x : ArrayLike
            stochastic process x ( x = r - f )
        t : ArrayLike
            current time.
        maturity: np.ndarray
            expiry of the option
        tenor: np.ndarray
            expiry of the zc underlying
        strike : np.ndarray
            strike of the option
        Returns
        -------
        np.ndaaray.
        """

        return price_zc_call(x, t, maturity, tenor, strike, self.func_spot_zc, self.kappa, self.sigma)

    def price_zc_put(self, x: ArrayLike, t: ArrayLike, maturity: ArrayLike,
                     tenor: ArrayLike, strike: ArrayLike) -> np.ndarray:
        """
        price of put option on ZC

        Parameters
        ----------
        x : ArrayLike
            stochastic process x ( x = r - f )
        t : ArrayLike
            current time.@
        maturity: np.ndarray
            expiry of the option
        tenor: np.ndarray
            expiry of the zc underlying
        strike : np.ndarray
            strike of the option
        Returns
        -------
        np.ndarray.
        """

        return price_zc_put(x, t, maturity, tenor, strike, self.func_spot_zc, self.kappa, self.sigma)

    def price_swaption(self, x: ArrayLike, t: ArrayLike, maturity: ArrayLike, tenor: ArrayLike,
                       strike: ArrayLike, dt: ArrayLike = 0.5) -> ArrayLike:
        """
        price of swaption

        Parameters
        ----------
        x : ArrayLike
            state process x at current time t
        t : ArrayLike
            current time.
        maturity : ArrayLike
        tenor : ArrayLike
        strike: ArrayLike
            strike of the option.
        dt : ArrayLike
            time step convention for the swap
        Returns
        -------
        np.ndarray.
        """
        return price_swaption(x, t, maturity, tenor, strike, self.func_spot_zc, self.kappa, self.sigma, dt)

    def simulate(self, number_simulations, horizon, dt, dN):
        """
        Simulates Monte Carlo paths for the short rate $r_t$
        Parameters
        ----------
        grid : TYPE
            DESCRIPTION.
        dN : TYPE
            DESCRIPTION.
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        time_grid = np.arange(0, horizon+dt,dt)
        simulations = np.ndarray((number_simulations, len(time_grid)))
        simulations[:, 0] = 0
        for j in range(1, simulations.shape[1]):
            simulations[:, j] = simulations[:, j - 1] * exp(-dt * self.kappa) + self.mean(dt) + \
                                np.sqrt(self.variance(dt)) * dN[:, j]
        return simulations + self.func_instant_forwards(time_grid)

    def fit(self, ircurve, swaptionsurface=None, params=None, bounds=None, dt=0.5):

        # calibrates kappa sigma to the data
        self.func_instant_forwards = ircurve.func_instant_forwards
        self.func_spot_zc = ircurve.func_zc_prices

        if swaptionsurface is not None:
            params = optimize.minimize(loss_function, params, method='SLSQP', bounds=bounds,
                                       options={"maxiter": 100, 'ftol': 5e-6},
                                       args=(ircurve, swaptionsurface, dt))
            self.kappa, self.sigma = params.x
        return params