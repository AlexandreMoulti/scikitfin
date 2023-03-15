import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
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

@np.vectorize
def G(tau: float, kappa: float) -> float:
    """
    function used to compute ZC prices
    Parameters
    ----------
    tau : np.ndarray
        time steps.
    kappa : float
        parameters of the model.
    Returns
    -------
    np.ndarray.
    """
    return 1e-10 if tau < 1e-10 else (1 - np.exp(-kappa * tau)) / kappa

def mean(t: float, x: float, kappa: float, sigma: float) -> np.ndarray:
    """
    Mean of the stochastic process x
    Parameters
    ----------
    t : float
        current time.
    x : float
        stochastic process x ( x = r - f )
    kappa : float
        parameters of the model.
    sigma : float
        volatility of the short rate
    Returns
    -------
    np.ndarray.
    """
    return x * np.exp(-kappa * t) + 0.5 * sigma ** 2 * G(t, kappa) ** 2


def variance(t: float, kappa: float, sigma: float) -> float:
    """
    The variance of the stochastic process x
    Parameters
    ----------
    t : float
        current time.
    kappa : float
        parameters of the model.
    sigma : float
        volatility of the short rate
    Returns
    -------
    float.
    """
    return 0.5 * sigma ** 2 * G(2 * t, kappa)


def zc_bond_volatility(t: float, maturities: np.ndarray, tenor: float, kappa: float, sigma: float) -> np.ndarray:
    """
    Volatility of ZC

    Parameters
    ----------
    t : float
        current time.
    maturities : np.ndarray
        expiry of the ZC bond
    tenor : float
        tenor of the underlying
    kappa : float
        parameters of the model.
    sigma : float
        volatility of the short rate
    Returns
    -------
    np.ndarray.
    """
    return sigma ** 2 * G(tenor, kappa) ** 2 * G(maturities - t, 2*kappa)


def price_zc(x: float, t: float, maturity: np.ndarray, func_spot_zc, kappa: float, sigma: float) -> np.ndarray:
    """
    ZC price
    Parameters
    ----------
    x : float
        stochastic process x ( x = r - f )
    t : float
        current time.
    maturity : np.ndarray
        expiry of the ZC bond
    func_spot_zc : function that spot ZC prices
    kappa : float
        parameters of the model.
    sigma : float
        volatility of the ZC
    Returns
    -------
    np.ndarray.
    """
    return (func_spot_zc(maturity) / func_spot_zc(t)) * \
           np.exp(-x * G(maturity - t, kappa) - 0.5 * variance(t, kappa, sigma) * G(maturity - t, kappa) ** 2)


def black_tools(x: float, t: float, maturity: np.ndarray, tenor: np.ndarray, strike: np.ndarray, func_spot_zc,
                kappa: float, sigma: float) -> Tuple:
    """
    black tools
    Parameters
    ----------
    x : float
        stochastic process x ( x = r - f )
    t : float
        current time.
    maturity: np.ndarray
        expiry of the option
    tenor: np.ndarray
        expiry of the zc underlying
    strike : np.ndarray
        strike of the option
    func_spot_zc : function that spot ZC prices
    kappa : float
        parameters of the model.
    sigma : float
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


def price_zc_call(x: float, t: float, maturity: np.ndarray, tenor: np.ndarray,
                  strike: np.ndarray, func_spot_zc, kappa: float, sigma: float) -> np.ndarray:
    """
    price of call option on ZC

    Parameters
    ----------
    x : float
        stochastic process x ( x = r - f )
    t : float
        current time.
    maturity: np.ndarray
        expiry of the option
    tenor: np.ndarray
        expiry of the zc underlying
    strike : np.ndarray
        strike of the option
    func_spot_zc : function that spot ZC prices
    kappa : float
        parameters of the model.
    sigma : float
        volatility of the short rate
    Returns
    -------
    np.ndaaray.
    """
    p_up, p_down, d_positif, d_negatif = black_tools(x, t, maturity, tenor, strike, func_spot_zc, kappa, sigma)
    return p_up * norm.cdf(d_positif) - p_down * strike * norm.cdf(d_negatif)


def price_zc_put(x: float, t: float, maturity: np.ndarray,
                 tenor: np.ndarray, strike: np.ndarray, func_spot_zc,kappa: float, sigma: float) -> np.ndarray:
    """
    price of put option on ZC

    Parameters
    ----------
    x : float
        stochastic process x ( x = r - f )
    t : float
        current time.@
    maturity: np.ndarray
        expiry of the option
    tenor: np.ndarray
        expiry of the zc underlying
    strike : np.ndarray
        strike of the option
    func_spot_zc : function that spot ZC prices
    kappa : float
        parameters of the model.
    sigma : float
        volatility of the short rate
    Returns
    -------
    np.ndarray.
    """
    p_up, p_down, d_positif, d_negatif = black_tools(x, t, maturity, tenor, strike, func_spot_zc,kappa, sigma)
    return p_down * strike * norm.cdf(-d_negatif) - p_up * norm.cdf(-d_positif)


@np.vectorize
def price_swaption(x: float, t: float, array_of_tuple: np.ndarray,
                   strike: np.ndarray, func_spot_zc, kappa: float, sigma: float, dt: float = 0.5) -> np.ndarray:
    """
    price of swaption

    Parameters
    ----------
    x : float
        state process x at current time t
    t : float
        current time.
    array_of_tuple: np.ndarray
        list of tuple (maturity, tenor).
    strike: np.ndarray
        list of strike of the option.
    func_spot_zc : function that spot ZC prices
    kappa : float
        parameters of the model.
    sigma : float
        volatility of the short rate
    dt : float
        time step convention for the swap
    Returns
    -------
    np.ndarray.
    """
    # schedule = (t0,t1,...,tN)
    maturity, tenor = array_of_tuple[0], array_of_tuple[1]
    schedule = np.arange(t+maturity, t+maturity+tenor+dt, dt)

    solution = optimize.root(x_root_function, 0.05, args=(strike, schedule, dt, kappa, sigma))
    optimal_x = solution.x
    vec_k = price_zc(optimal_x, schedule[0], schedule[1:], func_spot_zc, kappa, sigma)
    return strike * np.sum(dt * price_zc_put(x, t, maturity, schedule[1:] - schedule[0], vec_k, func_spot_zc, kappa,
           sigma)) + price_zc_put(x, t, maturity, tenor, vec_k[-1], func_spot_zc, kappa, sigma)


def x_root_function(x, *args):
    strike, schedule, vec_dt, kappa, sigma = args
    return -1 + price_zc(x, schedule[0], schedule[-1], kappa, sigma) + strike * np.sum(vec_dt * price_zc(x, schedule[0], schedule[1:], kappa, sigma))




def compute_model_premiums(ircurve, swaptionsurface, kappa: float, sigma: float, dt: float = 0.5):
    """
    Computes premiums premium of the ATM swpations matrix
    Parameters
    ----------
    ircurve : IRCUVE object
    swaptionsurface : np.array (maturity, tenor, value)
    kappa
    sigma
    dt : float
        time step convention for the swap

    Returns
    -------

    """
    x, t = 0
    premiums_market = swaptionsurface['value']
    vect_strike_ATM = ircurve.forward_swap_rate(swaptionsurface, dt)
    premiums_model = price_swaption(x, t, swaptionsurface, vect_strike_ATM, ircurve.func_zc_prices,
                                         kappa, sigma, dt)

    return premiums_model, premiums_market

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

    premiums_model, premiums_mkt = compute_model_premiums(ircurve, swaptionsurface, kappa, sigma, dt)

    return np.mean((premiums_mkt/premiums_model - 1) ** 2)

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
    premiums_model, premiums_mkt = compute_model_premiums(ircurve, swaptionsurface, kappa, sigma, dt)

    return max_error - np.nanmax((premiums_mkt/premiums_model - 1) ** 2)

########################################################################################################################
#
#                                   HULL WHITE
#
########################################################################################################################


class HullWhite(object):

    def __init__(self, kappa: float, sigma: float) -> None:
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

    def price_zc_spot(self, maturity: float):
        """

        Parameters
        ----------
        maturity : float
            maturity of the zero coupon

        Returns
        -------
            price of a zero coupon
        """
        return self.func_spot_zc(maturity)

    def price_zc(self, x: float, t: float, maturity: np.ndarray) -> np.ndarray:
        """
        ZC price
        Parameters
        ----------
        x : float
            stochastic process x ( x = r - f )
        t : float
            current time.
        maturity : np.ndarray
            expiry of the ZC bond
        Returns
        -------
        np.ndarray.
        """

        return price_zc(x, t, maturity, self.func_spot_zc, self.kappa, self.sigma)

    def price_zc_call(self, x: float, t: float, maturity: np.ndarray, tenor: np.ndarray,
                      strike: np.ndarray) -> np.ndarray:
        """
        price of call option on ZC

        Parameters
        ----------
        x : float
            stochastic process x ( x = r - f )
        t : float
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

    def price_zc_put(self, x: float, t: float, maturity: np.ndarray,
                     tenor: np.ndarray, strike: np.ndarray) -> np.ndarray:
        """
        price of put option on ZC

        Parameters
        ----------
        x : float
            stochastic process x ( x = r - f )
        t : float
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

    def price_swaption(self, x: float, t: float, maturity: float, tenor: float,
                       strike: float, dt: float = 0.5) -> float:
        """
        price of swaption

        Parameters
        ----------
        x : float
            state process x at current time t
        t : float
            current time.
        maturity : float
        tenor : float
        strike: float
            strike of the option.
        dt : float
            time step convention for the swap
        Returns
        -------
        np.ndarray.
        """

        return price_swaption(x, t, (maturity, tenor), strike, self.func_spot_zc, self.kappa, self.sigma, dt)

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
        simulations[:, 0] = self.init
        for j in range(1, self.simulations.shape[1]):
            simulations[:, j] = simulations[:, j - 1] * exp(-dt * self.kappa) + self.mean(dt) + \
                                np.sqrt(self.variance(dt)) * dN[:, j]
        return simulations + self.func_instant_forwards(time_grid)

    def fit(self, ircurve, swaptionsurface, bounds, max_error=0.1, n=256, iters=2, minimize_every_iter=True,
            maxiter=200, dt=0.5):

        # calibrates kappa sigma to the data
        self.func_instant_forwards = ircurve.func_instant_forwards
        self.func_spot_zc = ircurve.func_zc_prices

        constraints = [{'type': 'ineq', 'fun': constraint_func_max_error, 'args': (max_error, ircurve, swaptionsurface, dt)}]
        params = shgo(loss_function, bounds=bounds, n=n, iters=iters, minimizer_kwargs={'constraints': constraints},
                      sampling_method='sobol', options={'minimize_every_iter': minimize_every_iter},
                      args=(ircurve, swaptionsurface, dt))

        self.kappa, self.sigma = params.x

        return params