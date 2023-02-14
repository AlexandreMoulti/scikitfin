import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Literal, Union, List, Any, Tuple, Callable
from math import *
from scipy.stats import norm
from scipy import optimize


########################################################################################################################
#
# STATIC METHODS
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


def root_function(x, *args):
    price_zc, strike, schedule, vec_dt, kappa, sigma = args
    return -1 + price_zc(x, schedule[0], schedule[-1], kappa, sigma) + strike * np.sum(vec_dt * price_zc(x, schedule[0], schedule[1:], kappa, sigma))


######## Hull White ###############

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

    def price_zc(self, x: float, t: float, maturity: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
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
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the ZC
        Returns
        -------
        np.ndarray.
        """
        return (self.func_spot_zc(maturity) / self.func_spot_zc(t)) * \
               np.exp(-x * G(maturity - t, kappa) - 0.5 * variance(t, kappa, sigma) * G(maturity - t, kappa) ** 2)

    def black_tools(self, x: float, t: float, maturity: np.ndarray,
                    tenor: np.ndarray, strike: np.ndarray, kappa: float, sigma: float) -> Tuple:
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
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the short rate
        Returns
        -------
        np.ndarray.
        """
        v = zc_bond_volatility(t, maturity, tenor, kappa, sigma)
        p_up = self.price_zc(x, t, maturity + tenor, kappa, sigma)
        p_down = self.price_zc(x, t, maturity, kappa, sigma)
        d_positif = (np.log(p_up / (strike * p_down)) + v / 2) / np.sqrt(v)
        d_negatif = (np.log(p_up / (strike * p_down)) - v / 2) / np.sqrt(v)
        return p_up, p_down, d_positif, d_negatif

    def price_zc_call(self, x: float, t: float, maturity: np.ndarray, tenor: np.ndarray,
                      strike: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
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
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the short rate
        Returns
        -------
        np.ndaaray.
        """
        p_up, p_down, d_positif, d_negatif = self.black_tools(x, t, maturity, tenor, strike, kappa, sigma)
        return p_up * norm.cdf(d_positif) - p_down * strike * norm.cdf(d_negatif)

    def price_zc_put(self, x: float, t: float, maturity: np.ndarray,
                     tenor: np.ndarray, strike: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
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
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the short rate
        Returns
        -------
        np.ndarray.
        """
        p_up, p_down, d_positif, d_negatif = self.black_tools(x, t, maturity, tenor, strike, kappa, sigma)
        return p_down * strike * norm.cdf(-d_negatif) - p_up * norm.cdf(-d_positif)

    def price_swaption(self, x: float, t: float, schedule: np.ndarray,
                       strike: float, kappa: float, sigma: float) -> float:
        """
        price of swaption

        Parameters
        ----------
        x : float
            state process x at current time t
        t : float
            current time.
        schedule: Tuple
            Observation dates (T0, T1, ...)
        strike: float
            strike of the option
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the short rate
        Returns
        -------
        float.
        """
        # schedule = (t0,t1,...,tN)

        vec_dt = np.diff(schedule)

        solution = optimize.root(root_function, 0.05, args=(self.price_zc, strike, schedule, vec_dt, kappa, sigma))
        optimal_x = solution.x
        vec_k = self.price_zc(optimal_x, schedule[0], schedule[1:], kappa, sigma)
        return strike * np.sum(
            vec_dt * self.price_zc_put(x, t, schedule[0] - t, schedule[1:] - schedule[0], vec_k, kappa, sigma)
        ) + \
               self.price_zc_put(x, t, schedule[0] - t, schedule[-1] - schedule[0], vec_k[-1], kappa, sigma)

    def simulate(self, grid, dN):
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
        dt = grid.dt
        simulations = np.ndarray(grid.shape)
        simulations[:, 0] = self.init
        for j in range(1, self.simulations.shape[1]):
            simulations[:, j] = simulations[:, j - 1] * exp(-dt * self.kappa) + \
                                self.mean(dt) + \
                                np.sqrt(self.variance(dt)) * dN[:, j]
        return simulations + self.func_instant_forwards(grid.time_grid)

    def fit(self, ircurve):
        # calibrates kappa sigma to the data
        self.func_instant_forwards = ircurve.func_instant_forwards
        self.func_spot_zc = ircurve.func_zc_prices
