import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Literal, Union, List, Any, Tuple, Callable
from numpy.typing import ArrayLike, NDArray
from math import *

###############################################################################
#
# HELPER FUNCTIONS
#
###############################################################################

def inst_forwards_to_zc_prices(maturities: np.ndarray,
                               instant_forwards: np.ndarray
                               ) -> np.ndarray:
    """
    Computes zero-coupon prices from the instantaneous forwards
    Parameters
    ----------
    maturities : np.ndarray
        maturities of the curve.
    instant_forwards : np.ndarray
        instantaneous forwards.
    Returns
    -------
    np.ndarray
        zero-coupon pricies.
    """
    if maturities[0] != 0:
        maturities = np.concatenate(([0], maturities))
    return np.exp(-np.cumsum(instant_forwards * np.diff(maturities)))

def zc_prices_to_inst_forwards(maturities: np.ndarray,
                               zc_prices: np.ndarray
                               ) -> np.ndarray:
    """
    Computes the instantaneous forwards from the zero-coupon prices

    Parameters
    ----------
    maturities : np.ndarray
        maturities of the curve.
    zc_prices : np.ndarray
        zero-coupon prices.
    Returns
    -------
    np.ndarray
        instantaneous forwards.
    """
    if maturities[0] != 0:
        maturities = np.concatenate(([0], maturities))
        zc_prices = np.concatenate(([1], zc_prices))
    return np.diff(-np.log(zc_prices)) / np.diff(maturities)

def zc_prices_to_yields(maturities: np.ndarray,
                        zc_prices: np.ndarray
                        ) -> np.ndarray:
    """
    Computes the yield curve from the zero-coupon price curve
    $zerocoupon = e^{-yield * maturity}$
    Parameters
    ----------
    maturities : np.ndarray
        maturities of the curve.
    zc_prices : np.ndarray
        zero-coupon prices.
    Yields
    ------
    np.ndarray
        yield curve.
    """
    return -np.log(zc_prices) / maturities

def yields_to_zc_prices(maturities: np.ndarray,
                        yields: np.ndarray
                        ) -> np.ndarray:
    """
    Computes the zero-coupon prices from the yield curve
    Parameters
    ----------
    maturities : np.ndarray
        maturities of the curve.
    yields : np.ndarray
        yields of the curve.
    Yields
    ------
    np.ndarray
        yield curve.
    """
    return np.exp(-maturities * yields)


def zc_prices_to_swap_rates(maturities: np.ndarray,
                            zc_prices: np.ndarray
                            ) -> np.ndarray:
    """
    Computes the zero-coupon curve from the swap rate curve
    Parameters
    ----------
    maturities : np.ndarray
        maturities of the curve.
    zc_prices : np.ndarray
        swap rates of the curve.
    Returns
    -------
    np.ndarray
        swap rate curve.
    """
    if maturities[0] != 0:
        maturities = np.concatenate(([0], maturities))
    dt = np.diff(maturities)
    return (1 - zc_prices) / (dt * zc_prices).cumsum()

def swap_rates_to_zc_prices(maturities: np.ndarray,
                            swap_rates: np.ndarray
                            ) -> np.ndarray:
    """
    Computes the zero-coupon curve from the swap rate curve
    Parameters
    ----------
    maturities : np.ndarray
        maturities of the curve.
    swap_rates : np.ndarray
        swap rates of the curve.
    Returns
    -------
    np.ndarray
        zero-coupon curve.
    """
    zc = np.ones(swap_rates.shape)
    zc[0] = 1 / ((1 + swap_rates[0]) ** maturities[0])
    dt = np.diff(maturities)
    for idx, rate in enumerate(swap_rates[1:]):
        zc[idx+1] = (1 - np.sum(zc * dt[:idx + 1] * rate)) / (1 + dt[idx] * rate)
    return zc

def actuarial_rates_to_zc_prices(maturities: np.ndarray,
                                 actuarial_rates: np.ndarray
                                 ) -> np.ndarray:
    """
    Computes the zero-coupon curve from the actuarial rate curve

    Parameters
    ----------
    maturities : np.ndarray
        maturities of the curve.
    actuarial_rates : np.ndarray
        actuarial rates.
    Returns
    -------
    np.ndarray
        zero-coupon prices of the curve.
    """
    return np.power(1 + actuarial_rates, -maturities)


def zc_prices_to_actuarial_rates(maturities: np.ndarray,
                                 zc_prices: np.ndarray
                                 ) -> np.ndarray:
    """
    Computes the zero-coupon curve from the actuarial rate curve
    Parameters
    ----------
    maturities : np.ndarray
        maturities of the curve.
    zc_prices : np.ndarray
            zc_prices.
    Returns
    -------
    np.ndarray
        actuarial rates.
    """
    return np.power(zc_prices, -1 / maturities) - 1

###############################################################################
#
# INTEREST RATES CLASS
#
###############################################################################


class InterestRateCurve(object):

    def __init__(self,
                 maturities: np.ndarray = np.arange(51),
                 data: np.ndarray = np.ones(51),
                 method: Literal['ZeroCoupon', 'SwapRate', 'ActuarialRate', 'Yield'] = 'ZeroCoupon'
                 ) -> None:
        """
        Constructor of InterestRateCurve

        Parameters
        ----------
        maturities : np.ndarray
            maturities of the curve.
        data : np.ndarray
            values of the curve.
        method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield']
            method of description of the curve.
        """
        self.maturities: np.ndarray = maturities
        if method == 'ZeroCoupon':
            self.zc_prices: np.ndarray = data
        elif method == 'SwapRate':
            self.zc_prices: np.ndarray = swap_rates_to_zc_prices(maturities, data)
        elif method == 'ActuarialRate':
            self.zc_prices: np.ndarray = actuarial_rates_to_zc_prices(maturities, data)
        elif method == 'Yield':
            self.zc_prices: np.ndarray = yields_to_zc_prices(maturities, data)
        self.instant_forwards: np.ndarray = zc_prices_to_inst_forwards(self.maturities, self.zc_prices)
        # vectorized methods
        self.annuity=np.vectorize(self._annuity, excluded=['self'])

    def __str__(self) -> str:
        """
        Returns the description of the curve
        Returns
        -------
        str
            description of the curve.
        """
        return 'maturities -> {} \nzc_prices -> {} \ninstant_forward -> {}'.format(self.maturities, self.zc_prices,
                                                                                   self.instant_forwards)

    ########################## PROPERTY  ########################################################################

    @property
    def function_instant_forward(self):
        if self.maturities[0] != 0:
            maturities = np.concatenate(([0], self.maturities))
        else:
            maturities = self.maturities
        dt = np.diff(maturities)
        return interpolate.interp1d(maturities[:-1], self.instant_forwards, kind='nearest', bounds_error=False,
                                    fill_value=(self.instant_forwards[0], self.instant_forwards[-1]))

    @property
    def function_integral_fowards(self):
        if self.maturities[0] != 0:
            maturities = np.concatenate(([0], self.maturities))
        else:
            maturities = self.maturities
        dt = np.diff(maturities)
        return interpolate.interp1d(maturities, np.concatenate(([0], np.cumsum(self.instant_forwards * dt))),
                                    kind='linear', fill_value='extrapolate')

    @property
    def function_zc_price(self):
        return lambda t: np.exp(-self.function_integral_fowards(t))

    ########################## METHODS  ########################################################################
    def instant_forward(self, maturities):
        return self.function_instant_forward(maturities)
    
    def yields(self, maturities):
        return zc_prices_to_yields(maturities, self.zc_price(maturities))
    
    def actuarial_rate(self, maturities):
        return zc_prices_to_actuarial_rates(maturities, self.zc_prices(maturities))

    def zc_price(self,
                 maturities: ArrayLike
                ) -> ArrayLike:
        return self.function_zc_price(maturities)

    def _annuity(self,
                 expiry: float,
                 tenor: float,
                 dt: float = 0.5
                ) -> float:
        maturities = np.arange(expiry + dt, expiry + tenor + dt, dt)
        return np.sum(dt * self.zc_price(maturities))
    
    def swap_rate(self,
                  tenor: ArrayLike,
                  dt: float = 0.5
                 ) -> ArrayLike:
        return (1 - self.zc_price(tenor)) / self.annuity(0, tenor, dt)
    
    def forward_swap_rate(self,
                          expiry: ArrayLike,
                          tenor: ArrayLike,
                          dt: float = 0.5
                         ) -> float:
        return (self.zc_price(expiry) - self.zc_price(tenor + expiry)) / self.annuity(expiry, tenor, dt)
            
    def plot(self,
             method: Literal['ZeroCoupon', 'Swap', 'ActuarialRate', 'Yield'] = 'Yield'
             ) -> None:
        """
        Plots the curve
        Parameters
        ----------
        method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield']
            method of description of the curve.
        Returns
        -------
        None
            plot of the curve.
        """

        if method == 'Swap':
            values = zc_prices_to_swap_rates(self.maturities, self.zc_prices)
        elif method == 'ActuarialRate':
            values = zc_prices_to_actuarial_rates(self.maturities, self.zc_prices)
        elif method == 'Yield':
            values = zc_prices_to_yields(self.maturities, self.zc_prices)
        else:
            values = self.zc_prices
        plt.plot(self.maturities, values)
        plt.show()
