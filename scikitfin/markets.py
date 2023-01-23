import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Literal, Union, List, Any, Tuple, Callable
from numpy.typing import ArrayLike
from math import *
from scipy.stats import norm
from scipy import optimize




###############################################################################
#
# STATIC METHOHDS
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
    swap_rates = swap_rates
    zc = [1 / ((1 + swap_rates[0]) ** maturities[0])]
    dt = np.diff(maturities)
    for idx, rate in enumerate(swap_rates[1:]):
        zc_temp = (1 - np.sum(zc * dt[:idx + 1] * rate)) / (1 + dt[idx] * rate)
        zc.append(zc_temp)
    return np.array(zc)

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
# INTEREST RATES
#
###############################################################################


class InterestRateCurve(object):

    def __init__(self,
                 maturities: np.ndarray,
                 data: np.ndarray,
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
    def func_instant_forwards(self):
        if self.maturities[0] != 0:
            maturities = np.concatenate(([0], self.maturities))
        else:
            maturities = self.maturities
        dt = np.diff(maturities)
        return interpolate.interp1d(maturities[:-1], self.instant_forwards, kind='nearest', bounds_error=False,
                                    fill_value=(self.instant_forwards[0], self.instant_forwards[-1]))

    @property
    def func_integ_forwards(self):
        if self.maturities[0] != 0:
            maturities = np.concatenate(([0], self.maturities))
        else:
            maturities = self.maturities
        dt = np.diff(maturities)
        return interpolate.interp1d(maturities, np.concatenate(([0], np.cumsum(self.instant_forwards * dt))),
                                    kind='linear', fill_value='extrapolate')

    @property
    def func_zc_prices(self):
        if self.maturities[0] != 0:
            maturities = np.concatenate(([0], self.maturities))
        else:
            maturities = self.maturities
        dt = np.diff(maturities)
        return lambda t: np.exp(-self.func_integ_forwards(t))

    ########################## METHODS  ########################################################################

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

    ########################## CLASS METHODS  ########################################################################

    @classmethod
    def data_to_curve(cls,
                      data_file_path: str,
                      method: Literal['ZeroCoupon', 'Swap', 'ActuarialRate', 'Yield']
                      ) -> Any:
        """
        Alternative constructor from tabular data
        Parameters
        ----------
        data_file_path : str
            path of the file.
        method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield']
            method of description of the curve.
        Returns
        -------
        Any
            Curve.
        """
        pd_curve = pd.read_csv(data_file_path)
        maturities = pd_curve.iloc[:, 0].to_numpy()
        data = pd_curve.iloc[:, 1].to_numpy()
        return IRCurve(maturities, data, method)



###############################################################################
#
# SWAPTION VOL CUBE
#
###############################################################################


class SwaptionsCube(object):
    """ representation of a swaption cube
    """

    def __init__(self):
        # dictionary tau, mat -> normal vol
        # alternative constructors
        # dictionary tau, mat -> prices
        # forward prices
        pass

###############################################################################
#
# EQUITY VOL SURFACE
#
###############################################################################
