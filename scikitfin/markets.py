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
# DATA
#
###############################################################################


class IRCurve(object):
    
    def __init__(self,
                 maturities : ArrayLike, 
                 data : ArrayLike, 
                 method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield'] = 'ZeroCoupon'
                ) -> None:
        """
        Constructor

        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        data : ArrayLike
            values of the curve.
        method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield']
            method of description of the curve.
        """
        self.maturities : np.ndarray      = np.array(maturities) #dimension n
        if method == 'ZeroCoupon':
            self.price_zcs : np.ndarray   = np.array(data)  # dimension n
        elif method == 'Swap':
            self.price_zcs  = self.swap_rates_to_zc_prices(maturities, data)
        elif method == 'ActuarialRate':
            self.price_zcs = self.actuarial_rates_to_zc_prices(maturities, data)
        elif method == 'Yield':
            self.price_zcs  = self.yields_to_zc_prices(maturities, data)
        self.instant_forwards : np.ndarray = self.zc_prices_to_inst_forwards(self.maturities, self.price_zcs) #dimension n-1
       
    @property
    def func_instant_forwards(self):
        if self.maturities[0]!=0:
            maturities = np.concatenate(([0], self.maturities))
        else:
            maturities = self.maturities
        dt = np.diff(maturities) 
        return interpolate.interp1d( maturities[:-1], self.instant_forwards, kind='nearest', 
                                                      bounds_error=False, fill_value=(self.instant_forwards[0], self.instant_forwards[-1]) )
    @property
    def func_integ_forwards(self):
        if self.maturities[0]!=0:
            maturities = np.concatenate(([0], self.maturities))
        else:
            maturities = self.maturities
        dt = np.diff(maturities) 
        return interpolate.interp1d( maturities, np.concatenate(([0], np.cumsum(self.instant_forwards*dt))), kind='linear',
                                                    fill_value='extrapolate')
    @property
    def func_spot_zc(self):
        if self.maturities[0]!=0:
            maturities = np.concatenate(([0], self.maturities))
        else:
            maturities = self.maturities
        dt = np.diff(maturities)
        return lambda t:np.exp(-self.func_integ_forwards(t))

    @staticmethod
    def inst_forwards_to_zc_prices(maturities : ArrayLike, 
                                   instant_forwards : ArrayLike
                                  ) -> np.ndarray:
        """
        Computes zero-coupon prices from the instantaneous forwards

        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        instant_forwards : ArrayLike
            instantaneous forwards.

        Returns
        -------
        np.ndarray
            zero-coupon pricies.

        """
        return np.exp(-np.cumsum(instant_forwards*np.diff(maturities)))
    
    @staticmethod
    def zc_prices_to_inst_forwards(maturities: ArrayLike, 
                                   zc_prices : ArrayLike
                                  ) -> np.ndarray:
        """
        Computes the instantaneous forwards from the zero-coupon prices
        
        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        zc_prices : ArrayLike
            zero-coupon prices.

        Returns
        -------
        np.ndarray
            instantaneous forwards.

        """
        if np.array(maturities)[0]!=0 :
            maturities = np.concatenate(([0], maturities))
            zc_prices  = np.concatenate(([1], zc_prices))
        return np.diff(-np.log(zc_prices))/np.diff(maturities)
    
    @staticmethod
    def zc_prices_to_yields(maturities : ArrayLike,
                            zc_prices: ArrayLike
                            ) -> np.ndarray:
        """
        Computes the yield curve from the zero-coupon price curve
        $zerocoupon = e^{-yield * maturity}$

        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        zc_prices : ArrayLike
            zero-coupon prices.

        Yields
        ------
        np.ndarray
            yield curve.

        """
        return -np.log(zc_prices)/maturities
    
    @staticmethod
    def yields_to_zc_prices(maturities: ArrayLike,
                            yields : ArrayLike
                           ) -> np.ndarray:
        """
        Computes the zero-coupon prices from the yield curve

        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        yields : ArrayLike
            yields of the curve.

        Yields
        ------
        np.ndarray
            yield curve.

        """
        return np.exp(-np.array(maturities) * yields)
    
    @staticmethod
    def zc_prices_to_swap_rates(maturities : ArrayLike, 
                                 zc_prices : ArrayLike
                                ) -> np.ndarray:
        """
        Computes the zero-coupon curve from the swap rate curve

        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        zc_prices : ArrayLike
            swap rates of the curve.

        Returns
        -------
        np.ndarray
            swap rate curve.

        """
        if np.array(maturities)[0] != 0:
            maturities = np.concatenate(([0],maturities))
        dt = np.diff(maturities)
        return (1-np.array(zc_prices))/(dt*zc_prices).cumsum()
   
    @staticmethod
    def swap_rates_to_zc_prices(maturities : ArrayLike, 
                                swap_rates : ArrayLike
                                )-> np.ndarray:
        """
        Computes the zero-coupon curve from the swap rate curve

        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        swap_rates : ArrayLike
            swap rates of the curve.

        Returns
        -------
        np.ndarray
            zero-coupon curve.

        """
        swap_rates = np.array(swap_rates)
        zc = [1/((1+swap_rates[0])** np.array(maturities)[0])]
        dt = np.diff(maturities)
        for idx, rate in enumerate(swap_rates[1:]):
            zc_temp = (1 - np.sum(zc* dt[:idx+1] * rate)) / ( 1 + dt[idx]*rate)
            zc.append(zc_temp)
        return np.array(zc)
    
    @staticmethod
    def actuarial_rates_to_zc_prices(maturities : ArrayLike, 
                                      actuarial_rates : ArrayLike
                                     ) -> np.ndarray:
        """
        Computes the zero-coupon curve from the actuarial rate curve

        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        actuarial_rates : ArrayLike
            actuarial rates.

        Returns
        -------
        np.ndarray
            zero-coupon prices of the curve.

        """
        return np.power(1/(1+np.array(actuarial_rates)), maturities)

    @staticmethod
    def zc_prices_to_acturial_rates( maturities : ArrayLike, 
                                     zc_prices : ArrayLike
                                   ) -> np.ndarray:
        """
        Computes the zero-coupon curve from the actuarial rate curve

        Parameters
        ----------
        maturities : ArrayLike
            maturities of the curve.
        zc_prices : ArrayLike
             zc_prices.

        Returns
        -------
        np.ndarray
            actuarial rates.

        """
        return np.power(1/np.array(zc_prices), 1/np.array(maturities))
    
    @classmethod
    def data_to_curve(cls, 
                      data_file_path : str,
                      method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield'] 
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
        pd_curve   = pd.read_csv(data_file_path)
        maturities = pd_curve.iloc[:,0].to_numpy()
        data       = pd_curve.iloc[:,1].to_numpy()
        return IRCurve(maturities,data,method)
    
    def __str__(self) -> str:
        """
        Returns the description of the curve

        Returns
        -------
        str
            description of the curve.

        """
        return 'maturities -> {} \nzc_prices -> {} \ninstant_forward -> {}'.format(self.maturities, self.price_zcs, self.instant_forwards)

    def plot(self,
             method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield'] = 'Yield'
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
            values = self.zc_prices_to_swap_rates(self.maturities, self.price_zcs)
        elif method == 'ActuraialRate':
            values = self.zc_prices_to_acturial_rates(self.maturities, self.price_zcs)
        elif method == 'Yield':
            values = self.zc_prices_to_yields(self.maturities, self.price_zcs)
        else:
            values=self.price_zcs
        plt.plot(self.maturities, values)
        plt.show()



class swaptions_cube(object):
    """ representation of a swaption cube
    """
    
    def __init__(self):
        # dictionary tau, mat -> normal vol
        # alternative constructors
        # dictionary tau, mat -> prices
        # forward prices
        pass
