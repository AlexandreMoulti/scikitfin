# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:57:03 2022
@author: Alexandre
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Literal, Union, List, Any
###############################################################################
#
# DATA
#
###############################################################################


class IRCurve(object):
    
    def __init__(self,
                maturities : Union[np.ndarray, List[float]], 
                data : Union[np.ndarray, List[float]], 
                method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield']) -> None:
        """
        Constructor function
        Args:
            maturities (Union[np.ndarray, List[float]]): list of maturities
            data (Union[np.ndarray, List[float]]): list of value corresponding to a specific rate curve
            method (Literal[&#39;ZeroCoupon&#39;, &#39;Swap&#39;,&#39;ActuarialRate&#39;, &#39;Yield&#39;]): type of the curve
        """
        self.maturities : np.ndarray      = np.array(maturities) #dimension n
        if method=='ZeroCoupon':
            self.zc_prices : np.ndarray       = np.array(data)  # dimension n
        elif method=='Swap':
            self.zc_prices  = self.swap_rates_to_zero_coupon(maturities, data)
        elif method=='ActuarialRate':
            self.zc_prices = self.actuarial_rate_to_zero_coupon(maturities, data)
        elif method=='Yield':
            self.zc_prices  = self.yield_to_zero_coupon(maturities, data)
        self.instant_forwards : np.ndarray = self.compute_instant_forwards(self.maturities, self.zc_prices) #dimension n-1

    def interpolate_instant_forwards(self, 
                                    new_maturities: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
            function that permit to interpolate instantaneous forward curve 
        Args:
            new_maturities (Union[np.ndarray, List[float]]): List of new maturities to be interpolated 

        Returns:
            np.ndarray: new instantaneous forward curve with new maturities
        """
        interpol = interpolate.interp1d( self.maturities[:-1], self.instant_forwards, kind='nearest', 
                                         bounds_error=False, fill_value=(self.instant_forwards[0], self.instant_forwards[-1]) )
        return interpol(new_maturities)

    @staticmethod
    def compute_zc_prices(maturities : Union[np.ndarray, List[float]], 
                          instant_forwards : Union[np.ndarray, List[float]]) -> np.ndarray:
        """
            function that compute zero coupon prices from instantaneous forward curve 
        Args:
            maturities (Union[np.ndarray, List[float]]): List of new maturities to be interpolated 
            instant_forwards (Union[np.ndarray, List[float]]): instantaneous forward curve

        Returns:
            np.ndarray: zc prices 
        """
        return np.exp(-np.cumsum(instant_forwards*np.diff(maturities)))
    
    @staticmethod
    def compute_instant_forwards(maturities: Union[np.ndarray, List[float]], 
                                  zc_prices : Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        function that compute instantaneous forward curve from zc prices
        Args:
            maturities (Union[np.ndarray, List[float]]): list of maturities
            zc_prices (Union[np.ndarray, List[float]]): zero coupon prices

        Returns:
            np.ndarray: instantneous forward curve
        """
        return np.diff(-np.log(zc_prices))/np.diff(maturities)
    
    @staticmethod
    def zero_coupon_to_yield(maturities : Union[np.ndarray, List[float]],
                               zc_prices: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        function that compute yield  curve from zero coupon prices 

        Args:
            maturities (Union[np.ndarray, List[float]]): list of maturities
            zc_prices (Union[np.ndarray, List[float]]): zero coupon prices

        Returns:
            np.ndarray: yield curve 
        """
        return -np.log(zc_prices)/maturities
    
    @staticmethod
    def yield_to_zero_coupon(maturities: Union[np.ndarray, List[float]],
                             yields : Union[np.ndarray, List[float]]) -> np.ndarray:
        """
            function that compute zero coupon prices from yield curve 
        Args:
            maturities (Union[np.ndarray, List[float]]): list of maturities
            yields (Union[np.ndarray, List[float]]): yield curve 

        Returns:
            np.ndarray: zero coupon curve 
        """
        return np.exp(-np.array(maturities) * yields)
    
    @staticmethod
    def swap_rates_to_zero_coupon(maturities : Union[np.ndarray, List[float]], 
                                  swap_rates : Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        function that compute zero coupon prices from swap rate curve 
        Args:
            maturities (Union[np.ndarray, List[float]]): list of maturities
            swap_rates (Union[np.ndarray, List[float]]):  swap rate curve 

        Returns:
            np.ndarray:    zero coupon prices
        """
        zc = [1/((1+swap_rates[0])** maturities[0])]
        dt = np.diff(maturities) 
        for idx, rate in enumerate(swap_rates[1:]):
            zc_temp = (1 - np.sum(zc* dt[:idx+1] * rate)) / ( 1 + dt[idx]*rate)
            zc.append(zc_temp)
        return np.array(zc)
    

    @staticmethod
    def actuarial_rate_to_zero_coupon(maturities : Union[np.ndarray, List[float]], 
                                      actuarial_rates : Union[np.ndarray, List[float]]) -> np.ndarray:
        """
            function that compute zero coupon prices from actuarials rate curve 
        Args:
            maturities (Union[np.ndarray, List[float]]): list of maturities
            actuarial_rates (Union[np.ndarray, List[float]]): actuarial rates curve

        Returns:
            np.ndarray: zero coupon prices
        """
        return np.power(1/(1+np.array(actuarial_rates)), maturities)
    

    @classmethod
    def data_to_curve(cls, 
                      data_file_path : str,
                      method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield'] ) -> Any:
        """
        Create IRCurve object from csv file (that contains Maturities and a specific rate)
        Args:
            data_file_path (str): name of data file 
            method (Literal[&#39;ZeroCoupon&#39;, &#39;Swap&#39;,&#39;ActuarialRate&#39;, &#39;Yield&#39;]): type of rate

        Returns:
            Any: IRCurve Object
        """
        
        pd_curve   = pd.read_csv(data_file_path)
        maturities = pd_curve.iloc[:,0].to_numpy()
        data       = pd_curve.iloc[:,1].to_numpy()
        return IRCurve(maturities,data,method)
    
    def __str__(self):
        return 'maturities -> {} \nzc_prices -> {} \ninstant_forward -> {}'.format(self.maturities, self.zc_prices, self.instant_forwards)

    def plot(self):
        plt.plot(self.maturities, self.zc_prices)
        plt.show()

###############################################################################
#
# GRID
#
###############################################################################

class DiscretisationGrid(object):
    """ Discretisation to be used for the Monte-Carlo simulations
        Time grid, discretisation of the time 
        horizon: Horizon of the simulation
        nb_simulations represents the number of Monte-Carlo paths to simulate
    """
    
    def __init__(self, nb_simulations, horizon, nb_steps):
        """ instantiation of the object
        """
        self.nb_steps       = nb_steps #per year
        self.dt             = 1.0/self.nb_steps
        self.horizon        = horizon
        self.nb_simulations = nb_simulations
        self.time_grid      = np.linspace(0,self.horizon, nb_steps*self.horizon+1, endpoint=True)
        self.shape          = (self.nb_simulations, self.time_grid.shape[0])

###############################################################################
#
# MODELS
#
###############################################################################



###############################################################################
#
# CODE
#
###############################################################################
#path = '/Users/angeromualdossohou/Documents/swap.csv'

#curve = IRCurve.data_to_curve(path, 'Swap')
#print(curve.zc_prices)
# maturities = np.array([0,0.5,1,2,3])
# zc_prices  = np.array([1,0.99,0.9,0.8,0.7])
# int_forwards = -np.log(zc_prices)
# instant_forwards = (int_forwards[1:]-int_forwards[:-1])/ (maturities[1:]-maturities[:-1])
# np.exp(-np.cumsum(instant_forwards*np.diff(maturities)))
       


# interpol = interpolate.interp1d(maturities[:-1], instant_forwards, kind='nearest', bounds_error=False, fill_value=(instant_forwards[0], instant_forwards[-1]) )
# new_mat=np.linspace(0,10,20+1)
# interpol(new_mat)

myc= IRCurve([1,2,3,4,5,6], [0.02616,0.03056,0.03121,0.03128,0.03133,0.03129],'Swap')
print(myc)
print(myc.interpolate_instant_forwards(np.linspace(0,10,20+1)))
myc.plot()

#myc.get_new_instant_forwards(np.linspace(0,10,20+1))

