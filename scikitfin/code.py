import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Literal, Union, List, Any
from numpy.typing import ArrayLike
from math import *

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
            self.zc_prices : np.ndarray   = np.array(data)  # dimension n
        elif method == 'Swap':
            self.zc_prices  = self.swap_rates_to_zc_prices(maturities, data)
        elif method == 'ActuarialRate':
            self.zc_prices = self.actuarial_rates_to_zc_prices(maturities, data)
        elif method == 'Yield':
            self.zc_prices  = self.yields_to_zc_prices(maturities, data)
        self.instant_forwards : np.ndarray = self.zc_prices_to_inst_forwards(self.maturities, self.zc_prices) #dimension n-1
       
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
        if maturities[0]!=0 :
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
        if maturities[0] != 0:
            maturities = np.concatenate(([0],maturities))
        dt = np.diff(maturities)
        return (1-zc_prices)/(dt*zc_prices).cumsum()
   
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
        zc = [1/((1+swap_rates[0])** maturities[0])]
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
        pass
    
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
        return 'maturities -> {} \nzc_prices -> {} \ninstant_forward -> {}'.format(self.maturities, self.zc_prices, self.instant_forwards)

    def plot(self,
             method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield'] = 'Yield'
            ) -> Any:
        """
        Plots the curve

        Parameters
        ----------
        method : Literal['ZeroCoupon', 'Swap','ActuarialRate', 'Yield']
            method of description of the curve.

        Returns
        -------
        Any
            plot of the curve.

        """
        
        if method == 'Swap':
            values = self.zc_prices_to_swap_rates(self.maturities, self.zc_prices)
        elif method == 'ActuraialRate':
            values = self.zc_prices_to_acturial_rates(self.maturities, self.zc_prices)
        elif method == 'Yield':
            values = self.zc_prices_to_yields(self.maturities, self.zc_prices)
        else:
            values=self.zc_prices
        return plt.plot(self.maturities, values)



class swaptions_cube(object):
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

class IRGauss1FConst(object):
    
    def __init__(self, kappa, sigma, func_instant_forwards=lambda x:0):
        """
        

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
        self.kappa  = kappa
        self.sigma  = sigma
        self.func_instant_forwards = self.func_instant_fowards

    @np.vectorize
    def G(self, tau):
        if tau<1e-6:
            return tau
        else:
            return (1-np.exp(-self.kappa*tau))/self.kappa
    
    def variance(self, t):
        return 0.5*self.sigma**2*self.G(2*t)
    
    def mean(self, t):
        return 0.5*self.sigma**2*self.G(t)**2
    
    def zc_price(self, t, T, x):
        return np.exp(-x*self.G(T-t)-0.5*self.variance(t)*self.G(t,T)**2)

    
    def caplet_price(self):
        pass
    
    def cap_price(self):
        pass
    
    def swaption_price(self):
        pass
    
    def fit(self, ircurve, swaptiondata):
        #calibrates kappa sigma to the data
        pass
    
    def simulate(self, grid, dN):
        """
        

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
        dt               = grid.dt
        simulations      = np.ndarray(grid.shape)
        simulations[:,0] = self.init
        for j in range(1, self.simulations.shape[1]):
            simulations[:,j] = simulations[:,j-1]*exp(-dt*self.kappa) + \
                               self.mean(dt) + \
                               np.sqrt(self.variance(dt))*dN[:,j]
        return simulations+self.func_instant_forwards(grid.time_grid)
    

###############################################################################
#
# CODE
#
###############################################################################

myc= IRCurve([0.5, 1,2,3,4,5,6], [0.210, 0.03,0.0288,0.0292,0.0295,0.0298,0.03],'Swap')
print(myc)
print(myc.func_instant_forwards(np.linspace(0,10,20+1)))
myc.plot()
