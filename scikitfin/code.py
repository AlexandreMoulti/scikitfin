
import numpy as np
import pandas as pd
from scipy import interpolate

###############################################################################
#
# DATA MODELS
#
###############################################################################


class IRCurve(object):
    
    def __init__(self, maturities, data, method='ZeroCoupon'):
        self.maturities       = np.array([0] + maturities) #dimension n
        if method=='ZeroCoupon':
            self.zc_prices        = np.array([1] + data)  # dimension n
        elif method=='Swap':
            self.zc_prices = self.swap_rate_to_zero_coupon(maturities, data)
        elif method=='ActuarialRate':
            self.zc_prices = self.actuarial_rate_to_zero_coupon(maturities, data)
        elif method=='Yield':
            self.zc_prices = self.yield_to_zero_coupon(maturities, data)
        self.instant_forwards = self.compute_instant_forwards(self.maturities, self.zc_prices) #dimensioin n-1

    def interpolate_instant_forwards(self, new_maturities):
        interpol = interpolate.interp1d( self.maturities[:-1], self.instant_forwards, kind='nearest', 
                                         bounds_error=False, fill_value=(self.instant_forwards[0], self.instant_forwards[-1]) )
        return interpol(new_maturities)

    @staticmethod
    def compute_zc_prices(maturities, instant_fowards):
        return np.exp(-np.cumsum(instant_forwards*np.diff(maturities)))
    
    @staticmethod
    def compute_instant_forwards(maturities, zc_rpices):
        return np.diff(-np.log(zc_prices))/np.diff(maturities)
    
    @staticmethod
    def zero_coupon_to_yield(maturities, zc_prices):
        return -np.log(zc_prices[1:])/maturities[1:]
    
    @staticmethod
    def yield_to_zero_coupon(maturities, yields):
        pass
    
    @staticmethod
    def swap_rates_to_zero_coupon(maturities, swap_rates):
        pass
    
    
    
    @staticmethod
    def actuarial_rate_to_zero_coupon(maturities, actuarial_rates):
        pass
        
    def plot(self):
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



###############################################################################
#
# CODE
#
###############################################################################
pd_curve   = pd.read_csv('zc_curve.csv')
maturities = np.array([0,0.5,1,2,3])
zc_prices  = np.array([1,0.99,0.9,0.8,0.7])
int_forwards = -np.log(zc_prices)
instant_forwards = (int_forwards[1:]-int_forwards[:-1])/ (maturities[1:]-maturities[:-1])
np.exp(-np.cumsum(instant_forwards*np.diff(maturities)))
       


interpol = interpolate.interp1d(maturities[:-1], instant_forwards, kind='nearest', bounds_error=False, fill_value=(instant_forwards[0], instant_forwards[-1]) )
new_mat=np.linspace(0,10,20+1)
interpol(new_mat)

myc= Curve([0.5,1,2,3], [0.99,0.9,0.8,0.7])
myc.get_new_instant_forwards(np.linspace(0,10,20+1))
