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
    
    def __init__(self, kappa : float, sigma: float, func_instant_forwards : Callable[[ArrayLike], np.ndarray],
                   func_spot_zc : Callable[[ArrayLike], np.ndarray] 
                ) -> None:
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
        self.kappa  = kappa
        self.sigma  = sigma
        self.func_instant_forwards = func_instant_forwards
        self.func_spot_zc = func_spot_zc

    @np.vectorize
    @staticmethod
    def G(tau : ArrayLike, kappa : float) -> np.ndarray:
        """
        function used to compute ZC prices

        Parameters
        ----------
        tau : ArrayLike
            time steps.
        kappa : float
            parameters of the model.

        Returns
        -------
        np.ndarray.

        """
        tau = np.array(tau)
        return 1e-6 if tau < 1e-6 else (1-np.exp(-kappa*tau))/kappa
     
    
    @staticmethod
    def variance(t : float, kappa : float, sigma : float) -> float:
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
        return 0.5 * sigma**2 * IRGauss1FConst.G(2*t, kappa)
    
    @np.vectorize
    @staticmethod
    def zc_bond_volatility(t : float ,T : ArrayLike , kappa : float, sigma : float) -> np.ndarray:
        """
        Volotility of ZC  

        Parameters
        ----------
        t : float
            current time.
        T : ArrayLike 
            expiry of the ZC bond 
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the short rate
        Returns
        -------
        np.ndarray.
        """
        T = np.array(T)
        return 0.5 * sigma**2 * IRGauss1FConst.G(T, kappa)**2 * IRGauss1FConst.G(2*(T-t), kappa)

    @np.vectorize
    @staticmethod
    def mean(t : float ,x: float ,kappa: float ,sigma : float) -> np.ndarray:
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
        nd.ndaray.
        """
        return  x * np.exp(-kappa*t) + 0.5 * sigma**2 * IRGauss1FConst.G(t, kappa)**2 
    
    def price_zc(self,x : float, t : float, T: ArrayLike , kappa: float, sigma: float) -> np.ndarray:
        """
        ZC price

        Parameters
        ----------
        x : float
            stochastic process x ( x = r - f )
        t : float
            current time.
        T : ArrayLike 
            expiry of the ZC bond 
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the ZC
        Returns
        -------
        np.ndarray.
        """
        T = np.array(T)
        return (self.func_spot_zc(T)/self.func_spot_zc(t)) * \
               np.exp(-x * IRGauss1FConst.G(T-t, kappa) - 0.5 * IRGauss1FConst.variance(t, kappa, sigma) \
                * IRGauss1FConst.G(T-t, kappa)**2)

    
    def fit(self, ircurve, swaptiondata):
        #calibrates kappa sigma to the data
        pass
    
    def black_tools(self, x:float, t:float, maturity: ArrayLike, 
                    tenor: ArrayLike, K: ArrayLike, kappa: float, sigma : float) -> Tuple :
        """
        black tools

        Parameters
        ----------
        x : float
            stochastic process x ( x = r - f )
        t : float
            current time.
        maturity: ArrayLike
            expiry of the option
        tenor: ArrayLike
            expiry of the zc underlying
        K : ArrayLike
            strike of the option
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the short rate
        Returns
        -------
        np.ndarray.
        """
        maturity = np.array(maturity)
        v         = IRGauss1FConst.zc_bond_volatility(t,maturity,kappa,sigma)
        P_up      = self.price_zc(x, t, maturity + tenor, kappa, sigma)
        P_down    = self.price_zc(x, t, maturity, kappa, sigma)
        d_positif = (np.log(P_up/(K * P_down)) + v/2 ) / np.sqrt(v)
        d_negatif = (np.log(P_up/(K * P_down)) - v/2 ) / np.sqrt(v)

        return P_up,P_down,d_positif,d_negatif

    def price_zc_call(self, x:float, t:float, maturity: ArrayLike, tenor: ArrayLike,
                       K: ArrayLike, kappa : float, sigma : float) -> np.ndarray:
        """
        price of call option on ZC
        P185
        Parameters
        ----------
        x : float
            stochastic process x ( x = r - f )
        t : float
            current time.
        maturity: ArrayLike 
            expiry of the option
        tenor: ArrayLike
            expiry of the zc underlying
        K : ArrayLike
            strike of the option
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the short rate
        Returns
        -------
        np.ndaaray.
        """
        P_up, P_down, d_positif, d_negatif = self.black_tools(x,t,maturity,tenor,K,kappa,sigma)

        return P_up * norm.cdf(d_positif) - P_down * K * norm.cdf(d_negatif)
    
    
    def price_zc_put(self, x:float, t:float, maturity: ArrayLike, 
                      tenor: ArrayLike, K: ArrayLike, kappa: float, sigma: float) -> np.ndarray:
        """
        price of put option on ZC
        p185
        Parameters
        ----------
        x : float
            stochastic process x ( x = r - f )
        t : float
            current time.@
        maturity: ArrayLike 
            expiry of the option
        tenor: ArrayLike
            expiry of the zc underlying
        K : ArrayLike
            strike of the option
        kappa : float
            parameters of the model.
        sigma : float
            volatility of the short rate
        Returns
        -------
        np.ndarray.
        """
        
        P_up, P_down, d_positif, d_negatif = self.black_tools(x,t,maturity,tenor,K,kappa,sigma)

        return  P_down * K * norm.cdf(-d_negatif) -  P_up * norm.cdf(-d_positif) 

    def price_swaption(self,x : float, t : float, schedule : ArrayLike,
                      strike : float, kappa : float, sigma: float) -> float:

        """
        price of swaption

        Parameters
        ----------
        x : float
            stochastic process x ( x = r - f )
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
        #p421
        #schedule = (t0,t1,...,tN)

        steps_time   = np.diff(schedule)
        payment_date = np.array(schedule)[1:]
        Tn     = payment_date[-1]

        def root_function(x):
            output = -1 + self.price_zc(x,t,Tn,kappa,sigma) + np.sum(strike * steps_time * self.price_zc(x,t,payment_date,kappa,sigma))    
            return output

        solution = optimize.root(root_function, 1)
        optimal_x = solution.x
        K  = self.price_zc(optimal_x,t, payment_date,kappa,sigma)
        Vswaption = np.sum(steps_time * K *  self.price_zc_put(x,t,t,payment_date -t,K,kappa,sigma)) + self.price_zc_put(x,t,t,Tn-t,K[-1],kappa,sigma)

        return Vswaption
    
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
        dt               = grid.dt
        simulations      = np.ndarray(grid.shape)
        simulations[:,0] = self.init
        for j in range(1, self.simulations.shape[1]):
            simulations[:,j] = simulations[:,j-1]*exp(-dt*self.kappa) +\
                               self.mean(dt) +\
                               np.sqrt(self.variance(dt))*dN[:,j]
        return simulations+self.func_instant_forwards(grid.time_grid)
    

###############################################################################
#
# CODE
#
###############################################################################


myc= IRCurve([0.5, 1,2,3,4,5,6], [0.210, 0.03,0.0288,0.0292,0.0295,0.0298,0.03],'Swap')
kappa = 0.5
sigma = 0.2 
model = IRGauss1FConst(kappa,sigma,myc.func_instant_forwards, myc.func_spot_zc)
print(model.func_instant_forwards(10))


print(IRGauss1FConst.mean(0,0.2,kappa,sigma))
zc_prices = model.price_zc(0.2, 0, [1,2,3], kappa, sigma)
print(zc_prices)

swaption = model.price_swaption(0.2, 0, (1,1.5,2,2.5,3), 0.02 ,kappa, sigma) 
print("swaption", swaption)
print(myc.func_instant_forwards(np.linspace(0,10,20+1)))
myc.plot()