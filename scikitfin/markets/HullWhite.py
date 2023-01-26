

######## static methods #################

@np.vectorize
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
      

######## Hull White ###############

class HullWhite(object):
    
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
        return (self.func_spot_zc(T)/self.func_spot_zc(t)) * \
               np.exp(-x * IRGauss1FConst.G(T-t, kappa) - 0.5 * IRGauss1FConst.variance(t, kappa, sigma) \
                * IRGauss1FConst.G(T-t, kappa)**2)

    
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
      
   def fit(self, swaptiondata):
        #calibrates kappa sigma to the data
        pass
    
