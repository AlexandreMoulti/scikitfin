import numpy as np
import matplotlib.pyplot as plt


def prices_to_normal_vols(ircurve, expiry, tenor, price):
    """
    compute the ATM swaption vols from prices

    All the swap pay coupon each semester ( by convention)
    Parameters
    ----------
    ircurve
    prices
    Returns
    -------

    """
    normal_vols = []
    dt = 0.5
    maturities = np.arange(expiry + dt, expiry + tenor + dt, dt)
    annuity_factor = np.sum(dt * ircurve.func_zc_prices(maturities))
    return price * np.sqrt(2 * np.pi / expiry) / annuity_factor

prices_to_normal_vols = np.vectorize(prices_to_normal_vols, excluded=['ircurve'])

def normal_vols_to_prices(ircurve, expiry, tenor, volatility):
    """
    compute the ATM swaption prices from normal volatility
       All the swap pay coupon each semester ( by convention)
    Parameters
    ----------
    ircurve
    normal_vols

    Returns
    -------
    """
    normal_prices = []
    dt = 0.5
    maturities = np.arange(expiry + dt, expiry + tenor + dt, dt)
    annuity_factor = np.sum(dt * ircurve.func_zc_prices(maturities))
    return volatility * annuity_factor * np.sqrt(expiry / 2 * np.pi)

normal_vols_to_prices = np.vectorize(normal_vols_to_prices, excluded=['ircurve'])

class SwaptionSurface(object):
    def __init__(self, ircurve, expiries, tenors, values, mode):
        """
        np.array (maturity, tenor, value)
        """
        self.expiries = np.asarray(expiries)
        self.tenors = np.asarray(tenors)
        if mode == 'price':
            self.prices = np.asarray(values)
            self.normal_vols = prices_to_normal_vols(ircurve, self.expiries, self.tenors, self.prices)
        else:
            self.normal_vols = np.asarray(values)
            self.prices = normal_vols_to_prices(ircurve, self.expiries, self.tenors, self.normal_vols)


    def plot_surface(self, mode = 'vols'):
        """
        plot  surfaces depending on mode
        Parameters
        ----------
        mode

        Returns
        -------

        """
        pass