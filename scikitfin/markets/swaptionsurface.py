import numpy as np
import matplotlib.pyplot as plt

# static methods
def prices_to_normal_vols(ircurve, prices):
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
    for (expiry, tenor, price) in prices:
        maturities = np.arange(expiry + dt, expiry + tenor + dt, dt)
        annuity_factor = np.sum(dt * ircurve.func_zc_prices(maturities))
        vol = 100 * price * np.sqrt(2 * np.pi / expiry) / annuity_factor
        normal_vols.append((expiry, tenor, vol))

    return np.array(normal_vols, dtype=[("maturity", "i4"), ("tenor", "i4"), ("value", "f8")])


def normal_vols_to_prices(ircurve, normal_vols):
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
    for (expiry, tenor, volatility) in normal_vols:
        maturities = np.arange(expiry + dt, expiry + tenor + dt, dt)
        annuity_factor = np.sum(dt * ircurve.func_zc_prices(maturities))
        price = volatility * annuity_factor * np.sqrt(expiry / 2 * np.pi) / 100
        normal_prices.append((expiry, tenor, price))

    return np.array(normal_prices, dtype=[("maturity", "i4"), ("tenor", "i4"), ("value", "f8")])


class SwaptionSurface(object):

    def __init__(self, ircurve, array_of_tuple, mode):
        """
        np.array (maturity, tenor, value)
        """
        if mode == 'Price':
            self.prices = array_of_tuple
            self.normal_vols = prices_to_normal_vols(ircurve, self.prices)
        else:
            self.normal_vols = array_of_tuple
            self.prices = normal_vols_to_prices(ircurve, self.normal_vols)


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