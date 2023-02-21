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

    normal_vols = dict()
    dt = 0.5
    for (expiry, tenor) in prices:
        price = prices[(expiry, tenor)]
        maturities = np.arange(expiry + dt, expiry + tenor + dt, dt)
        annuity_factor = np.sum(dt * ircurve.func_zc_prices(maturities))
        normal_vols[(expiry, tenor)] = 100 * price * np.sqrt(2 * np.pi / expiry) / annuity_factor

    return normal_vols


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
    normal_prices = dict()
    dt = 0.5
    for (expiry, tenor) in normal_vols:
        volatility = normal_vols[(expiry, tenor)]
        maturities = np.arange(expiry + dt, expiry + tenor + dt, dt)
        annuity_factor = np.sum(dt * ircurve.func_zc_prices(maturities))
        normal_prices[(expiry, tenor)] = volatility * annuity_factor * np.sqrt(expiry / 2 * np.pi) / 100

    return normal_vols


def dict_surface_to_matrix(data):
    expiries = []
    tenors = []

    for (expiry, tenor) in data:
        if expiry not in expiries:
            expiries.append(expiry)
        if tenor not in tenors:
            tenors.append(tenor)

    z_value = np.zeros((len(tenors), len(expiry)))
    for (i, tenor) in enumerate(tenors):
        for (j, expiry) in enumerate(expiries):
            z_value[i][j] = data[(expiry, tenor)]

    return np.array(expiries), np.array(tenors), np.array(z_value)


class SwaptionSurface(object):

    def __init__(self, ircurve, dictionary, mode):
        """
        dictionary (maturity, tenor):valeur
        """
        if mode =='Price':
            self.prices = dictionary
            self.normal_vols = prices_to_normal_vols(ircurve, self.prices)
        else:
            self.normal_vols = dictionary
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
        data = self.normal_vols if mode != 'Price' else self.prices
        expiries, tenors, z = dict_surface_to_matrix(data)
        x, y = np.meshgrid(expiries, tenors)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x, y, z, cmap='plasma')
        ax.set_xlabel("expiries")
        ax.set_ylabel("tenors")
        if mode == "Price":
            ax.set_zlabel("Premium (%)")
        else:
            ax.set_zlabel("volatilities (bps)")