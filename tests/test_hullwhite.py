from scikitfin.markets import InterestRateCurve
from scikitfin.markets import ircurve
from scikitfin.models import HullWhite, hullwhite
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import pandas as pd
from swaptionprice import array_of_tuple

def test_zc_price():
    """
    idea : test_zc at any time
    Returns
    -------

    """
    data_curve = pd.read_csv("courbe.csv")
    maturities = data_curve['maturity'].to_numpy()
    discounts = data_curve['Discount'].to_numpy()
    irc = InterestRateCurve(maturities, discounts, 'ZeroCoupon')
    x = 0
    t = 0
    kappa = 0.1
    sigma = 0.01

    spot_prices = []
    new_spot_prices = []
    for mat in maturities:
        spot_prices.append(hullwhite.price_zc_spot(mat, irc.func_zc_prices))
        new_spot_prices.append(hullwhite.price_zc(x, t, mat, irc.func_zc_prices, kappa, sigma))

    assert_array_almost_equal(spot_prices, new_spot_prices)


def test_zc_call():
    """
    idea : test the zero coupon bond call with premia prices
    Returns
    -------

    """
    data_curve = pd.read_csv("courbe.csv")
    maturities = data_curve['maturity'].to_numpy()
    yields = np.ones(maturities.shape[0]) * 0.03
    irc = InterestRateCurve(maturities, yields, 'Yield')
    x = 0
    t = 0
    kappa = 0.1
    sigma = 0.01

    premia_prices = [0.456542, 0.428708, 0.133566, 0.050064, 0.404916, 0.380230, 0.119513, 0.053572]
    hw_prices = []
    strike = 0.5

    for expiry in [1, 5]:
        for tenor in [1, 2, 15, 20]:
            hw_prices.append(hullwhite.price_zc_call(x, t, expiry, tenor, strike, irc.func_zc_prices, kappa, sigma))
    assert_array_almost_equal(premia_prices, hw_prices)


def test_call_parity():
    """
    test call parity on ZCB
    Returns
    -------

    """
    data_curve = pd.read_csv("courbe.csv")
    maturities = data_curve['maturity'].to_numpy()
    discounts = data_curve['Discount'].to_numpy()
    irc = InterestRateCurve(maturities, discounts, 'ZeroCoupon')
    tenor = 10
    strike = 0.5
    x = 0
    t = 0
    kappa = 0.1
    sigma = 0.01

    zbc_prices = hullwhite.price_zc_call(x, t, maturities, tenor, strike, irc.func_zc_prices, kappa, sigma)
    zbp_prices = hullwhite.price_zc_put(x, t, maturities, tenor, strike, irc.func_zc_prices, kappa, sigma)

    assert_array_almost_equal(zbc_prices + strike * hullwhite.price_zc(x, t, maturities, irc.func_zc_prices, kappa, sigma),
                              zbp_prices + hullwhite.price_zc(x, t, maturities + tenor, irc.func_zc_prices, kappa, sigma))


def test_fit_method():
    """
    test the coherence ( not the return) of the fit method
    Returns
    -------

    """

    swaptionsurface = array_of_tuple
    bounds = [(0, 10), (0,10)]
    data_curve = pd.read_csv("courbe.csv")
    maturities = data_curve['maturity'].to_numpy()
    discounts = data_curve['Discount'].to_numpy()
    irc = InterestRateCurve(maturities, discounts, 'ZeroCoupon')
    kappa = 0.1
    sigma = 0.01
    hw = HullWhite(kappa, sigma)
    a = hullwhite.price_swaption(0, 0, np.array([(1, 30, 0.2)], dtype=[("maturity", "i4"), ("tenor", "i4"), ("value", "f8")])
                                 , 0.02, irc.func_zc_prices, kappa, sigma)

    hw.fit(irc, swaptionsurface, bounds)

    b, c = hullwhite.compute_model_premiums(irc, swaptionsurface, hw.kappa, hw.sigma)
    print(hw.kappa, hw.sigma)
    assert_array_almost_equal(b, c)