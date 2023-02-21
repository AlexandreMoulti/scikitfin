from scikitfin.markets import InterestRateCurve
from scikitfin.markets import ircurve
from scikitfin.models import HullWhite
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import pandas as pd


def test_zc_price():
    """
    idea : test_zc at any time
    Returns
    -------

    """
    data_curve = pd.read_csv("courbe.csv")
    maturities = data_curve['maturity'].to_numpy()
    discounts  = data_curve['Discount'].to_numpy()
    irc = InterestRateCurve(maturities, discounts, 'ZeroCoupon')
    x = 0
    t = 0
    kappa = 0.1
    sigma = 0.01
    hw = HullWhite(kappa, sigma)
    hw.fit(irc)
    spot_prices = []
    new_spot_prices = []
    for mat in maturities:
        spot_prices.append(hw.price_zc_spot(mat))
        new_spot_prices.append(hw.price_zc(x, t, mat, kappa, sigma))

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
    hw = HullWhite(kappa, sigma)
    hw.fit(irc)

    premia_prices = [0.456542, 0.428708, 0.133566, 0.050064, 0.404916, 0.380230, 0.119513, 0.053572]
    hw_prices = []
    strike = 0.5

    for expiry in [1, 5]:
        for tenor in [1, 2, 15, 20]:
            hw_prices.append(hw.price_zc_call(x, t, expiry, tenor, strike, kappa, sigma))
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
    hw = HullWhite(kappa, sigma)
    hw.fit(irc)
    zbc_prices = hw.price_zc_call(x, t, maturities, tenor, strike, kappa, sigma)
    zbp_prices = hw.price_zc_put(x, t, maturities, tenor, strike, kappa, sigma)

    assert_array_almost_equal(zbc_prices + strike * hw.price_zc(x, t, maturities, kappa, sigma),
                              zbp_prices + hw.price_zc(x, t, maturities + tenor, kappa, sigma))

