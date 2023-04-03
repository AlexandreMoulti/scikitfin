from scikitfin.markets import InterestRateCurve
from scikitfin.models import HullWhite
from scikitfin.markets import SwaptionSurface
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal


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
    hw = HullWhite(kappa, sigma)
    hw.fit(irc)
    new_spot_prices = hw.price_zc_spot(maturities)
    assert_array_almost_equal(irc.zc_prices, new_spot_prices)


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
    hw=HullWhite(kappa, sigma)
    hw.fit(irc)
    premia_prices = [0.456542, 0.428708, 0.133566, 0.050064, 0.404916, 0.380230, 0.119513, 0.053572]
    hw_prices = []
    strike = 0.5

    for expiry in [1, 5]:
        for tenor in [1, 2, 15, 20]:
            hw_prices.append(hw.price_zc_call(x, t, expiry, tenor, strike))

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
    hw=HullWhite(kappa, sigma)
    hw.fit(irc)
    zbc_prices = hw.price_zc_call(x, t, maturities, tenor, strike)
    zbp_prices = hw.price_zc_put(x, t, maturities, tenor, strike)

    assert_array_almost_equal(zbc_prices + strike * hw.price_zc(x, t, maturities),
                              zbp_prices + hw.price_zc(x, t, maturities + tenor))


def test_fit_method():
    """
    test the coherence ( not the return) of the fit method
    Returns
    -------

    """
    bounds = [(0, 10), (0,10)]
    expiry, tenor = 1, 30
    data_curve = pd.read_csv("courbe.csv")
    maturities = data_curve['maturity'].to_numpy()
    discounts = data_curve['Discount'].to_numpy()
    irc = InterestRateCurve(maturities, discounts, 'ZeroCoupon')
    swaptionsurface = SwaptionSurface(irc, expiry, tenor, 0.02, mode='vol')

    hw = HullWhite()

    strike=irc.forward_swap_rate(expiry, tenor)
    hw.fit(irc, swaptionsurface, bounds)
    model_prices = hw.price_swaption(0, 0, expiry, tenor, strike)
    print(hw.kappa, hw.sigma)
    assert_array_almost_equal(swaptionsurface.prices, model_prices)