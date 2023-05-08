from scikitfin.markets import InterestRateCurve
from scikitfin.models import HullWhite
from scikitfin.markets import SwaptionSurface
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
import QuantLib as ql

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


def test_swaption_price():
    """
    test the swpation price methodology vs quantlib
    """
    data_curve = pd.read_csv("courbe_1.csv")
    maturities = data_curve['maturity'].astype(int).to_list()
    discounts = data_curve['Discount'].to_list()
    yields = np.ones(len(maturities)) * 0.03
    irc = InterestRateCurve(np.array(maturities), yields, 'Yield')
    #irc = InterestRateCurve(maturities, discounts, 'ZeroCoupon')

    #### QUantLib

    tradeDate = ql.Date(25, 4, 2023)
    ql.Settings.instance().evaluationDate = tradeDate

    spot_dates = [tradeDate + ql.Period(i, ql.Years) for i in maturities]
    day_count = ql.Actual360()
    calendar = ql.France()
    interpolation = ql.Linear()
    compounding = ql.Compounded
    compounding_frequency = ql.Annual
    spot_curve = ql.ZeroCurve(spot_dates,
                              yields,
                              day_count,
                              calendar,
                              interpolation,
                              compounding,
                              compounding_frequency)
    #spot_curve = ql.FlatForward(tradeDate, ql.QuoteHandle(ql.SimpleQuote(0.03)), ql.Actual360(), ql.Compounded, ql.Annual)
    term_structure = ql.YieldTermStructureHandle(spot_curve)
    index = ql.EURLibor6M(term_structure)

    false_vols = [0.3] * 15

    fixed_leg_tenor = ql.Period(6, ql.Months)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()

    #### internal models parameters
    kappa = 0.1
    sigma = 0.02
    hw = HullWhite(kappa, sigma)
    hw.fit(irc)
    swpations_prices = []
    swpations_helpers = []

    model = ql.HullWhite(term_structure, kappa, sigma)
    engine = ql.JamshidianSwaptionEngine(model)
    method = ql.LevenbergMarquardt()
    for i, vol in enumerate(false_vols):
        expiry = i + 1
        tenor = len(false_vols) - i
        quote_vol = ql.QuoteHandle(ql.SimpleQuote(vol))
        strike_atm = float(irc.forward_swap_rate(expiry, tenor))
        helper = ql.SwaptionHelper(ql.Period(expiry, ql.Years),
                                   ql.Period(tenor, ql.Years),
                                   quote_vol,
                                   index,
                                   fixed_leg_tenor,
                                   fixed_leg_daycounter,
                                   floating_leg_daycounter,
                                   term_structure,
                                   ql.BlackCalibrationHelper.RelativePriceError,
                                   strike_atm,
                                   1,
                                   ql.Normal,
                                   0)
        helper.setPricingEngine(engine)
        swpations_helpers.append(helper)
        prices = hw.price_swaption(0, 0, expiry, tenor, strike_atm)
        swpations_prices.append(prices)

    fixedParameters = [False, True]
    endCriteria = ql.EndCriteria(10000, 100, 0.000001, 0.00000001, 0.00000001)
    model.calibrate(swpations_helpers, method, endCriteria,
                    ql.BoundaryConstraint(0, 0), [], fixedParameters)

    swaptions_qt_prices = []

    for swpt in swpations_helpers:
        swaptions_qt_prices.append(swpt.modelValue())

    assert_array_almost_equal(swaptions_qt_prices, swpations_prices, decimal=2)



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

    strike = irc.forward_swap_rate(expiry, tenor)
    hw.fit(irc, swaptionsurface, bounds)
    model_prices = hw.price_swaption(0, 0, expiry, tenor, strike)
    assert_array_almost_equal(swaptionsurface.prices, model_prices, decimal=4)