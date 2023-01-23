import scikitfin.markets as markets
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal



def test_actuarial_rates0() -> None:
    maturities = np.array([1,2,3,4,5,6,7,8,9,10])
    actuarial_rates = np.array([0.01]*10)
    irc = markets.InterestRateCurve(maturities, actuarial_rates, 'ActuarialRate')
    zcprices = irc.zc_prices
    new_actuarial_rates = markets.zc_prices_to_actuarial_rates(maturities, zcprices)
    assert_array_almost_equal(actuarial_rates, new_actuarial_rates)

def test_yields0() -> None:
    maturities = np.array([1,2,3,4,5,6,7,8,9,10])
    yields = np.array([0.01]*10)
    irc = markets.InterestRateCurve(maturities, yields, 'Yield')
    zcprices = irc.zc_prices
    new_yields = markets.zc_prices_to_yields(maturities, zcprices)
    assert_array_almost_equal(yields, new_yields)

def test_swap_rates() -> None:
    maturities = np.array([1,2,3,4,5,6,7,8,9,10])
    swap_rates = np.array([0.01]*10)
    irc = markets.InterestRateCurve(maturities, swap_rates, 'SwapRate')
    zcprices = irc.zc_prices
    new_swap_rates = markets.zc_prices_to_swap_rates(maturities, zcprices)
    assert_array_almost_equal(swap_rates, new_swap_rates)

def test_instant_forwards() -> None:
    maturities = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    actuarial_rates = np.array([0.01] * 10)
    irc = markets.InterestRateCurve(maturities, actuarial_rates, 'ActuarialRate')
    zcprices = irc.zc_prices
    instant_forwards = irc.instant_forwards
    new_inst_forwards = markets.zc_prices_to_inst_forwards(maturities, zcprices)
    assert_array_almost_equal(instant_forwards, new_inst_forwards)
