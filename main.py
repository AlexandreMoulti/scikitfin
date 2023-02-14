import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from scikitfin.markets import InterestRateCurve
from scikitfin.markets import SwaptionSurface
from scikitfin.models import HullWhite
from scikitfin.markets import ircurve

import pandas as pd
data_curve = pd.read_csv('courbe.csv')
######### market data ###################################

maturities = data_curve['maturity'].to_numpy()
discounts = data_curve['Discount'].to_numpy()
yields = np.ones(maturities.shape[0])*0.03
print(ircurve.zc_prices_to_actuarial_rates(maturities, discounts))
#actuarial_rates = np.array([0.01] * 10)
#swaptiondic = {(1,1):0.1, (10,10):0.05}


######## market structures ##########################################
irc = InterestRateCurve(maturities, yields, 'Yield')
#swapsurface = SwaptionSurface(irc, swaptiondic, 'price')
#print(irc.zc_prices)


######## models ###########################################


grid=0 #use grid object
#short_rates = hw.simulate(grid) #
#hw.price_zc_spot(10) #prix model
#hw.fit(ircurve)
x=0
t=0
maturity =10
tenor = 5
kappa = 0.1
sigma=0.01
schedule=np.arange(10,15.5,0.5)
strike=0.03
hw = HullWhite(kappa, sigma)
hw.fit(irc)
print(hw.price_zc_spot(10))
print(hw.price_zc(x,t,10,kappa, sigma))
print('swaption price = {}'.format(hw.price_swaption(x, t, schedule, strike, kappa, sigma)))

strike = 0.5
print('zc call price : {}'.format(hw.price_zc_call(0,0,maturity, tenor, strike, kappa, sigma)))
 #prix spot = prix market
# hw.swpation(maturity,  tenor) = prix swap manager

#plus tard
#hw.fit(ircurve, swaptiondata)
#new_short_rates = hw.simulate()

print(irc.func_zc_prices(1))
