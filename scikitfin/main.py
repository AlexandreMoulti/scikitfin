import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from scikitfin.markets import InterestRateCurve
from scikitfin.markets import SwaptionSurface
from scikitfin.models import HullWhite

######### market data ###################################

maturities = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
actuarial_rates = np.array([0.01] * 10)
swaptiondic = {(1,1):0.1, (10,10):0.05}


######## market structures ##########################################
irc = InterestRateCurve(maturities, actuarial_rates, 'ActuarialRate')
swapsurface = swaptionsurface.SwaptionSurface(irc, swaptiondic, 'price')
print(irc.zc_prices)


######## models ###########################################

hw = HullWhite(kappa=2/10, sigma=2)
grid=0 #use grid object
short_rates = hw.simulate(grid) #
hw.price_zc(10) #prix model
hw.fit(ircurve)
hw.price_zc(10) #prix spot = prix market
# hw.swpation(maturity,  tenor) = prix swap manager

#plus tard
hw.fit(ircurve, swaptiondata)
#new_short_rates = hw.simulate()
