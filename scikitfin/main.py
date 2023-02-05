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

hw = HullWhite(init=0.01, kappa=2/10, sigma=2)
grid=0 #use grid object
short_rates = hw.simulate(grid) #
hw.fit(ircurve, swaptiondata)
#new_short_rates = hw.simulate()
