import numpy as np


# static methods
def prices_to_normal_vols(ircurve, prices):
    pass

def normal_vols_to_prices(ircurve, normal_vols):
    pass


# object
class SwaptionSurface(object):

    def __init__(self, ircurve, dictionary, mode):
        """
        dictionary (maturity, tenor):valeur
        """
        if mode=='Price':
            self.prices = dictionary
            self.normal_vols = prices_to_normal_vols(ircurve, self.prices)
        else:
            self.normal_vols = dictionary
            self.prices = normal_vols_to_prices(ircurve, self.normal_vols)
