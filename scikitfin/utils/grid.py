
class DiscretizationGrid(object):
    """ 
    Represents the discretization grid used in Monte Carlo simulations.
    """

    def __init__(self, frequency, nb_simulations, horizon, horizon_sim=None):
        """
        Initializer
        
        Parameters
        ----------
        frequency      : str
            frequency of the projection dates within the following string :\n
                'Daily', 'Weekly', 'Monthly', 'Quarterly', 'biAnnualy', 'Yearly'
        nb_simulations : int
            number of Monte-Carlo simulations performed
        horizon        : int
            horizon of the simulations displayed (in years)
        horizon_sim    : int
            horizon of the simulations computed (in years)
        """
        self.horizon        = horizon
        self.horizon_sim    = horizon_sim if not(horizon_sim==None) else horizon
        self.nb_simulations = nb_simulations
        if frequency=='Daily':
            self.nb_steps         = 240
            self.dt               = 1.0/self.nb_steps
            self.projection_dates = np.linspace(0, self.horizon_sim, 240*self.horizon_sim+1)
            self.time_grid        = np.linspace(0, self.horizon, self.nb_steps*self.horizon+1, endpoint=True)
            self.time_grid_sim    = np.linspace(0, self.horizon_sim, self.nb_steps*self.horizon_sim+1, endpoint=True)
        elif frequency=='Weekly':
            self.nb_steps         = 48
            self.dt               = 1.0/self.nb_steps
            self.projection_dates = np.linspace(0, self.horizon_sim, 48*self.horizon_sim+1)
            self.time_grid        = np.linspace(0, self.horizon, self.nb_steps*self.horizon+1, endpoint=True)
            self.time_grid_sim    = np.linspace(0, self.horizon_sim, self.nb_steps*self.horizon_sim+1, endpoint=True)
        elif frequency=='Monthly':
            self.nb_steps         = 48
            self.dt               = 1.0/self.nb_steps
            self.projection_dates = np.linspace(0, self.horizon_sim, 12*self.horizon_sim+1)
            self.time_grid        = np.linspace(0, self.horizon, self.nb_steps*self.horizon+1, endpoint=True)
            self.time_grid_sim    = np.linspace(0, self.horizon_sim, self.nb_steps*self.horizon_sim+1, endpoint=True)            
        elif frequency=='Quarterly':
            self.nb_steps         = 48
            self.dt               = 1.0/self.nb_steps
            self.projection_dates = np.linspace(0, self.horizon_sim, 4*self.horizon_sim+1)
            self.time_grid        = np.linspace(0, self.horizon, self.nb_steps*self.horizon+1, endpoint=True)
            self.time_grid_sim    = np.linspace(0, self.horizon_sim, self.nb_steps*self.horizon_sim+1, endpoint=True)            
        elif frequency=='biAnnually':
            self.nb_steps         = 48
            self.dt               = 1.0/self.nb_steps
            self.projection_dates = np.linspace(0, self.horizon_sim, 2*self.horizon_sim+1)
            self.time_grid        = np.linspace(0, self.horizon, self.nb_steps*self.horizon+1, endpoint=True)
            self.time_grid_sim    = np.linspace(0, self.horizon_sim, self.nb_steps*self.horizon_sim+1, endpoint=True)
        else:
            self.nb_steps         = 48
            self.dt               = 1.0/self.nb_steps
            self.projection_dates = np.linspace(0, self.horizon_sim, self.horizon_sim+1)
            self.time_grid        = np.linspace(0, self.horizon, self.nb_steps*self.horizon+1, endpoint=True)
            self.time_grid_sim    = np.linspace(0, self.horizon_sim, self.nb_steps*self.horizon_sim+1, endpoint=True)
        self.shape     = (self.nb_simulations, self.time_grid.size)
        self.shape_sim = (self.nb_simulations, self.time_grid_sim.size)
