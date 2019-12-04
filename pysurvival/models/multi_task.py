from __future__ import absolute_import
import torch 
import numpy as np
import copy
import multiprocessing
from pysurvival import HAS_GPU
from pysurvival import utils
from pysurvival.utils import neural_networks as nn
from pysurvival.utils import optimization as opt
from pysurvival.models import BaseModel
# %matplotlib inline

class BaseMultiTaskModel(BaseModel):
    """ Base class for all  Multi-Task estimators:
        * Multi-Task Logistic Regression model (MTLR)
        * Neural Multi-Task Logistic Regression model (N-MTLR)
    BaseMultiTaskModel shouldn't be used as is. 
    The underlying model is written in PyTorch.

    The original Multi-Task model, a.k.a the Multi-Task Logistic Regression
    model (MTLR), was first introduced by  Chun-Nam Yu et al. in 
    *Learning Patient-Specific Cancer Survival Distributions as a Sequence of 
    Dependent Regressors*

    The Neural Multi-Task Logistic Regression model (N-MTLR) was developed 
    by S. Fotso in the paper *Deep Neural Networks for Survival 
    Analysis Based on a Multi-Task Framework*, allowing the use of 
    Neural Networks within the original design.

    Parameters
    ----------
    * `structure`:  **list of dictionaries** -- 
        Provides the structure of the MLP built within the N-MTLR.
        
        ex: `structure = [ {'activation': 'ReLU', 'num_units': 128}, ]`.

        Each dictionary corresponds to a fully connected hidden layer:

        * `num_units` is the number of hidden units in this layer
        * `activation` is the activation function that will be used. 
        The list of all available activation functions can be found :
            * Atan
            * BentIdentity
            * BipolarSigmoid
            * CosReLU
            * ELU
            * Gaussian
            * Hardtanh
            * Identity
            * InverseSqrt
            * LeakyReLU
            * LeCunTanh
            * LogLog
            * LogSigmoid
            * ReLU
            * SELU
            * Sigmoid
            * Sinc
            * SinReLU
            * Softmax
            * Softplus
            * Softsign
            * Swish
            * Tanh
    
        In case there are more than one dictionary, 
        each hidden layer will be applied in the resulting MLP, 
        using the order it is provided in the structure:
        ex: structure = [ {'activation': 'relu', 'num_units': 128}, 
                          {'activation': 'tanh', 'num_units': 128}, ] 

    * `bins`: **int** *(default=100)* -- 
         Number of subdivisions of the time axis 

    * `auto_scaler`: **boolean** *(default=True)* -- 
        Determines whether a sklearn scaler should be automatically applied
    """

    def __init__(self, structure, bins = 100, auto_scaler=True):

        # Saving the attributes
        self.loss_values = []
        self.bins = bins
        self.structure = structure

        # Initializing the elements from BaseModel
        super(BaseMultiTaskModel, self).__init__(auto_scaler)
        
        
    def get_times(self, T, is_min_time_zero = True, extra_pct_time = 0.1):
        """ Building the time axis (self.times) as well as the time intervals 
            ( all the [ t(k-1), t(k) ) in the time axis.
        """

        # Setting the min_time and max_time
        max_time = max(T)
        if is_min_time_zero :
            min_time = 0. 
        else:
            min_time = min(T)
        
        # Setting optional extra percentage time
        if 0. <= extra_pct_time <= 1.:
            p = extra_pct_time
        else:
            raise Exception("extra_pct_time has to be between [0, 1].") 

        # Building time points and time buckets
        self.times = np.linspace(min_time, max_time*(1. + p), self.bins)
        self.get_time_buckets()
        self.num_times = len(self.time_buckets)


    def compute_XY(self, X, T, E, is_min_time_zero, extra_pct_time):
        """ Given the survival_times, events and time_points vectors, 
            it returns a ndarray of the encodings for all units 
            such that:
                Y = [[[0, 0, 1, 0, 0], # unit experienced event 1 at t = 3
                     [0, 1, 0, 0, 0],  # unit experienced event 2 at t = 2
                     [0, 1, 1, 1, 1],  # unit was censored at t = 2
                     ], 
                    [[0, 0, 1, 1, 1],  # unit was censored for event 2 at t = 2
                     [0, 1, 1, 1, 1],  # unit was censored event 2 at t = 2
                     [0, 1, 1, 1, 1],  # unit was censored for event 2 at t = 2
                    ],]
        """

        # building times axis
        self.get_times(T, is_min_time_zero, extra_pct_time)
        n_units = T.shape[0]

        # Initializing the output variable
        Y = []

        # Building the output variable
        for i, (t, e) in enumerate(zip(T, E)):
            y = np.zeros(self.num_event_types * (self.num_times + 1))
            min_abs_value = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_abs_value)
            if e != 0:
                y[int((self.num_times + 1) * (e-1) + index)] = 1.0
            for j in range(self.num_event_types):
                if j != e - 1 or e == 0:
                    y[(self.num_times + 1) * (j) + index: ] = 1.0
            Y.append(y)

        # Transform into torch.Tensor
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)

        return X, Y

    def loss_function(self, model, X, Y, Triangle, l2_reg, l2_smooth):
        """ Computes the loss function of the any MTLR model. 
            All the operations have been vectorized to ensure optimal speed
        """
        score = model(X)
        loss = 0
        norm = torch.FloatTensor(np.zeros((X.shape[0]), dtype=np.float32))
        phi_reduced = torch.FloatTensor(np.zeros((X.shape[0]), dtype=np.float32))
        for i in range(self.num_event_types):
            subScore = score[:, ((self.num_times + 1) * i) : ((self.num_times + 1) * (i + 1))]
            subY = Y[:, ((self.num_times + 1) * i) : ((self.num_times + 1) * (i + 1))]
            phi = torch.exp(torch.mm(subScore, Triangle))
            # phi_reduced = phi_reduced + torch.sum(phi * subY, dim = 1)
            phi_reduced = torch.sum(phi * subY, dim = 1)
            # norm = norm + torch.sum(torch.mm(phi, subTri), dim=1)
            norm = torch.sum(torch.mm(phi, Triangle), dim=1)

            loss = - torch.sum(torch.log(phi_reduced)) + torch.sum(torch.log(norm))
        # loss = - torch.sum(torch.log(phi_reduced)) + torch.sum(torch.log(norm))


        # Adding the regularized loss
        nb_set_parameters = len(list(model.parameters()))
        for i, w in enumerate(model.parameters()):
            loss += l2_reg*torch.sum(w*w)/2.
            
            if i >= nb_set_parameters - 2:
                loss += l2_smooth*norm_diff(w)
                
        return loss


    def fit(self, X, T, E, init_method = 'glorot_uniform', optimizer ='adam', 
            lr = 1e-4, num_epochs = 1000, dropout = 0.2, l2_reg=1e-2, 
            l2_smooth=1e-2, batch_normalization=False, bn_and_dropout=False,
            verbose=True, extra_pct_time = 0.1, is_min_time_zero=True):
        """ Fit the estimator based on the given parameters.

        Parameters:
        -----------
        * `X` : **array-like**, *shape=(n_samples, n_features)* --
            The input samples.

        * `T` : **array-like** -- 
            The target values describing when the event of interest or censoring
            occurred.

        * `E` : **array-like** --
            The values that indicate if the event of interest occurred i.e.: 
            E[i]=1 corresponds to an event, and E[i] = 0 means censoring, 
            for all i.

        * `init_method` : **str** *(default = 'glorot_uniform')* -- 
            Initialization method to use. Here are the possible options:

            * `glorot_uniform`: Glorot/Xavier uniform initializer
            * `he_uniform`: He uniform variance scaling initializer
            * `uniform`: Initializing tensors with uniform (-1, 1) distribution
            * `glorot_normal`: Glorot normal initializer,
            * `he_normal`: He normal initializer.
            * `normal`: Initializing tensors with standard normal distribution
            * `ones`: Initializing tensors to 1
            * `zeros`: Initializing tensors to 0
            * `orthogonal`: Initializing tensors with a orthogonal matrix,

        * `optimizer`:  **str** *(default = 'adam')* -- 
            iterative method for optimizing a differentiable objective function.
            Here are the possible options:

            - `adadelta`
            - `adagrad`
            - `adam`
            - `adamax`
            - `rmsprop`
            - `sparseadam`
            - `sgd`

        * `lr`: **float** *(default=1e-4)* -- 
            learning rate used in the optimization

        * `num_epochs`: **int** *(default=1000)* -- 
            The number of iterations in the optimization

        * `dropout`: **float** *(default=0.5)* -- 
            Randomly sets a fraction rate of input units to 0 
            at each update during training time, which helps prevent overfitting.

        * `l2_reg`: **float** *(default=1e-4)* -- 
            L2 regularization parameter for the model coefficients

        * `l2_smooth`: **float** *(default=1e-4)* -- 
            Second L2 regularizer that ensures the parameters vary smoothly 
            across consecutive time points.

        * `batch_normalization`: **bool** *(default=True)* -- 
            Applying Batch Normalization or not

        * `bn_and_dropout`: **bool** *(default=False)* -- 
            Applying Batch Normalization and Dropout at the same time

        * `display_loss`: **bool** *(default=True)* -- 
            Whether or not showing the loss function values at each update

        * `verbose`: **bool** *(default=True)* -- 
            Whether or not producing detailed logging about the modeling

        * `extra_pct_time`: **float** *(default=0.1)* -- 
            Providing an extra fraction of time in the time axis

        * `is_min_time_zero`: **bool** *(default=True)* -- 
            Whether the the time axis starts at 0

        **Returns:**

        * self : object


        Example:
        --------
            
        #### 1 - Importing packages
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        from sklearn.model_selection import train_test_split
        from pysurvival.models.simulations import SimulationModel
        from pysurvival.models.multi_task import LinearMultiTaskModel
        from pysurvival.utils.metrics import concordance_index
        #%matplotlib inline  # To use with Jupyter notebooks


        #### 2 - Generating the dataset from a Weibull parametric model
        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'Weibull',  
                               risk_type = 'linear',
                               censored_parameter = 10.0, 
                               alpha = .01, beta = 3.0 )

        # Generating N random samples 
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features = 3)

        # Showing a few data-points 
        time_column = 'time'
        event_column = 'event'
        dataset.head(2)

        #### 3 - Creating the modeling dataset
        # Defining the features
        features = sim.features

        # Building training and testing sets #
        index_train, index_test = train_test_split( range(N), test_size = 0.2)
        data_train = dataset.loc[index_train].reset_index( drop = True )
        data_test  = dataset.loc[index_test].reset_index( drop = True )

        # Creating the X, T and E input
        X_train, X_test = data_train[features], data_test[features]
        T_train, T_test = data_train['time'].values, data_test['time'].values
        E_train, E_test = data_train['event'].values, data_test['event'].values

        #### 4 - Initializing a MTLR model and fitting the data.
        # Building a Linear model
        mtlr = LinearMultiTaskModel(bins=50) 
        mtlr.fit(X_train, T_train, E_train, lr=5e-3, init_method='orthogonal')

        # Building a Neural MTLR
        # structure = [ {'activation': 'Swish', 'num_units': 150},  ]
        # mtlr = NeuralMultiTaskModel(structure=structure, bins=150) 
        # mtlr.fit(X_train, T_train, E_train, lr=5e-3, init_method='adam')

        #### 5 - Cross Validation / Model Performances
        c_index = concordance_index(mtlr, X_test, T_test, E_test) #0.95
        print('C-index: {:.2f}'.format(c_index))

        """

        # Checking data format (i.e.: transforming into numpy array)
        X, T, E = utils.check_data(X, T, E)
        self.num_event_types = int(max(E))

        # Extracting data parameters
        nb_units, self.num_vars = X.shape
        input_shape = self.num_vars
    
        # Scaling data 
        if self.auto_scaler:
            X = self.scaler.fit_transform( X ) 

        # Building the time axis, time buckets and output Y
        X, Y = self.compute_XY(X, T, E, is_min_time_zero, extra_pct_time)

        # Initializing the model
        model = nn.NeuralNet(input_shape, (self.num_times + 1) * self.num_event_types, self.structure, 
                             init_method, dropout, batch_normalization, 
                             bn_and_dropout )


        # Creating the Triangular matrix
        Triangle = torch.FloatTensor(np.tri(self.num_times+1, self.num_times + 1 ))

        # Performing order 1 optimization
        model, loss_values = opt.optimize(self.loss_function, model, optimizer, 
            lr, num_epochs, verbose,  X=X, 
            Y=Y, Triangle=Triangle, 
            l2_reg=l2_reg, l2_smooth=l2_smooth)

        # Saving attributes
        self.model = model.eval()
        self.loss_values = loss_values

        return self

    def get_triangle(self):
        Triangle = torch.FloatTensor(np.tri(self.num_times + 1, self.num_times + 1, dtype=np.float32))
        for i in range(self.num_event_types - 1):
            Triangle = torch.cat((Triangle, torch.FloatTensor(np.tri(self.num_times + 1, self.num_times + 1, dtype=np.float32))), 0)
        return Triangle.numpy()
    

    def predict(self, x, t = None):
        """ Predicting the hazard, density and survival functions
        
        Parameters:
        ----------
        * `x` : **array-like** *shape=(n_samples, n_features)* --
            array-like representing the datapoints. 
            x should not be standardized before, the model
            will take care of it

        * `t`: **double** *(default=None)* --
             time at which the prediction should be performed. 
             If None, then return the function for all available t.
        """
        
        # Convert x into the right format
        x = utils.check_data(x)

        # Scaling the data
        if self.auto_scaler:
            if x.ndim == 1:
                x = self.scaler.transform( x.reshape(1, -1) )
            elif x.ndim == 2:
                x = self.scaler.transform( x )
        else:
            # Ensuring x has 2 dimensions
            if x.ndim == 1:
                x = np.reshape(x, (1, -1))

        # Transforming into pytorch objects
        x = torch.FloatTensor(x)
                
        # Predicting using linear/nonlinear function
        score_torch = self.model(x)
        score = score_torch.data.numpy()
                
        phis = []
        # norm = np.zeros((x.shape[0], self.num_times+1), dtype=np.float32)
        norm = np.zeros((x.shape[0]), dtype=np.float32)

        densities = []
        hazards = []
        Incidences = []
        Survivals = []

        # Calculating the score, density, hazard and Survival
        Triangle = np.tri(self.num_times+1, self.num_times + 1 )
        for i in range(self.num_event_types):
            subScore = score[:, (self.num_times + 1) * i : (self.num_times + 1) * (i + 1)]
            phi = np.exp(np.matmul(subScore, Triangle))

            # phis.append(phi)
            # norm += np.sum(phi, axis=1)

            norm = np.sum(phi, axis=1)
            density = (phi.T / norm).T
            Survival = np.dot(density, Triangle)
            # hazard = density[:, :]/Survival[:, :]
            hazard = density[:, :-1]/Survival[:, 1:]
            densities.append(density)
            Survivals.append(Survival)
            Incidences.append(1-Survival)
            hazards.append(hazard)



            # div = np.repeat(np.sum(phi, 1).reshape(-1, 1), phi.shape[1], axis=1)
            # norm += div

            
        # for phi in phis:
        #     density = (phi.T / norm).T
        #     Survival = np.dot(density, Triangle)
        #     # hazard = density[:, :]/Survival[:, :]
        #     hazard = density[:, :-1]/Survival[:, 1:]

        #     densities.append(density)
        #     hazards.append(hazard)
        #     Survivals.append(Survival)
        #     Incidences.append(1-Survival)

            # Incidences.append((1 - np.matmul(phi, subTri).T / norm).T)
           
            #     lambda dens: reduce(lambda a, b: a + [a[-1] + b], dens, [0])[1:],
            #     axis=1,
            #     arr=density[-1],
            # ))

        # Returning the full functions of just one time point
        if t is None:
            return None, densities, Incidences
        else:
            min_abs_value = [abs(a_j_1-t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_abs_value)
            return None, density[:, :, index], Incidence[:, :, index]


    def predict_risk(self, x, use_log=False):
        """ Computing the risk score 

        Parameters:
        -----------
        * `x` : **array-like** *shape=(n_samples, n_features)* --
            array-like representing the datapoints. 
            x should not be standardized before, the model
            will take care of it

        * `use_log`: **bool** *(default=True)* -- 
            Applies the log function to the risk values

        """

        risk = super(BaseMultiTaskModel, self).predict_risk(x)
        if use_log:
            return np.log(risk)
        else:
            return risk



class LinearMultiTaskModel(BaseMultiTaskModel):
    """ LinearMultiTaskModel is the original Multi-Task model, 
        a.k.a the Multi-Task Logistic Regression model (MTLR).
        It was first introduced by  Chun-Nam Yu et al. in 
        Learning Patient-Specific Cancer Survival Distributions 
        as a Sequence of Dependent Regressors
        
        Reference:
        ----------
            * http://www.cs.cornell.edu/~cnyu/papers/nips11_survival.pdf
        
        Parameters:
        ----------
            * bins: int 
                Number of subdivisions of the time axis 

            * auto_scaler: boolean (default=True)
                Determines whether a sklearn scaler should be automatically 
                applied

    """
    
    def __init__(self, bins = 100, auto_scaler=True):
        super(LinearMultiTaskModel, self).__init__(
            structure = None, bins = bins, auto_scaler=auto_scaler)


    def fit(self, X, T, E, init_method = 'glorot_uniform', optimizer ='adam', 
            lr = 1e-4, num_epochs = 1000, l2_reg=1e-2, l2_smooth=1e-2, 
            verbose=True, extra_pct_time = 0.1, is_min_time_zero=True):

        super(LinearMultiTaskModel, self).fit(X=X, T=T, E=E, 
            init_method = init_method, optimizer =optimizer, 
            lr = lr, num_epochs = num_epochs, dropout = None, l2_reg=l2_reg, 
            l2_smooth=l2_smooth, batch_normalization=False, 
            bn_and_dropout=False, verbose=verbose, 
            extra_pct_time = extra_pct_time, is_min_time_zero=is_min_time_zero)

        return self
    

class NeuralMultiTaskModel(BaseMultiTaskModel):
    """ NeuralMultiTaskModel is the Neural Multi-Task Logistic Regression 
        model (N-MTLR) was developed by Fotso S. in 
        Deep Neural Networks for Survival Analysis Based on a 
        Multi-Task Framework, 
        allowing the use of Neural Networks within the original design.
        
    Reference:
    ----------
    * https://arxiv.org/pdf/1801.05512

    Parameters:
    ----------
    * `structure`:  **list of dictionaries** -- 
        Provides the structure of the MLP built within the N-MTLR.
        
        ex: `structure = [ {'activation': 'ReLU', 'num_units': 128}, ]`.

        Each dictionary corresponds to a fully connected hidden layer:

        * `units` is the number of hidden units in this layer
        * `activation` is the activation function that will be used. 
        The list of all available activation functions can be found :
            * Atan
            * BentIdentity
            * BipolarSigmoid
            * CosReLU
            * ELU
            * Gaussian
            * Hardtanh
            * Identity
            * InverseSqrt
            * LeakyReLU
            * LeCunTanh
            * LogLog
            * LogSigmoid
            * ReLU
            * SELU
            * Sigmoid
            * Sinc
            * SinReLU
            * Softmax
            * Softplus
            * Softsign
            * Swish
            * Tanh
    
        In case there are more than one dictionary, 
        each hidden layer will be applied in the resulting MLP, 
        using the order it is provided in the structure:
        ex: structure = [ {'activation': 'relu', 'num_units': 128}, 
                          {'activation': 'tanh', 'num_units': 128}, ] 

    * `bins`: **int** *(default=100)* -- 
         Number of subdivisions of the time axis 

    * `auto_scaler`: **boolean** *(default=True)* -- 
        Determines whether a sklearn scaler should be automatically applied
    """
    
    def __init__(self, structure, bins = 100, auto_scaler = True):

        # Checking the validity of structure
        structure = nn.check_mlp_structure(structure)

        # Initializing the instance
        super(NeuralMultiTaskModel, self).__init__(
            structure = structure, bins = bins, auto_scaler = auto_scaler)
    
    
    def __repr__(self):
        """ Representing the class object """

        if self.structure is None:
            super(NeuralMultiTaskModel, self).__repr__()
            return self.name
            
        else:
            S = len(self.structure)
            self.name = self.__class__.__name__
            empty = len(self.name)
            self.name += '( '
            for i, s in enumerate(self.structure):
                n = 'Layer({}): '.format(i+1)
                activation = nn.activation_function(s['activation'], 
                    return_text=True)
                n += 'activation = {}, '.format( s['activation'] )
                n += 'units = {} '.format( s['num_units'] )
                
                if i != S-1:
                    self.name += n + '; \n'
                    self.name += empty*' ' + '  '
                else:
                    self.name += n
            self.name = self.name + ')'
            return self.name


def norm_diff(W):
    """ Special norm function for the last layer of the MTLR """
    dims=len(W.shape)
    if dims==1:
        diff = W[1:]-W[:-1]
    elif dims==2:
        diff = W[1:, :]-W[:-1, :]
    return torch.sum(diff*diff)
