import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
import time
import inspect

from sklearn.preprocessing import LabelBinarizer, StandardScaler
import part1_nn_lib
import matplotlib.pyplot as plt

class Regressor():

    def __init__(self, x, test_set=None, batch_size = 32, nb_epoch = 10, layers=2, input_neurons=24, optimizer='adam', lr=0.001, activation_function=nn.ReLU(),
                 loss_fn=nn.MSELoss()):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - layers {int} -- number of layers 
            - input_neurons {int} -- number of neurons in first layer (output size)
            - optimizer {string} -- optimizer to use
            - lr {float} -- learning rate to use
            - activation_function - {torch.nn} -- activation function to use
            - test_set -- test set for plotting test/train curve

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
    
        ## Network Setup ##
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.model = self._build_network(self.input_size, input_neurons, layers, activation_function)
        self.loss_fn = loss_fn

        # available optimizers
        self.optimizers = {
            'adam' : optim.Adam(self.model.parameters(), lr = lr),
            'adagrad' : optim.Adagrad(self.model.parameters(), lr=lr),
            'adadelta' : optim.Adadelta(self.model.parameters(), lr=lr)
        }
        
        # chooses optimizer
        self.optimizer = self._select_optimizer(optimizer) 
        
        ## Preprocessing Setup ##
        self.columns=None
        self.xmin=None
        self.xmax=None
        self.ymin=None
        self.ymax=None
        self.means=None

        ## Errors to save during training ## 
        self.train_errors = []
        self.test_errors = []
        self.test_set = test_set
        self.batch_size = batch_size
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _build_network(self, input_size, input_neurons, layers, activation_function):
        """
        Builds the regressor model iteratively based on input parameters 

        Arguments:
            - input_size {int} -- input size to network
            - input_neurons {int} -- number of neurons (output size) of first layer
            - layers {int} -- number of layers in the network 
            - activation_function {torch.nn} -- activation function to add 
 
        """
        # set the input layer and first activation function 
        input_layer = nn.Linear(input_size, input_neurons)
        model = nn.Sequential(input_layer, activation_function)

        # add all other layers
        output_size = input_neurons 
        for i in range(layers-1):
            model.append(nn.Linear(output_size, int(output_size/2)))
            model.append(activation_function)
            output_size = int(output_size/2)

        # add final layer
        model.append(nn.Linear(output_size, 1))

        return model

    def _select_optimizer(self, optimizer):
        """
        Helper function to select an optimizer for the network from dictionary of optimizers 
        based on user inputer
        """
        optimizer = optimizer.lower()
        if optimizer not in self.optimizers:
            optimizer = self.optimizers['adam']
        else:
            return self.optimizers[optimizer]
    
    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        # print(x)
        
        # when training determine mean and min/max to use for future test examples
        if training:
            # determine mean of columns and categorical classes
            self.means = x.mean(numeric_only=True)
            # fill NaN with mean, encode categorical labels with OHC 
            x = x.fillna(self.means)
            x_encoded = pd.get_dummies(x, columns=['ocean_proximity'],dtype=int)
            self.columns =x_encoded.columns
            # determine min and max to be used for normalizing
            self.xmin=x_encoded.min()
            self.xmax=x_encoded.max()
            normalized_encoded_x=(x_encoded-self.xmin)/(self.xmax-self.xmin)

            np_x=normalized_encoded_x.to_numpy()
            if y is None:
                return torch.tensor(np_x, dtype=torch.float32), y
            else:
                self.ymin=y.min()
                self.ymax=y.max()
                normalized_y=(y- self.ymin) / (self.ymax - self.ymin)
                np_y=normalized_y.to_numpy()
                return torch.tensor(np_x, dtype=torch.float32), torch.tensor(np_y, dtype=torch.float32)
                
        else:
            # fill empty examples with the mean
            x = x.fillna(self.means)
            # encode categorical variable
            x_encoded = pd.get_dummies(x, columns=['ocean_proximity'], dtype=int)
            # reindex in order of training data and add any missing columns 
            x_encoded = x_encoded.reindex(columns=self.columns, fill_value=0)
            # feature scaling using the same parameters as in the training data
            normalized_encoded_x = (x_encoded - self.xmin) / (self.xmax - self.xmin)
            np_x=normalized_encoded_x.to_numpy()
            if y is None:
                return torch.tensor(np_x, dtype=torch.float32), y
            else:
                normalized_y=(y-self.ymin) / (self.ymax - self.ymin)
                np_y=normalized_y.to_numpy()
                return torch.tensor(np_x, dtype=torch.float32), torch.tensor(np_y, dtype=torch.float32)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # • Perform forward pass though the model given the input.
        # • Compute the loss based on this forward pass.
        # • Perform backwards pass to compute gradients of loss with respect to parameters of the model.
        # • Perform one step of gradient descent on the model parameters.

        X_train, y_train = self._preprocessor(x, y, training = True)
    
        if(self.test_set != None):
            X_test, y_test = self._preprocessor(self.test_set[0], self.test_set[1])
            number_of_batches_test = len(X_test) // self.batch_size

        number_of_batches_train = len(X_train) // self.batch_size

        self.model.train() # set model in train mode
        for epoch in range(self.nb_epoch):
            batch_train_error=[]
            batch_test_error=[]
            for i in range(number_of_batches_train):
                x, y = (X_train[i * self.batch_size : (i + 1) * self.batch_size],
                y_train[i * self.batch_size : (i + 1) * self.batch_size])
                # reset gradients 
                self.optimizer.zero_grad()
                self.model.train()
                # forward pass train set 
                y_pred = self.model(x)
                # calculate loss function
                loss = self.loss_fn(y_pred, y)
                # backward pass
                loss.backward()
                # update weights
                self.optimizer.step()
                # record error 
                batch_train_error.append(float(self.score(x, y)))

            # if using test set 
            if(self.test_set != None):
                for i in range(number_of_batches_test):
                    x, y = (X_test[i * self.batch_size : (i + 1) * self.batch_size],
                    y_test[i * self.batch_size : (i + 1) * self.batch_size])
                    # append to errors list
                    batch_test_error.append(float(self.score(x, y)))
                self.test_errors.append(np.array(batch_test_error).mean())
            self.train_errors.append(np.array(batch_train_error).mean())

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """
        #make sure in same order as training dataset
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #print(inspect.getsource(inspect.getmodule(inspect.currentframe())))
        predictions = []
        # preprocess data 
        X, _ = self._preprocessor(x, training = False)
        for row in X:
            y = self.model.forward(row).detach().numpy()[0]   
            predictions.append((y * (self.ymax - self.ymin)) + self.ymin)   
        return np.array(predictions)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if (not torch.is_tensor(x)):
            x, y = self._preprocessor(x, y, training = False)
        numpy_y_pred = np.array(self.model(x).detach())
        numpy_y = (np.array(y))    
        normalised_y = ((numpy_y * (self.ymax.iloc[0] - self.ymin.iloc[0])) + self.ymin.iloc[0])
        normalised_y_pred = ((numpy_y_pred * (self.ymax.iloc[0] - self.ymin.iloc[0])) + self.ymin.iloc[0])
        score = metrics.mean_squared_error(normalised_y, normalised_y_pred)
        return np.sqrt(score)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

def save_regressor(trained_model):  
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")

def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

def RegressorHyperParameterSearch(train_set): 
    
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs hyper-parameter search for fine-tuning the regressor implemented 
    in the Regressor class.

    Searches every possible parameter specified including:
        1. Number of epochs 
        2. Number of hidden layers 
        3. Number of neurons 
        4. Optimizer
        5. Learning rate 
        6. batch size
        7. activation_function 
        8. loss_function

    Arguments:
        train_set {pd.DataFrame} -- dataset to train models
        test_set {pd.DataFrame} -- dataset to test models during training
        
    Returns:
        The function should return your optimised hyper-parameters. 
    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # start time of search
    start = time.time()
    # range of epochs to try
    epoch_range = [40]
    # range of layers try 
    layer_range = [5]
    # range of neurons to try
    neuron_range = [64]
    # optimizers to try
    optimizers = ['adam']
    # learning rates to try
    lrs = [0.001]
    # batch sizes to try
    batch_sizes = [16]
    # activation functions to try
    activation_functions = [nn.ReLU()]
    # loss functions to try
    loss_functions = [nn.MSELoss()]
    # get training and val set
    training_set, val_set = test_train_split(pd.concat([train_set[0], train_set[1]], axis=1), test_size=0.25)
    # score to compare to   
    best_score = 1e6
    # dictionary to store the top 3 best models 

    # keeps track of run time
    variations = (len(layer_range) * len(epoch_range) * len(neuron_range) * len(optimizers) * len(lrs) 
                  * len(batch_sizes) * len(activation_functions) 
                    * len(loss_functions))
    count = 1
    top_models = {}
    for epoch in epoch_range:
        for layers in layer_range:
            for neurons in neuron_range:
                for optimizer in optimizers:
                    for lr in lrs:
                        for activation_fn in activation_functions:
                            for loss_fn in  loss_functions:
                                for batch_size in batch_sizes:
                                    # build Regressor with parameters 
                                    model = Regressor(training_set[0], val_set, batch_size, epoch, layers, neurons, optimizer, 
                                                    lr, activation_fn, loss_fn)
                                    # train
                                    model.fit(training_set[0], training_set[1])
                                    # compute score 
                                    score = model.score(val_set[0], val_set[1])
                                    # get sorted scores
                                    sorted_scores = sorted(list(top_models.keys()))
                                    # update best models if less than 5 models are added or the 
                                    # new score is less than the current highest score
                                    if len(sorted_scores) < 3 or score < sorted_scores[-1]:
                                            if len(sorted_scores) > 3:
                                                # remove the largest score
                                                top_models.pop(sorted_scores[-1])
                                            # update with the new score and parameters
                                            parameters = {}
                                            parameters['epoch'] = epoch
                                            parameters['layers'] = layers
                                            parameters['neurons'] = neurons
                                            parameters['optimizer'] = optimizer
                                            parameters['lr'] = lr
                                            parameters['activation_fn'] = activation_fn
                                            parameters['batch_size'] = batch_size
                                            top_models[score] = [model.train_errors, model.test_errors, parameters]
                                    # if score is improvement update
                                    print(score)
                                    if score < best_score:
                                        # update best model
                                        best_model = model
                                        # update best score
                                        best_score = score
                                    print(f"{count}/{variations} completed!")
                                    count+=1

    end = time.time()
    print("Run time = ", end-start)
    
    return best_model, top_models

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def plot_errors(train_errors, test_errors, parameters, title="Loss curve", xlabel="epochs"):
        """
        Function to plot the change in score while training 

        Arguments:
            - data {array} -- array of score to plot
        
        """
        for train, test, param in zip(train_errors, test_errors, parameters):
            epochs = range(len(train))
            plt.plot(epochs, train, label=f"TRAIN: {param}")
            plt.plot(epochs, test, label=f"TEST: {param}")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend(loc='upper center')
        plt.show()

def test_train_split(data, output_label = "median_house_value", test_size=0.2):
    """
    Splits the data into a test, train, validation split 
    """
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)

def example_main():
    # prepares data
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv") 
    train_set, test_set = test_train_split(data, test_size=0.2)
    run_hyperparameter_search(train_set, test_set, True)

   # best_model = load_regressor()
   # print(best_model.model)
    

def run_hyperparameter_search(train_data, test_data, save_model=False):
    """
    Runs hyperparameter search specified in RegressorHyperParameterSearch.

    Args:
        train_data (pd.DataFrame, pd.DataFrame) -- x,y set for training
        test_data (pd.DataFrame, pd.DataFrame) -- x,y set for testing
        save_model (bool) -- if model will be saved
    """
    # best_model is best Regressor, top_models is top 3 models
    best_model, top_models = RegressorHyperParameterSearch(train_data)
    # extract the data
    train_errors, test_errors, parameters, val_scores = extract_data(top_models)
    print("BEST MODEl: ", parameters[0])
    print("BEST VAL SCORE: ", val_scores[0])
    # plot the errors 
    plot_errors(train_errors, test_errors, parameters, title="Loss Curves for the Top 3 Models from 0 epochs")
    cut_train_errors = [errors[3:] for errors in train_errors]
    cut_test_errors = [errors[3:] for errors in test_errors]
    plot_errors(cut_train_errors, cut_test_errors, parameters, title="Loss Curves for the Top 3 Models from 0 epochs")
    if(save_model):
        save_regressor(best_model)
    return train_errors, test_errors, parameters, val_scores

def extract_data(models):
    """
    Helper function to extract the train errors,
    test errors and parameters of the best
    models from RegressorHyperParameter search.

    Args: models {dict}: dictionary returned from RHP

    Returns:
        {list} - list of list of train errors 
        {list} - list of list of test errors
        {list} - list of dict of model parameters
        {list} - list of the final val scores
    
    """

    sorted_errors = list(sorted(models.keys()))
    train_errors = []
    test_errors = []
    parameters = []
    val_scores = []
    for error in sorted_errors:
          train_errors.append(models[error][0])
          test_errors.append(models[error][1])
          parameters.append(models[error][2])
          val_scores.append(error)

    return train_errors, test_errors, parameters, val_scores
    


def default_model(train_set, test_set): 
    regressor = Regressor(train_set[0], nb_epoch = 10)
    regressor.fit(train_set[0], train_set[1])
    error = regressor.score(test_set[0], test_set[1])
    save_regressor(regressor)
    print("Default model error score: {}\n".format(error))

if __name__ == "__main__":
    example_main()

