import math
import numpy as np

class Classifier():
    """This class creates a discriminator to classify real and false agents
    """

    def __init__(
        self,
        id,
        weights
    ):

        self.id = id
        # Elman NN with 2 input neurons, 1 hidden layer with 5 neurons, and 1 outputs
        # The input is the fish's number of neihgbors within 2 radii
        # Distance to consider a neigbor is a learned parameter.
        # the output is classification as agent or replica.
        # agent >0.5, return   1, replica <0.5, return 0
        # Both hidden and output layer have bias
        # activation function is the logistic sigmoid

        self.input_to_hidden = np.reshape(weights[:10], (2, 5))
        self.bias_hidden = np.reshape(weights[10:15], (1, 5))
        self.hidden_to_hidden = np.reshape(weights[15:40], (5, 5))
        self.hidden_to_output = np.reshape(weights[40:45], (5, 1))
        self.bias_output = np.reshape(weights[45:46], (1, 1))

        self.radius1 = weights[46]
        self.radius2 = weights[47]
    def logistic_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def run_network(self, input_sequence):
        # input sequence is 2+num_fish x t, where t is time.
        # first convert to num_neighbors
        count_rad1 = (input_sequence[:, 4:] > self.radius1).sum(axis = 1)
        count_rad2 = (input_sequence[:, 4:] > self.radius2).sum(axis = 1)
        input_sequence[:, 2] = count_rad1
        input_sequence[:, 3] = count_rad2

        # run first input through, then hold hidden layer
        hidden = None
        first_input = True

        for row in range(len(input_sequence)):
            if first_input:
                hidden = np.dot(input_sequence[row,2:4], self.input_to_hidden) + \
                         self.bias_hidden
                first_input = False
            else:
                hidden = np.dot(input_sequence[row,2:4], self.input_to_hidden) + \
                         self.bias_hidden + \
                         np.dot(hidden, self.hidden_to_hidden)
            hidden = self.logistic_sigmoid(hidden)


        # run output layer - only need to do this with final input, as we discard previous outputs
        output = np.dot(hidden, self.hidden_to_output) + self.bias_output

        output = self.logistic_sigmoid(output)
        if output < 0.5:
            return 0
        else:
            return 1

    def classify_models(self, models):
        """ given a matrix of num_fish x t x 2 matrix, classify all models
            input: num_fish x t x 2 matrix

            output: 1 x num_fish matrix, with each value classifying model/non model
        """
        outputs = [self.run_network(fish) for fish in models]
        return np.reshape(outputs, (len(outputs)))

    def classify_all_models(self, all_sim_data):
        """
        Given a matrix of model population size x numfish/trial x time steps/trial x 2
        Outout a matrix population size x number fish per model
        Last col in matrix is the fish models (rather than idea agents)

        """

        outputs = [self.classify_models(trial) for trial in all_sim_data]
        return np.stack(outputs, axis = 0)
