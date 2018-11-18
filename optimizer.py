from cma import CMAEvolutionStrategy
import numpy as np

class Optimizer():
    """
    This class implements an optimizer for the genetic algorithm

    Includes initialization, updating, scoring criteria, and more
    """

    def init_model(self, num_weights, popsize):
        # sigma and weights begin at 1
        self.model_opt = CMAEvolutionStrategy(num_weights * [-1], 1,
            {'popsize': popsize})

    def get_model_weights(self):
        # for all models
        self.model_weights = self.model_opt.ask()
        return self.model_weights

    def give_model_scores(self, scores):
        # for all models
        self.model_opt.tell(self.model_weights, scores)

    def check_model_stop(self):
        return self.model_opt.stop()

    def init_classifier(self, num_weights, popsize):
        self.class_opt = CMAEvolutionStrategy(num_weights * [0], 1,
            {'popsize': popsize})

    def get_classifier_weights(self):
        # for all classifiers
        self.class_weights = self.class_opt.ask()
        return self.class_weights

    def give_classifier_scores(self, scores):
        self.class_opt.tell(self.class_weights, scores)

    def check_classifier_stop(self):
        return self.class_opt.stop()
