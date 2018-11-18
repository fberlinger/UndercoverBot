import math
import numpy as np

class Fitness():
    """This class analyzes the fitness of both models and classifiers
    """

    def __init__(
        self,
        trial_data
    ):
        """
        input: trial data.
        A list of matrices. Each matrix has dimension num_models x num_fish

        The ith matrix contains classification data from the ith classifier
        The last column of the jth row of each matrix represents classification of
        the jth model
        """
        self.trial_data = trial_data

    def score_classifiers(self):
        score = 0
        classifier_scores = []
        for classifier_data in self.trial_data:
            specificity = np.mean(classifier_data[:1,:])
            sensitivity = 1 - np.mean(classifier_data[1:,:])
            classifier_scores.append(0.5 * (sensitivity + specificity))

        return classifier_scores

    def score_models(self):
        models = [data[1:, :] for data in self.trial_data]
        model_arr = np.stack(models, axis = 1)
        model_scores = np.mean(model_arr, axis = (1,2))
        return model_scores.flatten().tolist()

