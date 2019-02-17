import math
import numpy as np

class Fitness():
    """This class analyzes the fitness of both models and classifiers
    """

    def __init__(
        self,
        trial_data,
        truth_list
    ):
        """
        Class to calculate fitness of models and classifiers from classification output

        Argument: 
            trial data {np.matrix} - Dimension num_classifiers x (ideal agents + models)
                Contains the classification of each swarm by each classifier
            truth_list {np.matrix} - 1 x Dimension num_classifiers x (ideal agents + models)
                Contains the true classification of each swarm
                
        """
        self.trial_data = trial_data
        self.truth_list = truth_list


    def score_classifiers(self):
        """
        Give classifiers a fitness score

        Classifiers are scored on specificity and sensitivity. These are
        averaged so that guessing all replica or all real gives 0.5 fitness
        (ie classification is not biased by number of imposter vs. real
        data samples)

        Returns:
            float|list -- fitness of each classifier
        """
        score = 0

        classifier_scores = []
        for classifier_data in self.trial_data:
            specificity = np.mean(classifier_data[self.truth_list == 1])
            sensitivity = 1 - np.mean(classifier_data[self.truth_list == 0])
            classifier_scores.append(0.5 * (sensitivity + specificity))

        return [score * -1 for score in classifier_scores]

    def score_models(self):
        """
        Give models a fitness score.

        Each model is judged on the fraction of classifiers it
        tricked into believing it is real.

        Returns:
            float|list -- fitness of each model
        """
        models = self.trial_data[:, self.truth_list == 0]
        model_scores = np.mean(models, axis = 0) * (-1)
        return model_scores.flatten().tolist()

