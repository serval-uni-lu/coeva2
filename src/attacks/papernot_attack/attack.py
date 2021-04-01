from art.attacks import DecisionTreeAttack
from art.utils import projection, check_and_transform_label_format
import random
import numpy as np
from art.classifiers import SklearnClassifier

from sklearn import metrics

import logging

logger = logging.getLogger(__name__)


class IterativeDecisionTreeAttack(DecisionTreeAttack):
    project = False
    random_search = False
    eps = 0.01
    norm_p = 2

    def __init__(self, classifier, offset=0.001, eps=0.1, norm_p=2, random_search=False):
        """
        :param classifier: A trained model of type scikit decision tree.
        :type classifier: :class:`.Classifier.ScikitlearnDecisionTreeClassifier`
        :param offset: How much the value is pushed away from tree's threshold. default 0.001
        :type classifier: :float:
        """
        super(IterativeDecisionTreeAttack, self).__init__(classifier, offset)

        params = {'eps': eps, 'norm_p': norm_p, 'random_search': random_search}
        self.set_params(**params)

    def _df_subtree(self, position, original_class, target=None):
        """
        Search a decision tree for a mis-classifying instance.

        :param position: An array with the original inputs to be attacked.
        :type position: `int`
        :param original_class: original label for the instances we are searching mis-classification for.
        :type original_class: `int` or `np.ndarray`
        :param target: If the provided, specifies which output the leaf has to have to be accepted.
        :type target: `int`
        :return: An array specifying the path to the leaf where the classification is either != original class or
                 ==target class if provided.
        :rtype: `list`
        """
        # base case, we're at a leaf
        if self.classifier.get_left_child(position) == self.classifier.get_right_child(position):
            if target is None:  # untargeted case
                if self.classifier.get_classes_at_node(position) != original_class:
                    path = [position]
                else:
                    path = [-1]
            else:  # targeted case
                if self.classifier.get_classes_at_node(position) == target:
                    path = [position]
                else:
                    path = [-1]
        else:  # go deeper, depths first

            if self.random_search:
                r = random.random()
            else:
                r = 0
            if r > 0.5:
                res = self._df_subtree(self.classifier.get_left_child(
                    position), original_class, target)
            else:
                res = self._df_subtree(self.classifier.get_right_child(
                    position), original_class, target)
            if res[0] == -1:
                # no result, try right subtree
                if r <= 0.5:
                    res = self._df_subtree(self.classifier.get_left_child(
                        position), original_class, target)
                else:
                    res = self._df_subtree(self.classifier.get_right_child(
                        position), original_class, target)

                if res[0] == -1:
                    # no desired result
                    path = [-1]
                else:
                    res.append(position)
                    path = res
            else:
                # done, it is returning a path
                res.append(position)
                path = res

        return path

    def perturbate(self, x, index, adv_path):
        eps = self.offset
        for i in range(1, 1 + len(adv_path[1:])):
            go_for = adv_path[i - 1]
            threshold = self.classifier.get_threshold_at_node(adv_path[i])
            feature = self.classifier.get_feature_at_node(adv_path[i])
            # only perturb if the feature is actually wrong
            if x[index][feature] > threshold and go_for == self.classifier.get_left_child(adv_path[i]):
                x[index][feature] = threshold - eps
            elif x[index][feature] <= threshold and go_for == self.classifier.get_right_child(adv_path[i]):
                x[index][feature] = threshold + eps

        return x

    def generate(self, x0, y=None, y0=None, **kwargs):
        """
        Generate adversarial examples and return them as an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.classifier.nb_classes(), return_one_hot=False)
        # print(x0.shape)
        x = np.copy(x0)

        for index in range(np.shape(x)[0]):
            path = self.classifier.get_decision_path(x[index])
            # print("path: {}".format(path))
            if y0 is None:
                legitimate_class = np.argmax(self.classifier.predict(x[index].reshape(1, -1)))
            else:
                legitimate_class = y0[index]
            position = -2
            adv_path = [-1]
            ancestor = path[position]
            while np.abs(position) < (len(path) - 1) or adv_path[0] == -1:
                ancestor = path[position]
                current_child = path[position + 1]
                # search in right subtree
                if current_child == self.classifier.get_left_child(ancestor):
                    if y is None:
                        adv_path = self._df_subtree(self.classifier.get_right_child(ancestor), legitimate_class)
                    else:
                        adv_path = self._df_subtree(self.classifier.get_right_child(ancestor), legitimate_class,
                                                    y[index])
                else:  # search in left subtree
                    if y is None:
                        adv_path = self._df_subtree(
                            self.classifier.get_left_child(ancestor), legitimate_class)
                    else:
                        adv_path = self._df_subtree(self.classifier.get_left_child(ancestor), legitimate_class,
                                                    y[index])
                position = position - 1  # we are going the decision path upwards
                # print("going upward")
            adv_path.append(ancestor)
            # we figured out which is the way to the target, now perturb
            # first one is leaf-> no threshold, cannot be perturbed
            # print("adv_path: {}".format(adv_path))

            x = self.perturbate(x, index, adv_path)

        if self.project:
            # print("project")
            return x0 + projection(x - x0, self.eps, self.norm_p)
        return x


class RFAttack(object):
    classifier = None
    attack = None

    def __init__(self, cls, tree_attack=IterativeDecisionTreeAttack, threshold=0.5, nb_estimators=10, nb_iterations=10,
                 **kwargs):
        super().__init__()
        self.classifier = cls
        self.threshold = threshold
        self.attack = tree_attack
        self.attack_args = kwargs
        self.nb_estimators = nb_estimators
        self.nb_iterations = nb_iterations

    def l_metrics(self, X, X_adv):
        l_2 = np.linalg.norm(X_adv - X, axis=1).mean()
        l_inf = np.linalg.norm(X_adv - X, axis=1, ord=np.inf).mean()
        return l_2, l_inf

    def c_metrics(self, y_true, y_proba):
        y_pred = y_proba[:, 1] >= self.threshold
        acc = metrics.accuracy_score(y_true, y_pred)
        return None, acc, None, None

    def score_performance(self, x0, x, y0):

        y = self.classifier.predict_proba(x)
        _, accuracy, _, _ = self.c_metrics(y0, y)

        return 1 - accuracy

    def generate(self, x0, y0, **kwargs):

        rf_success_rate = 0
        x = np.copy(x0)
        rf_success_x = x

        for e in range(self.nb_estimators):
            logging.info("Attacking tree {}. Prev success rate {}".format(e, rf_success_rate))
            est_x = rf_success_x
            tree = self.classifier.estimators_[e]
            # Create ART classifier for scikit-learn Descision tree
            art_classifier = SklearnClassifier(model=tree)

            # Create ART Zeroth Order Optimization attack
            bounded = self.attack(classifier=art_classifier, **self.attack_args)
            tree_success_rate = rf_success_rate
            tree_success_x = None

            for i in range(self.nb_iterations):
                # print(tree_success_rate)
                logging.info("Attacking iteration {}. Prev success rate {}".format(i, tree_success_x))
                est_x = bounded.generate(est_x, y0=y0)
                score = self.score_performance(x0, est_x, y0)
                if score > tree_success_rate:
                    tree_success_rate = score
                    tree_success_x = est_x

            if tree_success_rate > rf_success_rate:
                rf_success_rate = tree_success_rate
                rf_success_x = tree_success_x

            l_2, l_inf = self.l_metrics(rf_success_x, x0)

        return rf_success_x, rf_success_rate, l_2, l_inf



