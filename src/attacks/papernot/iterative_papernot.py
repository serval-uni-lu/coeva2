import logging
import random
from copy import deepcopy

import numpy as np
from art.attacks.evasion import DecisionTreeAttack
from art.classifiers import SklearnClassifier
from art.utils import projection, check_and_transform_label_format
from joblib import Parallel, delayed
from sklearn import metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)


def cut_in_batch(arr, n_desired_batch=1, batch_size=None):

    if batch_size is None:
        n_batch = max(n_desired_batch, len(arr))
    else:
        n_batch = np.ceil(len(arr / batch_size))

    batches_i = np.array_split(np.arange(arr.shape[0]), n_batch)

    return [arr[batch_i] for batch_i in batches_i]


class IterativeDecisionTreeAttack(DecisionTreeAttack):
    random_search = False

    def __init__(self, classifier, eps_step, eps, norm_p=2, random_search=False):
        """
        :param classifier: A trained model of type scikit decision tree.
        :type classifier: :class:`.Classifier.ScikitlearnDecisionTreeClassifier`
        :param offset: How much the value is pushed away from tree's threshold. default 0.001
        :type classifier: :float:
        """
        self.eps = eps
        self.offset = eps_step
        self.eps_step = eps_step
        self.project = True
        self.norm_p = norm_p
        super(IterativeDecisionTreeAttack, self).__init__(classifier, eps_step)
        self.classifier = classifier
        params = {"eps": eps, "norm_p": norm_p, "random_search": random_search}
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
        if self.classifier.get_left_child(position) == self.classifier.get_right_child(
            position
        ):
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
                res = self._df_subtree(
                    self.classifier.get_left_child(position), original_class, target
                )
            else:
                res = self._df_subtree(
                    self.classifier.get_right_child(position), original_class, target
                )
            if res[0] == -1:
                # no result, try right subtree
                if r <= 0.5:
                    res = self._df_subtree(
                        self.classifier.get_left_child(position), original_class, target
                    )
                else:
                    res = self._df_subtree(
                        self.classifier.get_right_child(position),
                        original_class,
                        target,
                    )

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
        eps_step = self.eps_step
        for i in range(1, 1 + len(adv_path[1:])):
            go_for = adv_path[i - 1]
            threshold = self.classifier.get_threshold_at_node(adv_path[i])
            feature = self.classifier.get_feature_at_node(adv_path[i])
            # only perturb if the feature is actually wrong
            if x[index][
                feature
            ] > threshold and go_for == self.classifier.get_left_child(adv_path[i]):
                x[index][feature] = threshold - eps_step
            elif x[index][
                feature
            ] <= threshold and go_for == self.classifier.get_right_child(adv_path[i]):
                x[index][feature] = threshold + eps_step

        return x

    def generate(self, x0, y=None, y0=None, x_adv=None, **kwargs):
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
        y = check_and_transform_label_format(
            y, self.classifier.nb_classes, return_one_hot=False
        )
        # print(x0.shape)
        if x_adv is None:
            x = np.copy(x0)
        else:
            x = np.copy(x_adv)

        for index in range(np.shape(x)[0]):
            path = self.classifier.get_decision_path(x[index])
            # print("path: {}".format(path))
            if y0 is None:
                legitimate_class = np.argmax(
                    self.classifier.predict(x[index].reshape(1, -1))
                )
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
                        adv_path = self._df_subtree(
                            self.classifier.get_right_child(ancestor), legitimate_class
                        )
                    else:
                        adv_path = self._df_subtree(
                            self.classifier.get_right_child(ancestor),
                            legitimate_class,
                            y[index],
                        )
                else:  # search in left subtree
                    if y is None:
                        adv_path = self._df_subtree(
                            self.classifier.get_left_child(ancestor), legitimate_class
                        )
                    else:
                        adv_path = self._df_subtree(
                            self.classifier.get_left_child(ancestor),
                            legitimate_class,
                            y[index],
                        )
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

    def __init__(
        self,
        cls,
        tree_attack=IterativeDecisionTreeAttack,
        threshold=0.5,
        nb_estimators=10,
        nb_iterations=10,
        n_jobs=1,
        **kwargs
    ):
        super().__init__()
        self.classifier = cls
        self.threshold = threshold
        self.attack = tree_attack
        self.attack_args = kwargs
        self.nb_estimators = nb_estimators
        self.nb_iterations = nb_iterations
        self.n_jobs = n_jobs

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

    def generate(self, x, y, index=0, **kwargs):

        self.classifier = deepcopy(self.classifier)
        x0 = x
        y0 = y

        rf_success_rate = 0
        x = np.copy(x0)
        rf_success_x = x
        iterate = range(self.nb_estimators)
        if index == 0:
            iterate = tqdm(iterate, total=self.nb_estimators)
        for e in iterate:
            logging.info(
                "Attacking tree {}. Prev success rate {}".format(e, rf_success_rate)
            )
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
                logging.info(
                    "Attacking iteration {}. Prev success rate {}".format(
                        i, tree_success_x
                    )
                )
                est_x = bounded.generate(x0, y0=y0, x_adv=est_x)
                score = self.score_performance(x0, est_x, y0)
                if score > tree_success_rate:
                    tree_success_rate = score
                    tree_success_x = est_x

            if tree_success_rate > rf_success_rate:
                rf_success_rate = tree_success_rate
                rf_success_x = tree_success_x

            l_2, l_inf = self.l_metrics(rf_success_x, x0)

        return rf_success_x

    def generate_parallel(self, x, y):
        if self.n_jobs == 1:
            self.generate(x, y)
        else:
            out = Parallel(n_jobs=self.n_jobs)(
                delayed(self.generate)(x[batch_indexes], y[batch_indexes], i)
                for i, batch_indexes in enumerate(
                    cut_in_batch(np.arange(len(x)), n_desired_batch=self.n_jobs)
                )
            )

            return np.concatenate(out)
