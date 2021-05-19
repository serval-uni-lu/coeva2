import copy

from src.attacks.moeva2.default_problem import DefaultProblem


AVOID_ZERO = 0.00000001


class LcldProblem(DefaultProblem):

    def _evaluate_additional_objectives(self, x, x_f, x_f_mm, x_ml):

        # Maximize the amount
        amount_feature_index = 0
        amount = copy.deepcopy(x_ml[:, amount_feature_index])
        amount[amount <= 0] = AVOID_ZERO
        f3 = 1 / amount

        return [f3]
