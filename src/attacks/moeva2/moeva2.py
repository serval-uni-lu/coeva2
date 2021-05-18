from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.constraints import Constraints
from src.attacks.moeva2.default_problem import DefaultProblem
from src.attacks.moeva2.feature_encoder import get_encoder_from_constraints

from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import os

from src.attacks.moeva2.result_process import HistoryResult, EfficientResult

from pymoo.factory import get_termination, get_mutation, get_crossover, get_sampling
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling,
    MixedVariableCrossover,
    MixedVariableMutation,
)

import warnings


class Coeva2:
    def __init__(
        self,
        classifier: Classifier,
        constraints: Constraints,
        ml_scaler=None,
        problem_class=None,
        n_gen=625,
        n_pop=640,
        n_offsprings=320,
        scale_objectives=True,
        save_history=False,
        n_jobs=-1,
        verbose=1,
    ) -> None:

        self._classifier = classifier
        self._constraints = constraints
        self._ml_scaler = ml_scaler
        self._problem_class = problem_class
        self._n_gen = n_gen
        self._n_pop = n_pop
        self._n_offsprings = n_offsprings
        self._scale_objectives = scale_objectives
        self._save_history = save_history
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._encoder = get_encoder_from_constraints(self._constraints)
        self._alg_class = NSGA2

        if problem_class is None:
            self._problem_class = DefaultProblem

    def _check_input_size(self, x: np.ndarray) -> None:
        if x.shape[1] != self._encoder.mutable_mask.shape[0]:
            raise ValueError(
                f"Mutable mask has shape (n_features,): {self._encoder.mutable_mask.shape[0]}, x has shaper (n_sample, "
                f"n_features): {x.shape}. n_features must be equal."
            )

    def _create_algorithm(self) -> GeneticAlgorithm:

        type_mask = self._encoder.get_type_mask_genetic()
        sampling = MixedVariableSampling(
            type_mask,
            {"real": get_sampling("real_random"), "int": get_sampling("int_random")},
        )

        # Default parameters for crossover (prob=0.9, eta=30)
        crossover = MixedVariableCrossover(
            type_mask,
            {
                "real": get_crossover("real_sbx", prob=0.9, eta=30),
                "int": get_crossover("int_sbx", prob=0.9, eta=30),
            },
        )

        # Default parameters for mutation (eta=20)
        mutation = MixedVariableMutation(
            type_mask,
            {
                "real": get_mutation("real_pm", eta=20),
                "int": get_mutation("int_pm", eta=20),
            },
        )

        algorithm = self._alg_class(
            pop_size=self._n_pop,
            n_offsprings=self._n_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
            return_least_infeasible=True,
        )

        return algorithm

    def _one_generate(self, x, minimize_class: int):
        # Reduce log
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)

        termination = get_termination("n_gen", self._n_gen)
        classifier = deepcopy(self._classifier)

        # self._encoder = get_encoder_from_constraints(con)
        algorithm = self._create_algorithm()
        constraints = deepcopy(self._constraints)
        encoder = get_encoder_from_constraints(self._constraints, x)

        problem = self._problem_class(
            x_initial_state=x,
            classifier=classifier,
            minimize_class=minimize_class,
            encoder=encoder,
            constraints=constraints,
            scale_objectives=self._scale_objectives,
            save_history=self._save_history,
            ml_scaler=self._ml_scaler,
        )

        result = minimize(
            problem,
            algorithm,
            termination,
            verbose=0,
            save_history=False,  # Implemented from library should always be False
        )

        if self._save_history:
            return HistoryResult(result)
        else:
            result = EfficientResult(result)
            return result

    # Loop over inputs to generate adversarials using the _one_generate function above
    def generate(self, x: np.ndarray, minimize_class):

        if isinstance(minimize_class, int):
            minimize_class = np.repeat(minimize_class, x.shape[0])

        if x.shape[0] != minimize_class.shape[0]:
            raise ValueError(
                f"minimize_class argument must be an integer or an array of shaper (x.shape[0])"
            )

        self._check_input_size(x)

        if len(x.shape) != 2:
            raise ValueError(f"{x.__name__} ({x.shape}) must have 2 dimensions.")

        iterable = enumerate(x)
        if self._verbose > 0:
            iterable = tqdm(iterable, total=len(x))

        # Sequential Run
        if self._n_jobs == 1:
            processed_result = [
                self._one_generate(initial_state, minimize_class[index]) for index, initial_state in iterable
            ]

        # Parallel run
        else:
            processed_result = Parallel(n_jobs=self._n_jobs)(
                delayed(self._one_generate)(initial_state)
                for index, initial_state in iterable
            )

        return processed_result
