from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy

import numpy as np

from pymoo.factory import get_termination, get_mutation, get_crossover, get_sampling
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.optimize import minimize
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling,
    MixedVariableCrossover,
    MixedVariableMutation,
)

from .classifier import Classifier
from .feature_encoder import get_encoder_from_constraints
from .problem import Coeva2Problem
from .constraints import Constraints
from .result_process import HistoryResult, EfficientResult


class Coeva2:
    def __init__(
        self,
        classifier: Classifier,
        constraints: Constraints,
        alg="nsga2",
        weights=None,
        n_gen=625,
        n_pop=640,
        n_offsprings=320,
        scale_objectives=True,
        save_history=False,
        n_jobs=-1,
        verbose=1,
    ) -> None:

        if weights is None:
            weights = {"alpha": 2, "beta": 1, "gamma": 100, "delta": 1000}

        if alg == "nsga2":
            self._alg_class = NSGA2
        elif alg == "wff":
            self._alg_class = GA
        else:
            raise NotImplementedError
        self._alg = alg
        self._constraints = constraints
        self._encoder = None
        self._classifier = classifier
        self._weights = weights

        self._n_offsprings = n_offsprings
        self._n_pop = n_pop
        self._n_gen = n_gen
        self._scale_objectives = scale_objectives
        self._save_history = save_history
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._encoder = get_encoder_from_constraints(self._constraints)

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
            eliminate_duplicates=True,
            return_least_infeasible=True,
        )

        return algorithm

    def _one_generate(self, x):
        termination = get_termination("n_gen", self._n_gen)
        algorithm = self._create_algorithm()
        classifier = deepcopy(self._classifier)
        constraints = deepcopy(self._constraints)
        encoder = deepcopy(self._encoder)

        problem = Coeva2Problem(
            x_initial_state=x,
            classifier=classifier,
            encoder=encoder,
            constraints=constraints,
            alg=self._alg,
            weights=self._weights,
            scale_objectives=self._scale_objectives,
            save_history=self._save_history,
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
            return (result)

    def generate(self, x: np.ndarray):
        self._check_input_size(x)

        if len(x.shape) != 2:
            raise ValueError(f"{x.__name__} ({x.shape}) must have 2 dimensions.")

        iterable = enumerate(x)
        if self._verbose > 0:
            iterable = tqdm(iterable, total=len(x))

        processed_result = Parallel(n_jobs=self._n_jobs)(
            delayed(self._one_generate)(initial_state)
            for index, initial_state in iterable
        )

        return processed_result
