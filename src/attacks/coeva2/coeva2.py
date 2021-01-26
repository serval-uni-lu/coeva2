from pymoo.factory import get_termination, get_mutation, get_crossover, get_sampling
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, MixedVariableMutation
from pymoo.optimize import minimize

from .constraints import Constraints
from .classifier import Classifier
import numpy as np
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm


class Coeva2:
    def __init__(
            self,
            constraints: Constraints,
            classifier: Classifier,
            mutable_mask: np.ndarray,
            type_mask: np.ndarray,
            alg="nsga2",
            weights=None,
            n_gen=625,
            n_pop=640,
            n_offsprings=320
    ) -> None:

        if weights is None:
            weights = {"alpha": 2, "beta": 1, "gamma": 100, "delta": 1000}
        if scaler.n_features_in_ != mutable_mask.shape[0]:
            raise ValueError(
                f"Scaler takes has {scaler.n_features_in_} features, but mutable mask"
                f"has {mutable_mask.shape[0]} features. Must be equal."
            )

        if alg == "nsga2":
            self._alg = NSGA2
        elif alg == "ga":
            self._alg = GA
        else:
            raise NotImplementedError

        self._type_mask = type_mask
        self._constraints = constraints
        self._scaler = scaler
        self._classifier = classifier
        self._mutable_mask = mutable_mask
        self._weights = weights
        self._n_features = mutable_mask.shape[0]

        self._n_offsprings = n_offsprings
        self._n_pop = n_pop
        self._n_gen = n_gen

        self._classifier.set_n_jobs(1)
        self._classifier.set_verbose(0)

    def _check_input_size(self, x: np.ndarray) -> None:
        if x.shape[1] != self._mutable_mask.shape[0]:
            raise ValueError(
                f"Mutable mask has shape (n_features,): {self._mutable_mask.shape[0]}, x has shaper (n_sample, "
                f"n_features): {x.shape}. n_features must be equal."
            )

    def _create_algorithm(self) -> GeneticAlgorithm:

        sampling = MixedVariableSampling(
            self._type_mask,
            {
                "real": get_sampling("real_random"),
                "int": get_sampling("int_random")
            },
        )

        # Default parameters for crossover (prob=0.9, eta=30)
        crossover = MixedVariableCrossover(
            self._type_mask,
            {
                "real": get_crossover("real_sbx", prob=0.9, eta=30),
                "int": get_crossover("int_sbx", prob=0.9, eta=30),
            },
        )

        # Default parameters for mutation (eta=20)
        mutation = MixedVariableMutation(
            self._type_mask,
            {
                "real": get_mutation("real_pm", eta=20),
                "int": get_mutation("int_pm", eta=20),
            },
        )

        algorithm = self._alg(
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

        result = minimize(problem, algorithm, termination, verbose=0, save_history=False, )

    def generate(self, x: np.ndarray):
        self._check_input_size(x)

        # For each x
        # Copy element
        # Create attack
        # Create attack
        # execute attack
