import os
import warnings
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.rnsga3 import RNSGA3
from pymoo.factory import (
    get_termination,
    get_mutation,
    get_crossover,
    get_reference_directions,
    get_sampling,
)
from pymoo.operators.mixed_variable_operator import (
    MixedVariableCrossover,
    MixedVariableMutation,
    MixedVariableSampling,
)
from pymoo.optimize import minimize
from tqdm import tqdm

from .classifier import Classifier
from .constraints import Constraints
from .adversarial_problem import AdversarialProblem
from .feature_encoder import get_encoder_from_constraints
from .sampling import MixedSamplingLp, InitialStateSampling
from .result_process import HistoryResult, EfficientResult
from .softmax_crossover import SoftmaxPointCrossover
from .softmax_mutation import SoftmaxPolynomialMutation
from ...utils.in_out import load_model


def tf_lof_off():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)


class Moeva2:
    def __init__(
        self,
        classifier_class,
        constraints: Constraints,
        norm=None,
        fun_distance_preprocess=lambda x: x,
        n_gen=100,
        n_pop=640,
        n_offsprings=320,
        save_history="none",
        seed=None,
        n_jobs=-1,
        verbose=1,
    ) -> None:

        self.classifier_class = classifier_class
        self.constraints = constraints
        self.norm = norm
        self.fun_distance_preprocess = fun_distance_preprocess
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.n_offsprings = n_offsprings

        self.save_history = save_history
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Defaults
        self.alg_class = RNSGA3
        self.problem_class = AdversarialProblem

        # Computed
        self.encoder = get_encoder_from_constraints(self.constraints)

    def _check_inputs(self, x: np.ndarray, y) -> None:
        if x.shape[1] != self.encoder.mutable_mask.shape[0]:
            raise ValueError(
                f"Mutable mask has shape (n_features,): {self.encoder.mutable_mask.shape[0]}, x has shaper (n_sample, "
                f"n_features): {x.shape}. n_features must be equal."
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"minimize_class argument must be an integer or an array of shaper (x.shape[0])"
            )

        if len(x.shape) != 2:
            raise ValueError(f"{x.__name__} ({x.shape}) must have 2 dimensions.")

    def _create_algorithm(self, n_obj) -> GeneticAlgorithm:

        type_mask = self.encoder.get_type_mask_genetic()

        sampling = InitialStateSampling(type_mask=type_mask)

        # Default parameters for crossover (prob=0.9, eta=30)
        modify_mask = type_mask.copy()
        # modify_mask[-256:] = ["softmax"] * 256
        crossover = MixedVariableCrossover(
            modify_mask,
            {
                "real": get_crossover(
                    "real_two_point",
                ),
                "int": get_crossover(
                    "int_two_point",
                ),
                "softmax": SoftmaxPointCrossover(n_points=2),
            },
        )

        # Default parameters for mutation (eta=20)
        mutation = MixedVariableMutation(
            modify_mask,
            {
                "real": get_mutation("real_pm", eta=20),
                "int": get_mutation("int_pm", eta=20),
                "softmax": SoftmaxPolynomialMutation(eta=20),
            },
        )

        ref_points = get_reference_directions("energy", n_obj, self.n_pop, seed=1)

        algorithm = self.alg_class(
            pop_per_ref_point=1,
            ref_points=ref_points,
            n_offsprings=self.n_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
            return_least_infeasible=True,
        )

        return algorithm

    def _one_generate(self, x, y: int, classifier):
        # Reduce log
        termination = get_termination("n_gen", self.n_gen)

        constraints = deepcopy(self.constraints)
        encoder = get_encoder_from_constraints(self.constraints, x)

        problem = self.problem_class(
            x_clean=x,
            classifier=classifier,
            y_clean=y,
            encoder=encoder,
            constraints=constraints,
            fun_distance_preprocess=self.fun_distance_preprocess,
            norm=self.norm,
            save_history=self.save_history,
        )

        algorithm = self._create_algorithm(n_obj=problem.get_nb_objectives())

        result = minimize(
            problem,
            algorithm,
            termination,
            verbose=0,
            seed=self.seed,
            save_history=False,  # Implemented from library should always be False
        )

        x_adv = np.array([ind.X.astype(np.float64) for ind in result.pop])
        x_adv = self.encoder.genetic_to_ml(x_adv, x)
        history = result.problem.get_history()

        return x_adv, history

    def _batch_generate(self, x, y, batch_i):
        tf_lof_off()

        iterable = enumerate(x)
        if (self.verbose > 0) and (batch_i == 0):
            iterable = tqdm(iterable, total=len(x))

        classifier = self.classifier_class()

        out = [self._one_generate(x[i], y[i], classifier) for i, _ in iterable]

        out = zip(*out)
        out = [np.array(out_0) for out_0 in out]

        return out

    # Loop over inputs to generate adversarials using the _one_generate function above
    def generate(self, x: np.ndarray, y, batch_size=None):

        n_batch = self.n_jobs
        if batch_size is not None:
            n_batch = np.ceil(x.shape[0] / batch_size)

        batches_i = np.array_split(np.arange(x.shape[0]), n_batch)

        if isinstance(y, int):
            y = np.repeat(y, x.shape[0])

        self._check_inputs(x, y)

        iterable = enumerate(batches_i)
        # if self.verbose > 0:
        #     iterable = tqdm(iterable, total=len(x), position=1)

        # Sequential Run
        if self.n_jobs == 1:
            print("Sequential run.")
            out = [
                self._batch_generate(x[batch_indexes], y[batch_indexes], i)
                for i, batch_indexes in iterable
            ]

        # Parallel run
        else:
            print("Parallel run.")
            out = Parallel(n_jobs=self.n_jobs)(
                delayed(self._batch_generate)(x[batch_indexes], y[batch_indexes], i)
                for i, batch_indexes in iterable
            )

        out = zip(*out)
        out = [np.concatenate(out_0) for out_0 in out]

        x_adv = out[0]
        histories = out[1]

        if self.save_history:
            return x_adv, histories
        else:
            return x_adv
