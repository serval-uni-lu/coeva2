import numpy as np
from art.utils import projection
from pymoo.model.sampling import Sampling
from pymoo.util.normalization import denormalize, normalize


class FloatRandomSamplingL2(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=float, l2_max=0.1, type_mask=None) -> None:
        super().__init__()
        self.var_type = var_type
        self.l2_max = l2_max
        self.type_mask = type_mask

    def _do(self, problem, n_samples, **kwargs):
        x_initial_f = problem.x_initial_ml
        x_initial_gen = problem.encoder.ml_to_genetic(x_initial_f.reshape(1, -1))[0]
        x_initial_gen = normalize(x_initial_gen, problem.xl, problem.xu)

        x_perturbation = np.random.random((n_samples, problem.n_var))
        x_perturbation = denormalize(
            x_perturbation, -np.ones(problem.n_var), np.ones(problem.n_var)
        )
        x_perturbation = projection(x_perturbation, self.l2_max, 2)

        x_perturbed = x_initial_gen + x_perturbation
        x_perturbed = np.clip(x_perturbed, 0.0, 1.0)
        x_perturbed = denormalize(x_perturbed, problem.xl, problem.xu)

        # Apply int
        mask_int = self.type_mask != "real"
        x_perturbed[:, mask_int] = np.rint(x_perturbed[:, mask_int])

        return x_perturbed
