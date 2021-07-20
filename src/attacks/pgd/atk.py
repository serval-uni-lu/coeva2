from art.attacks.evasion import ProjectedGradientDescentTensorFlowV2
from typing import Optional, Union, TYPE_CHECKING
import numpy as np

class PGDTF2(ProjectedGradientDescentTensorFlowV2):

    def __init__(
            self,
            estimator: "TensorFlowV2Classifier",
            norm: Union[int, float, str] = np.inf,
            eps: Union[int, float, np.ndarray] = 0.3,
            eps_step: Union[int, float, np.ndarray] = 0.1,
            max_iter: int = 100,
            targeted: bool = False,
            num_random_init: int = 0,
            batch_size: int = 32,
            random_eps: bool = False,
            tensor_board: Union[str, bool] = False,
            verbose: bool = True,
    ):
        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            tensor_board=tensor_board,
            verbose=verbose,
        )


    def _compute_perturbation(  # pylint: disable=W0221
        self, x: "tf.Tensor", y: "tf.Tensor", mask: Optional["tf.Tensor"]
    ) -> "tf.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :return: Perturbations.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad: tf.Tensor = self.estimator.loss_gradient(x, y, iter_i=self._i_max_iter, batch_id=self._batch_id)

        # Write summary
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "gradients/norm-L1/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=1),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-L2/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=2),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-Linf/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=np.inf),
                global_step=self._i_max_iter,
            )

            if hasattr(self.estimator, "compute_losses"):
                losses = self.estimator.compute_losses(x=x, y=y)

                for key, value in losses.items():
                    self.summary_writer.add_scalar(
                        "loss/{}/batch-{}".format(key, self._batch_id),
                        np.mean(value),
                        global_step=self._i_max_iter,
                    )

        # Check for NaN before normalisation an replace with 0
        if tf.reduce_any(tf.math.is_nan(grad)):
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)

        # Apply mask
        if mask is not None:
            grad = tf.where(mask == 0.0, 0.0, grad)

        # Apply norm bound
        if self.norm == np.inf:
            grad = tf.sign(grad)

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = tf.divide(grad, (tf.math.reduce_sum(tf.abs(grad), axis=ind, keepdims=True) + tol))

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = tf.divide(
                grad, (tf.math.sqrt(tf.math.reduce_sum(tf.math.square(grad), axis=ind, keepdims=True)) + tol)
            )

        assert x.shape == grad.shape

        return grad
