from .venus_constraints import evaluate
import numpy as np


def objectives_per_input(result, encoder, threshold, model):
    Xs = np.array(list(map(lambda x: x.X, result.pop))).astype(np.float64)
    Xs_ml = encoder.from_genetic_to_ml(result.initial_state, Xs).astype(np.float64)
    respectsConstraints = (
        encoder.constraint_scaler.transform(evaluate(Xs_ml)).mean(axis=1) <= 0
    ).astype(np.int64)
    isMisclassified = np.array(model.predict_proba(Xs_ml)[:, 1] < threshold).astype(
        np.int64
    )
    isBigAmount = (Xs[:, 0] >= 10000).astype(np.int64)

    o3 = respectsConstraints * isMisclassified
    o4 = o3 * isBigAmount
    objectives = np.array([respectsConstraints, isMisclassified, o3, o4])

    return objectives


def calculate_objectives(result, encoder, threshold, model):

    objectives = objectives_per_input(result, encoder, threshold, model)
    objectives = objectives.sum(axis=1)
    objectives = (objectives > 0).astype(np.int64)

    return objectives


def calculate_success_rates(results, encoder, threshold, model):

    objectives = np.array(
        [calculate_objectives(result, encoder, threshold, model) for result in results]
    )
    success_rates = np.apply_along_axis(lambda x: x.sum() / x.shape[0], 0, objectives)
    return success_rates


def adversarial_training(results, encoder, threshold, model):
    training = []

    for result in results:
        adv_filter = objectives_per_input(result, encoder, threshold, model)[3].astype(
            np.bool
        )
        Xs = np.array(list(map(lambda x: x.X, result.pop))).astype(np.float64)
        Xs_ml = encoder.from_genetic_to_ml(result.initial_state, Xs).astype(np.float64)
        training.append(Xs_ml[adv_filter])

    return np.concatenate(training)
