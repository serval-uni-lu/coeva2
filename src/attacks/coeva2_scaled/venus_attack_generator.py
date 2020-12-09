from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling,
    MixedVariableMutation,
    MixedVariableCrossover,
)
from .venus_coeva2_all_problem import VenusProblem


def create_attack(
    initial_state, weight, model, scaler, encoder, n_gen, pop_size, n_offsprings, save_history=False
):

    problem = VenusProblem(initial_state, weight, model, encoder, scaler, save_history=save_history)

    type_mask = [
        "real",
        "int",
        "real",
        "real",
        "real",
        "real",
        "real",
        "real",
        "real",
        "real",
        "real",
        "int",
        "real",
        "real",
        "int",
    ]

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

    algorithm = GA(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
        return_least_infeasible=True,
    )

    termination = get_termination("n_gen", n_gen)

    return problem, algorithm, termination
