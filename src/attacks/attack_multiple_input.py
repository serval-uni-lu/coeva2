from joblib import Parallel, delayed

from attacks.coeva2_scaled.venus_attack import attack as coeva2_attack
from attacks.nsga2_scaled.venus_attack import attack as nsga2_attack
from attacks.constrained_random.venus_attack import attack as random_attack
from tqdm import tqdm


def attack(
    model,
    scaler,
    encoder,
    n_gen,
    pop_size,
    n_offspring,
    X_initial_states,
    weight=None,
    attack_type="coeva2",
    save_history=False
):
    results = None
    if attack_type == "coeva2":
        attack = coeva2_attack
        results = Parallel(n_jobs=-1)(
            delayed(attack)(
                index,
                initial_state,
                model,
                scaler,
                encoder,
                n_gen,
                pop_size,
                n_offspring,
                weight,
                save_history=save_history
            )
            for index, initial_state in tqdm(enumerate(X_initial_states),  total=len(X_initial_states))
        )
    elif attack_type == "random":
        attack = random_attack
        results = Parallel(n_jobs=-1)(
            delayed(attack)(
                index,
                initial_state,
                model,
                scaler,
                encoder,
                n_gen,
                pop_size,
                n_offspring,
                weight,
            )
            for index, initial_state in tqdm(enumerate(X_initial_states),  total=len(X_initial_states))
        )
    elif attack_type == "nsga2":
        attack = nsga2_attack
        results = Parallel(n_jobs=-1)(
            delayed(attack)(
                index,
                initial_state,
                model,
                scaler,
                encoder,
                n_gen,
                pop_size,
                n_offspring,
            )
            for index, initial_state in tqdm(enumerate(X_initial_states),  total=len(X_initial_states))
        )

    return results
