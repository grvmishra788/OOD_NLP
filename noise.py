"""Functions adding noise to text"""

import random
import numpy as np


def random_bool(probability):
    """Returns True with given probability
    Args:
        probability: probability to return True
    """
    assert (0 <= probability <= 1), "probability needs to be >= 0 and <= 1"
    return random.random() < probability


def delete_random_token(line, probability):
    """Delete random tokens in a given String with given probability
    Args:
        line: a String
        probability: probability to delete each token
    """
    ret = [token for token in line if not random_bool(probability)]
    return ret


def replace_random_token(line, probability, filler_token=0):
    """Replace random tokens in a String by a filler token with given probability
    Args:
        line: a String
        probability: probability to replace each token
        filler_token: token replacing chosen tokens
    """
    for i in range(len(line)):
        if random_bool(probability):
            line[i] = filler_token
    return line


def random_token_permutation(line, _range):
    """Random permutation over the tokens of a String, restricted to a range, drawn from the uniform distribution
    Args:
        line: a String
        _range: Max range for token permutation
    """
    new_indices = [i+random.uniform(0, _range+1) for i in range(len(line))]
    res = [x for _, x in sorted(zip(new_indices, line), key=lambda pair: pair[0])]
    return res


def add_noise(X, token=0):
    noisy_X = []
    for line in X:
        line = replace_random_token(line, probability=1e-3, filler_token=token)
        line = random_token_permutation(line, _range=2)
        noisy_X.append(line)
    return np.asarray(noisy_X)
