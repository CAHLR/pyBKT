#########################################
# update_mastery.py                     #
# One-answer BKT mastery update         #
#########################################

import numpy as np

def update_mastery(mastery, correct, guess, slip, learn, forget = 0.0):
    """
    Returns a student's P(known) before their next answer, given P(known) before this
    answer and whether this answer was correct. No model is changed and nothing is refit.

    The update is Bayes' rule on the answer, P(known | answer), followed by the
    learn/forget transition: P(known next) = P(known | answer) * (1 - forget) + P(unknown | answer) * learn.

    Both ``mastery`` and the result follow the convention of the ``state_predictions``
    column of ``Model.predict``: the probability of knowing the skill before an attempt.
    For a student's first answer, pass the fitted ``prior``. Pass the guess and slip of the
    answer's guess/slip class and the learn and forget of its learn class
    (``fit_model[skill]['guesses'][i]`` and so on).

    >>> update_mastery(0.3, True, guess = 0.2, slip = 0.1, learn = 0.15)
    0.7097560975609756
    """
    known_likelihood, unknown_likelihood = (1 - slip, guess) if correct else (slip, 1 - guess)
    # The forward pass reads an answer probability of exactly 0 as 1, so a guess of 1 or a
    # slip of 0 never zeroes out both states. Doing the same keeps the results equal.
    known = mastery * (known_likelihood if known_likelihood != 0 else 1)
    unknown = (1 - mastery) * (unknown_likelihood if unknown_likelihood != 0 else 1)
    return _transition(known, unknown, learn, forget)

def update_mastery_array(mastery, correct, guess, slip, learn, forget = 0.0):
    """
    Vectorized ``update_mastery`` for many students at once. Every argument is a scalar or a
    NumPy array, broadcast together; ``correct`` is a boolean or 0/1 array. Each element gives
    the same result as ``update_mastery`` on that element.

    >>> update_mastery_array(np.array([0.3, 0.3]), np.array([1, 0]), 0.2, 0.1, 0.15)
    array([0.7097561 , 0.19322034])
    """
    correct = np.asarray(correct, dtype = bool)
    known_likelihood = np.where(correct, 1 - slip, slip)
    unknown_likelihood = np.where(correct, guess, 1 - guess)
    known = mastery * np.where(known_likelihood == 0, 1, known_likelihood)
    unknown = (1 - mastery) * np.where(unknown_likelihood == 0, 1, unknown_likelihood)
    return _transition(known, unknown, learn, forget)

def transition_mastery(mastery, learn, forget = 0.0):
    """
    Returns P(known) before the next attempt after an attempt with no observed answer, which is
    how ``Model.predict`` treats a ``correct`` value of -1: only the learn/forget transition,
    P(known next) = P(known) * (1 - forget) + P(unknown) * learn. Accepts scalars or NumPy arrays.

    >>> transition_mastery(0.5, learn = 0.2)
    0.6
    """
    return _transition(mastery, 1 - mastery, learn, forget)

def _transition(known, unknown, learn, forget):
    # known and unknown are the unnormalized posterior weights. Normalizing both, rather than
    # dividing by P(answer) and taking 1 - P(known | answer), keeps precision near 0 and 1 and
    # matches the operation order of the forward pass in EM_fit.
    total = known + unknown
    return learn * (unknown / total) + (1 - forget) * (known / total)
