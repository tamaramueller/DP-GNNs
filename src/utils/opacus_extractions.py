import math
from typing import List, Tuple

# from opacus import privacy_analysis
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


def _apply_dp_sgd_analysis(
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    alphas: List[float],
    delta: float,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Computes the privacy Epsilon at a given delta via RDP accounting and
    converting to an (epsilon, delta) guarantee for a target Delta.
    Args:
        sample_rate : The sample rate in SGD
        noise_multiplier : The ratio of the standard deviation of the Gaussian
            noise to the L2-sensitivity of the function to which the noise is added
        steps : The number of steps
        alphas : A list of RDP orders
        delta : Target delta
        verbose : If enabled, will print the results of DP-SGD analysis
    Returns:
        Pair of privacy loss epsilon and optimal order alpha
    """
    rdp = compute_rdp(q=sample_rate, noise_multiplier=noise_multiplier, steps=steps, orders=alphas)
    eps, opt_alpha = get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)

    if verbose:
        print(
            f"DP-SGD with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {noise_multiplier},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {eps:.3g},"
            f"\n\tdelta = {delta}."
            f"\nThe optimal alpha is {opt_alpha}."
        )

        if opt_alpha == max(alphas) or opt_alpha == min(alphas):
            print(
                "The privacy estimate is likely to be improved by expanding "
                "the set of alpha orders."
            )
    return eps, opt_alpha


def compute_dp_sgd_privacy(
    sample_rate: float,
    noise_multiplier: float,
    epochs: int,
    delta: float = 1e-5,
    alphas: List[float] = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Performs the DP-SGD privacy analysis.
    Finds sample rate and number of steps based on the input parameters, and calls
    DP-SGD privacy analysis to find the privacy loss epsilon and optimal order alpha.
    Args:
        sample_rate : probability of each sample from the dataset to be selected for a next batch
        noise_multiplier : The ratio of the standard deviation of the Gaussian noise
            to the L2-sensitivity of the function to which the noise is added
        epochs : Number of epochs
        delta : Target delta
        alphas : A list of RDP orders
        verbose : If enabled, will print the results of DP-SGD analysis
    Returns:
        Pair of privacy loss epsilon and optimal order alpha
    Raises:
        ValueError
            When batch size is greater than sample size
    """
    if sample_rate > 1:
        raise ValueError("sample_rate must be no greater than 1")
    steps = epochs * math.ceil(1 / sample_rate)

    return _apply_dp_sgd_analysis(
        sample_rate, noise_multiplier, steps, alphas, delta, verbose
    )