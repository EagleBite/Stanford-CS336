import math

def get_lr_cosine_schedule(t: int, amax: float, amin: float, Tw: int, Tc: int) -> float:
    """
    Args:
        - t: Current iteration number
        - amax: Maximum learning rate
        - amin: Minimum learning rate
        - Tw: warmup iterations
        - Tc: cosine annealing iterations
    """
    if t < Tw:
        if Tw == 0:
            return float(amax)
        return float(amax) * (float(t) / float(Tw))

    if t >= Tc:
        return float(amin)

    progress = (float(t) - float(Tw)) / (float(Tc) - float(Tw)) 
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return float(amin) + cosine * (float(amax) - float(amin))