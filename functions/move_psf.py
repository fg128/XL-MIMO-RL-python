def move_psf(current_psf: float, change_by: float):
    """Changes the power splitting factor by a specified amount.

    Args:
        current_psf: Current power splitting factor.
        change_by:   Amount to change the power splitting factor by.

    Returns:
        next_psf: New power splitting factor.
    """
    next_psf = current_psf + change_by
    next_psf = max(0.0, min(next_psf, 1.0))  # Clamp between 0 and 1
    return next_psf

