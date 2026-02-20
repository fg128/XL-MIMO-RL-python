def move_psf(current_psf_idx: int, psf_N: int, is_decrement: bool):
    """Changes the index of power splitting codebook by +/-1.

    Args:
        current_psf_idx: Current index of power splitting codebook.
        psf_N:           1/N linear spaing of psf codebook.
        is_decrement:    If True, decrease idx by 1; else increase by 1.

    Returns:
        next_psf_idx: New power splitting factor index.
    """
    if is_decrement:
        next_psf_idx = current_psf_idx - 1
    else:
        next_psf_idx = current_psf_idx + 1

    next_psf_idx = max(0, min(next_psf_idx, psf_N)) # Clamp
    return next_psf_idx
