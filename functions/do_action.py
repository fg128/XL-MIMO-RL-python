from functions.move_beam import move_beam
from functions.move_psf import move_psf


def do_action(action: int, curr_beam_idx: int, curr_psf_idx: int, size_cb: int, psf_N: int):
    """Routes the DQN action decision to performing the action.

    Args:
        action:        Action chosen by DQN (0=Stay, 1-8=Move beam, 9-10=Power).
        curr_beam_idx: Current beamforming codebook index (0-based).
        curr_psf_idx:  Current power splitting codebook index (0-based).
        size_cb:       Total size of the codebook (e.g., 1024).
        psf_N:         N = psf_codebook_size - 1.

    Returns:
        next_beam_idx: New beamforming codebook index (0-based).
        next_psf_idx:  New power splitting codebook index (0-based).
    """
    next_beam_idx = curr_beam_idx
    next_psf_idx = curr_psf_idx

    if action == 0:    # STAY
        pass
    elif action == 1:  # Angle +1
        next_beam_idx = move_beam(curr_beam_idx, 'angle', +1, size_cb)
    elif action == 2:  # Angle +8
        next_beam_idx = move_beam(curr_beam_idx, 'angle', +8, size_cb)
    elif action == 3:  # Angle -1
        next_beam_idx = move_beam(curr_beam_idx, 'angle', -1, size_cb)
    elif action == 4:  # Angle -8
        next_beam_idx = move_beam(curr_beam_idx, 'angle', -8, size_cb)
    elif action == 5:  # Range +1
        next_beam_idx = move_beam(curr_beam_idx, 'range', +1, size_cb)
    elif action == 6:  # Range +5
        next_beam_idx = move_beam(curr_beam_idx, 'range', +5, size_cb)
    elif action == 7:  # Range -1
        next_beam_idx = move_beam(curr_beam_idx, 'range', -1, size_cb)
    elif action == 8:  # Range -5
        next_beam_idx = move_beam(curr_beam_idx, 'range', -5, size_cb)
    elif action == 9:  # PSF +1
        next_psf_idx = move_psf(curr_psf_idx, psf_N, False)
    elif action == 10: # PSF -1
        next_psf_idx = move_psf(curr_psf_idx, psf_N, True)

    return next_beam_idx, next_psf_idx
