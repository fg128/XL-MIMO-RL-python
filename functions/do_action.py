from functions.move_beam import move_beam
from functions.move_psf import move_psf


def do_action(action: int, curr_beam_idx: int, curr_psf: float, size_cb: int):
    """Routes the DQN action decision to performing the action.

    Args:
        action:        Action chosen by DQN (0=Stay, 1-8=Move beam, 9-10=Power).
        curr_beam_idx: Current beamforming codebook index (0-based).
        curr_psf:  Current power splitting factor (0.0 to 1.0).
        size_cb:       Total size of the codebook (e.g., 1024).

    Returns:
        next_beam_idx: New beamforming codebook index (0-based).
        next_psf:  New power splitting factor (0.0 to 1.0).
    """
    next_beam_idx = curr_beam_idx
    next_psf = curr_psf

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
    elif action == 9:  # PSF +0.01
        next_psf = move_psf(curr_psf, 0.01)
    elif action == 10: # PSF -0.01
        next_psf = move_psf(curr_psf, -0.01)
    elif action == 11: # PSF +0.05
        next_psf = move_psf(curr_psf, 0.05)
    elif action == 12: # PSF -0.05
        next_psf = move_psf(curr_psf, -0.05)

    return next_beam_idx, next_psf
