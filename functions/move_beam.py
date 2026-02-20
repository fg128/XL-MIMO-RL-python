import numpy as np


def move_beam(current_beam_idx: int, move_type: str, step_size: int, size_cb: int):
    """Calculates the new codebook index based on a desired move.
       (i.e move angle right 3, move range deeper 1).

    Uses 0-based indexing (0 to size_cb-1).

    Args:
        current_beam_idx: Current linear index in the codebook (0-based).
        move_type:        'angle' or 'range'.
        step_size:        Amount to move (e.g., +1, -1, +5, -5).
        size_cb:          Total size of the codebook (e.g., 1024).

    Returns:
        next_beam_idx: The new valid linear index (0-based).
    """
    # 1. Recover Grid Dimensions (must match codebook function logic)
    # N_angle is the inner loop size
    N_angle = round(np.sqrt(size_cb))
    N_range = size_cb // N_angle

    # Cap total size to ensure valid math
    max_idx = N_angle * N_range - 1
    current_beam_idx = max(0, min(current_beam_idx, max_idx))

    # 2. Convert Linear Index to Grid Coordinates (Row, Col)
    # Row = Range Index (1 to N_range)
    # Col = Angle Index (1 to N_angle)
    # Formula: index = (row - 1)*N_angle + col
    current_row = current_beam_idx // N_angle   # Range index
    current_col = current_beam_idx % N_angle    # Angle index

    # 3. Apply the Move
    new_row = current_row
    new_col = current_col

    if move_type == 'angle':
        # Apply step to Column (Angle)
        new_col = current_col + step_size
        new_col = max(0, min(new_col, N_angle - 1))  # Clamping
    elif move_type == 'range':
        # Apply step to Row (Range/Depth)
        new_row = current_row + step_size
        new_row = max(0, min(new_row, N_range - 1))  # Clamping

    # 4. Convert Grid Coordinates back to Linear Index
    next_beam_idx = new_row * N_angle + new_col
    return next_beam_idx
