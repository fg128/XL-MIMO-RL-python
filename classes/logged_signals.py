class LoggedSignals:
    def __init__(self, bob_loc, eve_loc, current_beam_idx, current_psf):
        self.bob_loc = bob_loc
        self.eve_loc = eve_loc
        self.current_beam_idx = current_beam_idx
        self.current_psf = current_psf
