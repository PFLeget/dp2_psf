config.psf_determiner['piff'].spatialOrderPerBand = {
    "u": 0,
    "g": 0,
    "r": 0,
    "i": 0,
    "z": 0,
    "y": 0,
}
config.psf_determiner['piff'].zerothOrderInterpNotEnoughStars = False
config.psf_determiner['piff'].piffBasisPolynomialSolver = "cpp"
config.psf_determiner['piff'].piffPixelGridFitCenter = False
config.do_add_sky_moments = True
config.do_add_fgcm_photometry = True
config.fgcmPhotometryBands = ['u', 'g', 'r', 'i', 'z', 'y']
