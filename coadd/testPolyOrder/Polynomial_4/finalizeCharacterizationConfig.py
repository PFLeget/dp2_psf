config.psf_determiner['piff'].spatialOrderPerBand = {
    "u": 4,
    "g": 4,
    "r": 4,
    "i": 4,
    "z": 4,
    "y": 4,
}
config.psf_determiner['piff'].zerothOrderInterpNotEnoughStars = False
config.psf_determiner['piff'].piffBasisPolynomialSolver = "cpp"
config.psf_determiner['piff'].piffPixelGridFitCenter = False
config.do_add_sky_moments = True
config.do_add_fgcm_photometry = True
config.fgcmPhotometryBands = ['u', 'g', 'r', 'i', 'z', 'y']
