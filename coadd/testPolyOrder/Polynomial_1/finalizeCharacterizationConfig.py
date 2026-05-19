config.psf_determiner['piff'].spatialOrderPerBand = {
    "u": 1,
    "g": 1,
    "r": 1,
    "i": 1,
    "z": 1,
    "y": 1,
}
config.psf_determiner['piff'].zerothOrderInterpNotEnoughStars = False
config.psf_determiner['piff'].piffBasisPolynomialSolver = "cpp"
config.psf_determiner['piff'].piffPixelGridFitCenter = False
config.do_add_sky_moments = True
config.do_add_fgcm_photometry = True
config.fgcmPhotometryBands = ['u', 'g', 'r', 'i', 'z', 'y']
