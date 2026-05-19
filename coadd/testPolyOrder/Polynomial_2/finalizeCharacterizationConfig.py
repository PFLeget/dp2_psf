config.psf_determiner['piff'].spatialOrderPerBand = {
    "u": 2,
    "g": 2,
    "r": 2,
    "i": 2,
    "z": 2,
    "y": 2,
}
config.psf_determiner['piff'].zerothOrderInterpNotEnoughStars = False
config.psf_determiner['piff'].piffBasisPolynomialSolver = "cpp"
config.psf_determiner['piff'].piffPixelGridFitCenter = False
config.doAddSkyMoments = True
config.doAddFgcmPhotometry = True
config.fgcmPhotometryBands = ['u', 'g', 'r', 'i', 'z', 'y']
