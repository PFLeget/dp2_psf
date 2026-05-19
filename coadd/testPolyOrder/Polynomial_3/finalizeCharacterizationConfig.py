config.psf_determiner['piff'].spatialOrderPerBand = {
    "u": 3,
    "g": 3,
    "r": 3,
    "i": 3,
    "z": 3,
    "y": 3,
}
config.psf_determiner['piff'].zerothOrderInterpNotEnoughStars = False
config.psf_determiner['piff'].piffBasisPolynomialSolver = "cpp"
config.psf_determiner['piff'].piffPixelGridFitCenter = False
config.doAddSkyMoments = True
config.doAddFgcmPhotometry = True
config.fgcmPhotometryBands = ['u', 'g', 'r', 'i', 'z', 'y']
