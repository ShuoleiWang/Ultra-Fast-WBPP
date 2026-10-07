"""Multi-panel mosaics: one canvas for every panel and filter.

A mosaic is built in three steps.  The project plans one canvas from a solved
frame of every panel (:mod:`.canvas`); each panel run then resamples its
Lights once, straight onto its window of that canvas, through the panel's
TAN+SIP solution refined on catalog stars (:mod:`.astrometry`); and the
panel masters, already on one pixel lattice, are matched photometrically and
blended (:mod:`.assemble`).
"""
