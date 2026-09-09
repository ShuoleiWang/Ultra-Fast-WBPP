"""Runtime-only Astropy hook for the headless Ultra-Fast WBPP worker.

The upstream PyInstaller-contrib hook intentionally imports every Astropy
submodule.  Astropy's optional Matplotlib WCSAxes package raises a pytest Skip
when Matplotlib is absent, which makes a minimal headless release impossible to
analyze.  The worker does not use that UI package or Astropy's test data, so we
collect the runtime modules and parser tables explicitly.
"""

from PyInstaller.utils.hooks import collect_data_files, copy_metadata, is_module_satisfies


datas = collect_data_files(
    "astropy",
    include_py_files=True,
    includes=[
        "CITATION",
        "**/*_parsetab.py",
        "**/*_lextab.py",
        "wcs/wcsapi/data/*.txt",
    ],
    excludes=[
        "**/tests/**",
        "**/test/**",
        "**/visualization/**",
        "**/*.fit",
        "**/*.fits",
        "**/*.fts",
    ],
)
hiddenimports = [
    "astropy.constants.codata2010",
    "astropy.constants.codata2014",
    "astropy.constants.codata2018",
    "astropy.constants.codata2022",
    "astropy.constants.iau2012",
    "astropy.constants.iau2015",
    "astropy.coordinates",
    "astropy.io.fits",
    "astropy.units",
    "astropy.wcs",
    "astropy.wcs.utils",
    "numpy.lib.recfunctions",
]

if is_module_satisfies("astropy >= 5.0"):
    datas += copy_metadata("astropy")
    datas += copy_metadata("numpy")

hiddenimports += ["numpy.lib.recfunctions"]
