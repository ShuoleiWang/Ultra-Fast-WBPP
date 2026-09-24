"""Only register image codecs emitted by Ultra-Fast WBPP previews.

The upstream Pillow hook registers every installed codec, which pulled AVIF,
WebP, JPEG2000, Tk, and their native libraries into a headless PNG/TIFF worker.
Input astronomy frames are decoded by Astropy/XISF, never Pillow plugins.
"""

hiddenimports = ["PIL.PngImagePlugin", "PIL.TiffImagePlugin"]
