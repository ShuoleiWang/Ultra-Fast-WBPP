"""Platform-wheel tagging for the optional bundled ctypes runtime."""

from setuptools import Distribution, setup
from wheel.bdist_wheel import bdist_wheel


class PlatformBinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        # The package contains a platform-native shared library even though it
        # is loaded through ctypes rather than a CPython extension module.
        return True


class PlatformPy3Wheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(
    distclass=PlatformBinaryDistribution,
    cmdclass={"bdist_wheel": PlatformPy3Wheel},
)
