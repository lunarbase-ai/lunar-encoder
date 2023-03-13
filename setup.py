import sys
import os

os.environ["MPLCONFIGDIR"] = "."
from setuptools import find_packages
from setuptools import setup
from lunar_encoder import __version__

if sys.version_info < (3, 7, 0):
    raise OSError(f"LunarEncoder requires Python >=3.7, but yours is {sys.version}")

PKG_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements() -> list:
    """Load requirements from file, parse them as a Python list!"""

    with open(os.path.join(PKG_ROOT, "requirements.txt"), encoding="utf-8") as f:
        all_reqs = f.read().split("\n")
    install_requires = [x.strip() for x in all_reqs if "git+" not in x]

    return install_requires


setup(
    name="lunar-encoder",
    version=__version__,
    url="",
    license="",
    author="alexbogatu",
    author_email="alex.bogatu89@yahoo.com",
    description="Transformer-based sentence/passage embeddings as a service.",
    packages=find_packages(exclude=["contrib", "test-docs", "tests*"]),
    install_requires=load_requirements(),
    include_package_data=True,
    python_requires=">=3.7",
)
