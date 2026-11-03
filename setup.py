from setuptools import setup
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyPINTS",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["pints", ],
    install_requires=(
        "numpy>=1.19.2",
        "pandas>=1.1.5",
        "scipy>=1.5.2",
        "pysam>=0.16.0.1",
        "requests",
        "pyBigWig",
        "matplotlib",
    ),
    extras_require={"fast": ["numba>=0.56"]},
    scripts=["scripts/pints_caller", "scripts/pints_normalizer", "scripts/pints_visualizer",
             "scripts/pints_boundary_extender", "scripts/pints_counter", "scripts/pints_sample_qc"],
    url="https://pints.yulab.org",
    license="GPL",
    author="Li Yao",
    author_email="regulatorygenome@gmail.com",
    description="Peak Identifier for Nascent Transcripts Starts (PINTS)"
)
