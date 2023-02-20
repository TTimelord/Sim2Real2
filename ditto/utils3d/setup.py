from setuptools import find_packages, setup

setup(
    name="utils3d",  # change "src" folder name to your project name
    version="0.0.0",
    description="some 3D related utilities",
    author="Zhenyu Jiang",
    author_email="stevetod98@gmail.com",
    url="https://github.com/Steve-Tod/utils3d",  # replace with your own github project link
    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
        "trimesh",
        "open3d",
        "scipy",
        "pyrender",
        "pytest",
        "numba",
        "scikit-image",
    ],
    packages=find_packages(),
)
