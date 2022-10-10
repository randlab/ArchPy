import sys, platform
import setuptools

with open("README.md", "r") as file:
    long_desc = file.read()

setuptools.setup(
    name='Geoarchpy',
    version='0.3.0',
    author="Ludovic Schorpp",
    author_email="ludovic.schorpp@unine.ch",
    description="Simpler geological and property models",
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url = "https://github.com/randlab/ArchPy",
    install_requires=['matplotlib',
                      'numpy',
                      'pyvista==0.32.1',
                      'scipy',
                      'shapely==1.7.1',
                      'numba'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    data_files=[("lib\\site-packages\\ArchPy\\libraries", ["ArchPy\\libraries\\cov_facies.dll"])],
    license=open('LICENSE', encoding='utf-8').read()
)