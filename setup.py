import setuptools

# Load version
with open('ArchPy/_version.py', 'r') as f:
    exec(f.read())

# load version
with open("README.md", "r") as file:
    long_desc = file.read()

setuptools.setup(
    name='Geoarchpy',
    version=__version__,
    author="Ludovic Schorpp",
    author_email="ludovic.schorpp@unine.ch",
    description="Simpler geological and property models",
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url = "https://github.com/randlab/ArchPy",
    install_requires=['matplotlib',
                      'numpy==1.23.5',
                      'pyvista==0.40.0',
                      'trame',
                      'scipy',
                      'scikit-learn',
                      'scikit-image',
                      'pyyaml',
                      'pandas',
                      'shapely < 2.0',
                      'rasterio',
                      'geopandas',
                      'ipywidgets',
                      'tornado==6.1',
                      'notebook'],  # download geone
    packages=setuptools.find_packages(),
    include_package_data=True,
    #data_files=[("lib\\site-packages\\ArchPy\\libraries", ["ArchPy\\libraries\\cov_facies.dll"])],
    license=open('LICENSE', encoding='utf-8').read()
)
