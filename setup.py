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
                      'numpy<2',
                      'scipy',
                      'scikit-learn',
                      'scikit-image',
                      'geone',
                      'pandas',
                      'shapely',
                      'ipywidgets',
                      'numba',
                      'flopy',
                      ],  
    extras_require={
        "all": ['pyvista < 0.47',
                'trame',
                'trame-vuetify',
                'trame-vtk',
                'notebook',
                'ipympl',
                'pyyaml',
                'rasterio',
                'geopandas',
                'pyshp',
                'seaborn']
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    #data_files=[("lib\\site-packages\\ArchPy\\libraries", ["ArchPy\\libraries\\cov_facies.dll"])],
    data_files=[("lib\\site-packages\\ArchPy\\C_modules", ["ArchPy\\C_modules\\simplified_renorm_C.cp39-win_amd64.pyd",
                                                           "ArchPy\\C_modules\\simplified_renorm_C.cp310-win_amd64.pyd",
                                                           "ArchPy\\C_modules\\simplified_renorm_C.cp311-win_amd64.pyd",
                                                           "ArchPy\\C_modules\\simplified_renorm_C.cp312-win_amd64.pyd",
                                                           "ArchPy\\C_modules\\simplified_renorm_C.cp313-win_amd64.pyd",
                                                           ])],
    license=open('LICENSE', encoding='utf-8').read()
)
