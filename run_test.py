# python script to update github repository of ArchPy and run the notebooks
import os
import sys
import subprocess
import shutil

"""
This script is used to run the notebooks in the examples folder of the ArchPy repository.
The script creates a new conda environment, installs the package and runs the notebooks.

The script can be run with the following arguments:
- env: to create the conda environment
- notebook: to run the notebooks
"""

env_name = "archpy"
package_name = "Geoarchpy"
# create a new conda environment and run setup.py
def create_env():
    # Define the name of the new conda environment
    
    # Create the new conda environment
    subprocess.run(f"conda create --name {env_name} python=3.11 -yf", shell=True)

    # Activate the new conda environment and install packages
    subprocess.run(f"conda activate {env_name} && pip install .[all]", shell=True)

# function to run the notebooks and check if they run without errors
def run_notebooks():

    # create a log file to store the output of the notebooks
    with open("notebook_log.txt", "w") as f:
        f.write("")

    notebook_folder = "examples"
    # Activate the new conda environment
    alias = f"conda activate {env_name} && jupyter nbconvert --execute --to notebook --inplace"
    # use an alias

    # get list of all notebooks
    notebooks_subfolders = []

    for folder in notebooks_subfolders:
        for notebook in os.listdir(f"{notebook_folder}/{folder}"):
            if notebook.endswith(".ipynb"):
                print(f"\nRunning notebook: {notebook}")
                # subprocess.run(f"{alias} {notebook_folder}/01_basic/{notebook}", shell=True)
                output = subprocess.run(f"{alias} {notebook_folder}/{folder}/{notebook}", shell=True, capture_output=True)
                if output.returncode != 0:
                    print("Error in notebook {}".format(notebook))
                    with open("notebook_log.txt", "a") as f:
                        f.write(f"###################################################\n")
                        f.write(f"Error in notebook: {notebook}\n")
                        f.write(output.stderr.decode("utf-8"))
    
    # write yml file to store the environment
    subprocess.run(f"conda env export --name {env_name} > env.yml", shell=True)
    
def run_script_notebooks():
    """
    This function runs the script in sphinx folder to transfer the notebooks to the documentation folder
    """
    subprocess.run("python sphinx_build/source/notebooks/script.py", shell=True)

def upload_to_pypi():

    # remove previous build and egg-info folders
    if os.path.exists(f"{package_name}.egg-info"):
        shutil.rmtree(f"{package_name}.egg-info")
    if os.path.exists("build"):
        shutil.rmtree("build")

    # install/upgrade build and twine
    subprocess.run("pip install --upgrade build twine", shell=True)

    # build the package
    subprocess.run("python -m build", shell=True)

    # upload to pypi
    subprocess.run("twine upload dist/*", shell=True)

    # remove build and egg-info folders
    shutil.rmtree("Geoarchpy.egg-info")
    shutil.rmtree("dist")

def upload_to_github():

    # upload to github
    subprocess.run("git add .", shell=True)
    subprocess.run("git commit -m 'update'", shell=True)
    subprocess.run("git push", shell=True)

def check_errors(flag):

    with open("notebook_log.txt", "r") as f:
        log = f.read()
        if len(log) > 0:
            flag = False
        else:
            print("All notebooks ran without errors")
    return flag

def update_readme():
    import subprocess
    """
    Function to update the README.md file with the latest version of the used packaged
    """

    with open("README.md", "r") as f:
        readme = f.read()

    list_packages = ["numpy", "pandas", "matplotlib", "scipy", "sklearn", "geopandas", "rasterio", "shapely", "pyvista", "yaml", "ipywidgets", "flopy"]

    import numpy
    import pandas
    import matplotlib
    import scipy
    import sklearn
    import geopandas
    import rasterio
    import shapely
    import pyvista
    import yaml
    import ipywidgets
    import flopy

    for package in list_packages:
        
        # find where the package is mentioned in the README
        start = readme.find(f"{package}")
        end = readme.find("\n", start)
        # get the version of the package
        version = eval(package).__version__
        # replace the version
        readme = readme[:start] + f"{package} (tested with {version})" + readme[end:]

    with open("README.md", "w") as f:
        f.write(readme)
    

##############################################################################
if __name__ == "__main__":
    # clone the repository
    flag = True
    # depending on arguments, create the environment or run the notebooks
    if len(sys.argv) > 1:
        if "env" in sys.argv:
            create_env()

        if "notebook" in sys.argv:
            run_notebooks()
            run_script_notebooks()

            # check if there are any errors in the notebooks
            flag = check_errors(flag)
        
        if "readme" in sys.argv:
            update_readme()

        if "pypi" in sys.argv:
            check_errors(flag)
            if flag:
                upload_to_pypi()

            # if "github" in sys.argv:    
                # upload_to_github()
            else:
                print("There are errors in the notebooks")

        if "all" in sys.argv:
            create_env()
            run_notebooks()
            run_script_notebooks()
            check_errors(flag)
            update_readme()
            if flag:
                run_script_notebooks()
                upload_to_pypi()
                # upload_to_github()
            else:
                print("There are errors in the notebooks")
    else:
        print("Please provide arguments to the script")
