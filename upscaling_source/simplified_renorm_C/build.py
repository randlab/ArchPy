# python script to update uppy simplified renormalization C version
import sys
import subprocess
import shutil

env_name = "SR_C"
python_versions = ["3.9", "3.10", "3.11", "3.12", "3.13"]
# create a new conda environment and run setup.py

def create_modules():
    # Define the name of the new conda environment

    for py_ver in python_versions:
        
        full_env_name = f"{env_name}_{str(py_ver)}"
        print(full_env_name)

        # python 3.9
        # Create the conda environment
        subprocess.run(f"conda create --name {full_env_name} python={py_ver} -yf", shell=True)    
        # Activate the conda environment and install packages
        subprocess.run(f"conda activate {full_env_name} && pip install pybind11", shell=True)
        #build
        subprocess.run(f"conda activate {full_env_name} && python setup.py build_ext -b ../../ArchPy/C_modules ", shell=True)

        # clear build directory
        shutil.rmtree("build")

def clean():

    for py_ver in python_versions:
        full_env_name = f"{env_name}_{str(py_ver)}"
        # remove environement
        print(full_env_name)
        subprocess.run(f"conda env remove -n {full_env_name}", shell=True)

    shutil.rmtree("build")

if __name__ == "__main__":
    # clone the repository
    flag = True
    # depending on arguments, create the environment or run the notebooks
    # if len(sys.argv) > 1:
    if "env" in sys.argv:
        create_modules()
    
    if "clean" in sys.argv:
        clean()
    