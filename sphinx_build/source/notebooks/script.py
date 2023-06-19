import shutil
import os
import sys

dirname = os.path.dirname(__file__)

# copy notebooks from archpy examples folder recursively to notebooks folder
def copy_notebooks(source, dest):

    source =  os.path.abspath(source)
    print(source)
    for folder, subfolders, files in os.walk(source):
        print(folder, files)
        for file in files:
            
            # keep notebook but not checkpoints
            if file.endswith(".ipynb") and not file.endswith("checkpoint.ipynb"):
                print(file)
                shutil.copy(os.path.join(folder, file), dest)
                print(f"copying {file} to {dest}")

source =  os.path.join(dirname, "../../../examples")
dest = os.path.join(dirname, "../notebooks")

if __name__ == "__main__":
    # copy notebooks from archpy examples to notebooks folder
    copy_notebooks(source, dest)


