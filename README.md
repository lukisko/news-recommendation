# how to start
- you need to start by downloading dataset from https://recsys.eb.dk/
- place the zip files in the root of this folder and unzip it to appropriate folders
- I have also added requirements.txt if you are wondering what packages I am running
- please use venv of python - I have placed my in env/ folder but feel tree to name it differently and add an entry into .gitignore file

# how to train
- for training you can use lukas-NRMS-clean.ipynb
- there are all possible models, but currently it is made so that running all the things will train dummy model
- also if you are missing some package - we hav emade a dump of all loaded packages into requirements_final.txt - note that there are also packages which are not needed (such as pillow), so loading that file can be a last resort to get dependencies