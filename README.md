# USF Data Science Bowl Repo

Entry for the https://www.kaggle.com/c/data-science-bowl-2017


* *Goal:* Predicting whether a CT scan is of a patient who either has or will develop cancer within the next 12 months or not.

Using Keras, we build a pipeline for a submission for this competition.

Biggest challenge was to make every 3-D image the same size without loss of information.



Table of Contents
=================
* [Module Structure](#module-structure)
* [Somewhat Assumed File Structure](#somewhat-assumed-file-structure)
* [Installing Dependencies](#installing-dependencies)
* [Running Notebook on Server](#running-notebook-on-server)
* [Jupyter Theme Is Terrible](#jupyter-theme-is-terrible)


## Module Structure
In order to combat having a zillion notebook cells & making version control suck, I've created a few sample modules.
The issue with modularizing python code and running in a notebook is that if you change code in the module, it isn't imported automatically

ex:
```python
# if i change either of these functions
# no changes are seen in whatever imports
# the functions
from preprocessing import load_scan, get_pixels_hu
```

this issue can be solved by importing and reloading a module above the current cell.
this should be temporary and shouldn't be added in version control.

```python
import preprocessing
reload(preprocessing)
```

## Somewhat Assumed File Structure
- data/ : contains all data from kaggle w/ no further modification

## Installing Dependencies
Place all libraries installed into `requirements.txt` so installation is simple.
It also might be worth specifying which version of the library is installed in order to not have issues across machines.
**List libraries as they are on pypi, ex: skimage => scikit-image**.

```bash
# locally
pip install -r requirements.txt
# on server or locally
pip install --user -r requirements.txt
```

If you don't want versions of these libraries to overwrite your (local) current versions, use a virtual environment w/ `virtualenv` or `conda`.
If you remove a library from the a notebook / source file, remove it from `requirements.txt` if it isn't required elsewhere.

## Running Notebook on Server
According to Brad, all that is needed to run a notebook on the gpu-cluster is the following...

```bash
jupyter notebook --no-browser --ip=* --port=<port>
```

After running the following code you should be able to access the notebook at server_ip:port. ~~, but I haven't gotten that to work.~~

## Jupyter Theme Is Terrible
solution: https://github.com/dunovank/jupyter-themes
```
pip install --user --upgrade jupyterthemes
jt -t cherish
```
