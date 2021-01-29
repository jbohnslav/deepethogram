# Installation

## Brief version
* Install Anaconda
* Create a new anaconda environment: `conda create --name deg python=3.7`
* Activate your environment: `conda activate deg`
* Install PySide2: `conda install -c conda-forge pyside2==5.13.2`
* Install PyTorch: [Use this link for official instructions.](https://pytorch.org/) 
* `pip install deepethogram`. 

## Installing from source
* `git clone https://github.com/jbohnslav/deepethogram.git`
* `cd deepethogram`
* `conda env create -f environment.yml`
    * Be prepared to wait a long time!! On mechanical hard drives, this may take 5-10 minutes (or more). Interrupting here will cause installation to fail. 
* `conda activate deg`
* `python setup.py install`

### Installing Anaconda
For instructions on installing anaconda, 
please [use this link](https://www.anaconda.com/distribution/). This will install Python, some basic dependencies, and 
install the Anaconda package manager. This will ensure that if you use some other project that (say) requires Python 2, 
you can have both installed on your machine without interference.

* First things first, download and install Anaconda for your operating system. You can find the downloads [here](https://www.anaconda.com/distribution/#download-section). Make sure you pick the Python 3.7 version. When you're installing, make sure you pick the option something along the lines of "add anaconda to path". That way, you can use `conda` on the command line.
* Install git for your operating system (a good idea anyway!) [Downloads page here](https://git-scm.com/download)
* Open up the command line, such as terminal on mac or cmd.exe. **VERY IMPORTANT: On Windows, make sure you run the command prompt as an administrator! To do this, right click the shortcut to the command prompt, click `run as administrator`, then say yes to whatever pops up.**

## Installing from pip
First install the latest version of PyTorch for your system. [Use this link for official instructions.](https://pytorch.org/) 
It should be as simple as `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`, or 
`pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html`. 

After installing PyTorch, simply use `pip install deepethogram`. 

## Install FFMPEG
We use FFMPEG for reading and writing `.mp4` files (with libx264 encoding). Please use [this link](https://www.ffmpeg.org/)
to install on your system.
    
## Startup
* `conda activate deg`. This activates the environment.
* type `python -m deepethogram`, in the command line to open the GUI.

## Common installation problems
* You might have dependency issues with other packages you've installed. Please make a new anaconda or miniconda 
environment with `conda create --name deg python=3.7` before installation. 
* `module not found: PySide2`. Some versions of PySide2 install poorly from pip. use `pip uninstall pyside2`, then 
`conda install -c conda-forge pyside2`
* When opening the GUI, you might get `Segmentation fault (core dumped)`. In this case; please `pip uninstall pyside2`, 
`conda uninstall pyside2`. `conda install -c conda-forge pyside2==5.13.2`
