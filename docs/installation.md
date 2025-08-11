# Installation

## Quick Install with UV (Recommended for v0.3.0+)

We now recommend using [UV](https://docs.astral.sh/uv/) for fast, reliable installation:

1. **Install UV**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or see the uv website for Windows instructions
   ```

2. **Create environment and install**:
   ```bash
   uv venv --python 3.8
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install deepethogram
   ```

**Note**: DeepEthogram v0.3.0+ requires Python 3.8 and uses PySide6 (upgraded from PySide2).

## Traditional Installation, NOT RECOMMENDED!

### Brief version
* Install Anaconda
* Create a new anaconda environment: `conda create --name deg python=3.8`
* Activate your environment: `conda activate deg`
* Install PyTorch: [Use this link for official instructions.](https://pytorch.org/)
* `pip install deepethogram`

### Installing from source
* `git clone https://github.com/jbohnslav/deepethogram.git`
* `cd deepethogram`
* `conda create --name deg python=3.8`
* `conda activate deg`
* `pip install -e .`

### Installing Anaconda
For instructions on installing anaconda,
please [use this link](https://www.anaconda.com/distribution/). This will install Python, some basic dependencies, and
install the Anaconda package manager. This will ensure that if you use some other project that (say) requires Python 2,
you can have both installed on your machine without interference.

* First things first, download and install Anaconda for your operating system. You can find the downloads [here](https://www.anaconda.com/distribution/#download-section). Make sure you pick the Python 3.7 version. When you're installing, make sure you pick the option something along the lines of "add anaconda to path". That way, you can use `conda` on the command line.
* Install git for your operating system (a good idea anyway!) [Downloads page here](https://git-scm.com/download)
* Open up the command line, such as terminal on mac or cmd.exe. **VERY IMPORTANT: On Windows, make sure you run the command prompt as an administrator! To do this, right click the shortcut to the command prompt, click `run as administrator`, then say yes to whatever pops up.**

## Install FFMPEG
We use FFMPEG for reading and writing `.mp4` files (with libx264 encoding). Please use [this link](https://www.ffmpeg.org/)
to install on your system.

## Startup
* `source.venv/bin/python` or (old version) `conda activate deg`. This activates the environment.
* type `deepethogram` in the command line to open the GUI.

## Common installation problems (with old, conda installers)
* You might have dependency issues with other packages you've installed. Please make a new anaconda or miniconda
environment with `conda create --name deg python=3.8` before installation.
* `module not found: PySide2` or `module not found: PySide6`. 
  * For v0.3.0+, we use PySide6. Try: `pip install --force-reinstall PySide6`
  * For older versions with PySide2: Some versions of PySide2 install poorly from pip. use `pip uninstall pyside2`, then
`conda install -c conda-forge pyside2`
* When opening the GUI, you might get `Segmentation fault (core dumped)`. 
  * For PySide6 (v0.3.0+): `pip install --force-reinstall PySide6`
  * For PySide2 (older versions): In this case; please `pip uninstall pyside2`,
`conda uninstall pyside2`. `pip install pyside2`
* `ImportError: C:\Users\jbohn\.conda\envs\deg2\lib\site-packages\shiboken2\libshiboken does not exist`
  * something went wrong with your PySide2 installation, likely on Windows.
  * Make sure you have opened your command prompt as administrator
  * If it tells you to install a new version of Visual Studio C++, please do that.
  * Now you should be set up: let's reinstall PySide2 and libshiboken.
  * `pip install --force-reinstall pyside2`
* `_init_pyside_extension is not defined`
  * This is an issue where Shiboken and PySide2 are not playing nicely together. Please `pip uninstall pyside2` and `conda remove pyside2`. Don't manually install these packages; instead, let DeepEthogram install it for you via pip. Therefore, `pip uninstall deepethogram` and `pip install deepethogram`.
*  `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in ".../python3.8/site-packages/cv2/qt/plugins"  even though it was found. This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.`
   * This is an issue with a recent version of `opencv-python` not working well with Qt. Please do `pip install --force-reinstall opencv-python-headless==4.1.2.30`