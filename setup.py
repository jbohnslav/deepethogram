import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(name='deepethogram',
                 version='0.1.4',
                 author='Jim Bohnslav',
                 author_email='jbohnslav@gmail.com',
                 description='Temporal action detection for biology',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 include_package_data=True,
                 packages=setuptools.find_packages(),
                 classifiers=['Programming Language :: Python :: 3', 'Operating System :: OS Independent'],
                 entry_points={'console_scripts': ['deepethogram = deepethogram.gui.main:entry']},
                 python_requires='>=3.6',
                 install_requires=[
                     'chardet<4.0', 'h5py', 'kornia>=0.5', 'matplotlib', 'numpy', 'omegaconf>=2',
                     'opencv-python-headless', 'opencv-transforms', 'pandas<1.4', 'PySide2', 'scikit-learn<1.1',
                     'scipy<1.8', 'tqdm', 'vidio', 'pytorch_lightning>=2.0.7'
                 ])
