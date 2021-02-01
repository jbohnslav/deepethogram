import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='deepethogram',
    version='0.0.2.post0',
    author='Jim Bohnslav',
    author_email='jbohnslav@gmail.com',
    description='Temporal action detection for biology',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    entry_points={
        'console_scripts':[
            'deepethogram = deepethogram.gui.main:entry']
    },
    python_requires='>=3.6',
    install_requires=['h5py',
                      'hydra-core<1.0',
                      'matplotlib',
                      'numpy',
                      'opencv-python',
                      'opencv-transforms',
                      'pandas',
                      'scikit-learn',
                      'scipy',
                      'tifffile',
                      'tqdm']
)
