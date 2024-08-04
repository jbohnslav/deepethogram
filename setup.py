import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

def get_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setuptools.setup(name='deepethogram',
                 version='0.1.5',
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
                 install_requires=get_requirements())
