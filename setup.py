import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='smcfpl',
    version='1.0',
    scripts=['???'],
    author='Gabriel Seguel',
    author_email='traxiduswolf@gmail.com',
    long_description=long_description,
    long_desription_content_type='text/markdown',
    url='https://github.com/TraxidusWolf/SMCFPL.git',
    packages=setuptools.find_packages(),
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
    ],
)
