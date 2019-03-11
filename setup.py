import setuptools
from smcfpl import __version__

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().split('\n')

setuptools.setup(
    name='smcfpl',
    version=__version__,
    author='Gabriel Seguel',
    author_email='traxiduswolf@gmail.com',
    description="Case based of Montecarlo simuations for Electric Power Systems.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="Electric Power System Montecarlo Planning",
    url='https://github.com/TraxidusWolf/SMCFPL',
    project_urls={
        # 'Documentation': 'None',
        'Source': 'https://github.com/TraxidusWolf/SMCFPL',
    },
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Natural Language :: Spanish",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    data_files=[('inputdata', ['InputData/InputData_39Bus_v6.xlsx', 'InputData/InputData_SEN_v1.xlsx'])],
)
