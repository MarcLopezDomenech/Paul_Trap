# How to install Mayavi

Official reference: https://pypi.org/project/mayavi/

### Requirements

I needed packages "setuptools", "VTK" and a GUI toolkit like "PyQt5".

* pip install setuptooks
* pip install vtk
* pip install PyQt5

### Install mayavi

Try "pip install mayavi". That did not work for me so I downloaded the library through the mayavi GitHub repository

* git clone https://github.com/enthought/mayavi.git
* cd mayavi
* python setup.py install (I needed to run the command prompt as administrator for this)