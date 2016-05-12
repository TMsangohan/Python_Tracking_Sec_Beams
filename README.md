# Python_Tracking_Sec_Beams

Schema:
-------

cern_pymad_io_tfs.py
cern_pymad_domain.tfs.py -> MADX.ipynb -> madxmodule.py


This repository contains several python files which are used to construct a Python module (madxmodule.py).

This module allows for running MADX from within Python and has functionality to track distributions of particles
using MADX. More specifically this module was designed to generate impact distributions of the secondary beams
generated at the collision points in the LHC during ion operation.

Future update:
Use dataframes to read in the tfs files making the cern*.py files obsolete.

