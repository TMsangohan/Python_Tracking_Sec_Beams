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

# BPM plotting
LHCclass.py -> BPM_ALL_IP.ipynb

The LCHclass.py class allows to generate LHCFill objects that are initialized
by downloading most of the relevant data for studying the ion runs. This
data is then written to local folders in CSV format so that there is no need 
for re-downloading of the data. Note that this data is preprocessed before 
being written to CSV in order to have it in relevant form (mostly true for
BSRTS data). Through an instance of the LHCfill object all data becomes
available in a pandas DataFrame form, allowing for a very flexible setup
for analyzing and plotting. 

The class now also contains a dedicated plotting function for BPM data, the
reason for this is that the data needs to be manipulated and masked in order
to get the correct plots. By adding this plotting function the user does not 
need to worry about this increasing plotting efficiency.
