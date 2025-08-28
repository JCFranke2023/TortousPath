# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-14.55.25 RELr426 190762
# Run by franke on Thu Aug 28 14:27:04 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.55729, 1.55556), width=229.233, 
    height=154.311)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('simulation_runner.py', __main__.__dict__)
#: The model "PermeationModel" has been created.
#: Starting PECVD barrier permeation simulation...
#: Units: nanometers, seconds
#: Running single-sided simulation:
#:   Crack width: 1.0e+02 m
#:   Crack spacing: 1.0e+04 m
#:   Crack offset: 0.25
#: Created basic unit cell geometry: 1.0e+04 x 7.5e+02 m
#: Assigned PET material to entire geometry
#: Seeding part with element size: 50.0 nm
#: Geometry dimensions: width=10000.0 nm, height=750.0 nm
#: Crack width: 100.0 nm
#: Mesh generated successfully
#: Warning: Could not find top edge for inlet BC
#: Warning: Could not find bottom edge for outlet BC
#: :: WARNING: setvars.bat has already been run. Skipping re-execution.
#:    To force a re-execution of setvars.bat, use the '--force' option.
#:    Using '--force' can result in excessive use of your environment variables.
#: Default simulation completed: Job_SS_c1e02_d1e04_o0p25
print('RT script done')
#: RT script done
