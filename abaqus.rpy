# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-14.55.25 RELr426 190762
# Run by franke on Thu Aug 28 18:11:20 2025
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
#: Created directory: abaqus_files
#: Created directory: abaqus_files\jobs
#: Created directory: abaqus_files\models
#: Created directory: abaqus_files\results
#: Created directory: abaqus_files\temp
#: Created directory: abaqus_files\logs
#: Created directory: abaqus_files\archive
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
#: Organizing output files for job: Job_SS_c1e02_d1e04_o0p25
#: Organizing files for job: Job_SS_c1e02_d1e04_o0p25
#: Moved: Job_SS_c1e02_d1e04_o0p25.odb -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.odb
#: Moved: Job_SS_c1e02_d1e04_o0p25.sta -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.sta
#: Moved: Job_SS_c1e02_d1e04_o0p25.msg -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.msg
#: Moved: Job_SS_c1e02_d1e04_o0p25.dat -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.dat
#: Moved: Job_SS_c1e02_d1e04_o0p25.prt -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.prt
#: Moved: Job_SS_c1e02_d1e04_o0p25.com -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.com
#: Moved: Job_SS_c1e02_d1e04_o0p25.log -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.log
#: Moved: Job_SS_c1e02_d1e04_o0p25.inp -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.inp
#: Organized 8 files
#: Extracting data from ODB...
#: Model: C:/Users/franke/source/repos/JCFranke2023/TortousPath/./abaqus_files/results/Job_SS_c1e02_d1e04_o0p25.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       1
#: Number of Node Sets:          1
#: Number of Steps:              1
#: Data extraction successful: simulation_results\raw_data\Job_SS_c1e02_d1e04_o0p25_raw_data.json
#: Python analysis completed
#: Default simulation completed: Job_SS_c1e02_d1e04_o0p25
print('RT script done')
#: RT script done
