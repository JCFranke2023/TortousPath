# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-14.55.25 RELr426 190762
# Run by franke on Sat Aug 30 11:13:59 2025
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
#: Created section: PET_section
#: Created section: interlayer_section
#: Created section: barrier_section
#: Created section: air_crack_section
#: Creating layer partitions...
#: Created layer partition at y=5.0e+02
#: Created layer partition at y=5.5e+02
#: Created layer partition at y=6.0e+02
#: Created layer partition at y=6.5e+02
#: Created layer partition at y=7.0e+02
#: Creating crack partitions...
#: Created Barrier1 crack partition at x=1.0e+02 (y=5.5e+02 to 6.0e+02)
#: Created Barrier2 crack partition at x=2.5e+03 (y=6.5e+02 to 7.0e+02)
#: Created Barrier2 crack partition at x=2.6e+03 (y=6.5e+02 to 7.0e+02)
#: Created multilayer unit cell: 1.0e+04 x 7.5e+02 m
#: Assigning materials to regions...
#: Assigned PET to substrate
#: Assigned interlayer to adhesion
#: Assigned barrier material to barrier1 (simplified assignment)
#: Assigned interlayer to interlayer
#: Assigned barrier material to barrier2 (simplified assignment)
#: Assigned interlayer to topcoat
#: Seeding part with element size: 5.0e+01 m
#: Geometry dimensions: width=1.0e+04 m, height=7.5e+02 m
#: Crack width: 1.0e+02 m
#: Using uniform mesh seeding (crack refinement not implemented)
#: Mesh generated successfully
#: Warning: Could not find top edge for inlet BC
#: Warning: Could not find bottom edge for outlet BC
#: Warning: The following parts have some elements without any section assigned: 
#:  
#:     UnitCell
#: 
#: :: WARNING: setvars.bat has already been run. Skipping re-execution.
#:    To force a re-execution of setvars.bat, use the '--force' option.
#:    Using '--force' can result in excessive use of your environment variables.
#: Organizing output files for job: Job_SS_c1e02_d1e04_o0p25
#: Organizing files for job: Job_SS_c1e02_d1e04_o0p25
#: Moved: Job_SS_c1e02_d1e04_o0p25.odb -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.odb
#: Moved: Job_SS_c1e02_d1e04_o0p25.dat -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.dat
#: Moved: Job_SS_c1e02_d1e04_o0p25.prt -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.prt
#: Moved: Job_SS_c1e02_d1e04_o0p25.com -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.com
#: Moved: Job_SS_c1e02_d1e04_o0p25.log -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.log
#: Moved: Job_SS_c1e02_d1e04_o0p25.inp -> abaqus_files\jobs\Job_SS_c1e02_d1e04_o0p25\Job_SS_c1e02_d1e04_o0p25.inp
#: Organized 6 files
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
