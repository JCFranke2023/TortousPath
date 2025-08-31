# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-14.55.25 RELr426 190762
# Run by franke on Sun Aug 31 11:47:47 2025
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
#: === GEOMETRY GENERATOR STARTED ===
#: ABAQUS environment detected
#: === SIMULATION RUNNER STARTED ===
#: ABAQUS environment detected
#: Creating global SimulationRunner instance...
#: SimulationRunner initialized with model: PermeationModel
#: Created directory: abaqus_files
#: Created directory: abaqus_files\jobs
#: Created directory: abaqus_files\models
#: Created directory: abaqus_files\results
#: Created directory: abaqus_files\temp
#: Created directory: abaqus_files\logs
#: Created directory: abaqus_files\archive
#: File organization structure created
#: The model "PermeationModel" has been created.
#: Created new ABAQUS model: PermeationModel
#: GeometryGenerator initialized
#: Layer thicknesses initialized:
#:   h0_substrate: 5.0e+02
#:   h1_adhesion: 5.0e+01
#:   h2_barrier1: 5.0e+01
#:   h3_interlayer: 5.0e+01
#:   h4_barrier2: 5.0e+01
#:   h5_topcoat: 5.0e+01
#:   total_height: 7.5e+02
#: GeometryGenerator initialized with model
#: Simulation parameters initialized:
#:   total_time: 86400.0 s (24.0 hours)
#:   initial_increment: 1.0 s
#:   max_increment: 3600.0 s (60.0 min)
#:   min_increment: 1e-06 s
#:   dcmax: 1000.0
#:   inlet_concentration: 1.0
#:   outlet_concentration: 0.0
#: SimulationRunner instance created successfully
#: === MAIN EXECUTION STARTED ===
#: Starting PECVD barrier permeation simulation...
#: Units: nanometers, seconds
#: Running default simulation with parameters:
#:   crack_width: 100.0 nm
#:   crack_spacing: 10000.0 nm
#:   crack_offset: 0.25
#: === STARTING SINGLE SIMULATION ===
#: Simulation parameters (all nanometers):
#:   crack_width: 100.0 nm
#:   crack_spacing: 10000.0 nm (10.0 um)
#:   crack_offset: 0.25 (25.0% of spacing)
#: Expected simulation characteristics:
#:   crack area fraction: 1.000%
#: Phase 1: Model setup
#: === SETTING UP MODEL (NANOMETER UNITS) ===
#: Input parameters (all in nanometers):
#:   crack_width: 100.0 nm
#:   crack_spacing: 10000.0 nm (10.0 um)
#:   crack_offset: 0.25
#: Step 1: Setting geometry parameters...
#: Setting parameters (nanometer units):
#:   crack_width: 100.0 nm
#:   crack_spacing: 10000.0 nm (10.0 um)
#:   crack_offset: 0.25
#:   single_sided: True
#:   total_height updated: 750.0 -> 750.0 nm
#:   Geometry parameters set successfully (0.00s)
#: Step 2: Creating materials...
#: Creating materials...
#:   Created material PET: D=1.0e+06, S=1.0e-33
#:   Created material interlayer: D=5.0e+07, S=2.0e-33
#:   Created material barrier: D=1.0e-02, S=1.0e-36
#:   Created material air_crack: D=2.4e+13, S=4.0e-29
#:   Verifying materials creation:
#:     Expected materials: ['PET', 'interlayer', 'barrier', 'air_crack']
#:     Created materials: ['PET', 'air_crack', 'barrier', 'interlayer']
#:     All materials created successfully
#:     PET: D=1.0e+06 m²/s, S=1.0e-33 mol/m³/Pa
#:     air_crack: D=2.4e+13 m²/s, S=4.0e-29 mol/m³/Pa
#:     barrier: D=1.0e-02 m²/s, S=1.0e-36 mol/m³/Pa
#:     interlayer: D=5.0e+07 m²/s, S=2.0e-33 mol/m³/Pa
#:   Materials created successfully (0.01s)
#: Step 3: Creating sections...
#: Creating sections...
#:   Created section: PET_section
#:   Created section: interlayer_section
#:   Created section: barrier_section
#:   Created section: air_crack_section
#:   Verifying sections creation:
#:     Expected sections: ['PET_section', 'interlayer_section', 'barrier_section', 'air_crack_section']
#:     Created sections: ['PET_section', 'air_crack_section', 'barrier_section', 'interlayer_section']
#:     All sections created successfully
#:   Sections created successfully (0.01s)
#: Step 4: Creating unit cell geometry...
#: Creating unit cell geometry (nanometer units)...
#:   Creating part: UnitCell (width=10000.0 nm, height=750.0 nm)
#:   Base geometry created successfully
#:   Creating layer partitions (nanometer coordinates)...
#:     Layer 0 interface at y=500.0 nm (thickness=500.0 nm)
#:     Layer 1 interface at y=550.0 nm (thickness=50.0 nm)
#:     Layer 2 interface at y=600.0 nm (thickness=50.0 nm)
#:     Layer 3 interface at y=650.0 nm (thickness=50.0 nm)
#:     Layer 4 interface at y=700.0 nm (thickness=50.0 nm)
#:     Created layer partition at y=500.0 nm
#:     Created layer partition at y=550.0 nm
#:     Created layer partition at y=600.0 nm
#:     Created layer partition at y=650.0 nm
#:     Created layer partition at y=700.0 nm
#:   Creating crack partitions (nanometer coordinates)...
#:     Barrier1 cracks: y=550.0 to 600.0 nm, offset=0.0 nm
#:     Creating vertical crack partitions for Barrier1 (nm coordinates)
#:       Crack boundaries: x1=0.0, x2=100.0 nm (width=100.0 nm)
#:       Normal crack within domain
#:       Skipped partition at boundary x=0.0 nm
#:       Created Barrier1 crack partition at x=100.0 nm
#:     Barrier2 cracks: y=650.0 to 700.0 nm, offset=2500.0 nm
#:     Creating vertical crack partitions for Barrier2 (nm coordinates)
#:       Crack boundaries: x1=2500.0, x2=2600.0 nm (width=100.0 nm)
#:       Normal crack within domain
#:       Created Barrier2 crack partition at x=2500.0 nm
#:       Created Barrier2 crack partition at x=2600.0 nm
#:   Unit cell geometry completed with 9 faces
#:   Unit cell geometry created: 9 faces, 28 edges, 20 vertices
#:   Geometry details:
#:     Total faces: 9
#:     Total edges: 28
#:     Total vertices: 20
#:     Edges (first 10, coordinates in nm):
#:       Edge 0: (2600.0,650.0) to (2600.0,700.0) nm
#:       Edge 1: (2600.0,650.0) to (10000.0,650.0) nm
#:       Edge 2: (10000.0,700.0) to (10000.0,650.0) nm
#:       Edge 3: (2600.0,700.0) to (10000.0,700.0) nm
#:       Edge 4: (2500.0,700.0) to (2600.0,700.0) nm
#:       Edge 5: (2500.0,650.0) to (2500.0,700.0) nm
#:       Edge 6: (2500.0,650.0) to (2600.0,650.0) nm
#:       Edge 7: (100.0,550.0) to (100.0,600.0) nm
#:       Edge 8: (100.0,550.0) to (10000.0,550.0) nm
#:       Edge 9: (10000.0,600.0) to (10000.0,550.0) nm
#:       ... (18 more edges)
#:   Geometry creation completed (0.09s)
#: Step 5: Creating assembly instance...
#:   Assembly instance created: UnitCell-1
#:   Assembly creation completed (0.00s)
#: Step 6: Creating mesh...
#: Creating mesh (nanometer units)...
#:   Element type set to DC2D4
#:   Seeding part with element size: 50.0 nm
#:   Geometry dimensions: width=10000.0 nm, height=750.0 nm
#:   Part seeded successfully
#:   Mesh generated successfully
#:   Mesh created successfully
#:   Mesh creation completed (0.03s)
#: Step 7: Assigning materials to regions...
#: === STARTING MATERIAL ASSIGNMENT (FIXED CENTROID) ===
#: Total faces in part: 9
#: Skipping face centroid examination (causes ABAQUS API errors)
#: Checking available sections in model:
#:   - PET_section
#:   - air_crack_section
#:   - barrier_section
#:   - interlayer_section
#: Calculating layer boundaries (nanometer coordinates):
#:   substrate: y=0.0 to 500.0 nm
#:   adhesion: y=500.0 to 550.0 nm
#:   barrier1: y=550.0 to 600.0 nm
#:   interlayer: y=600.0 to 650.0 nm
#:   barrier2: y=650.0 to 700.0 nm
#:   topcoat: y=700.0 to 750.0 nm
#: Trying different assignment strategies...
#: Strategy 1: Centroid-based assignment
#:   Assigning materials by centroid analysis...
#:     Processing layer substrate: y=0.0e+00 to 5.0e+02
#:     Searching for faces in layer: y=0.0 to 500.0 nm
#:     Total faces in part: 9
#:       Face 0: centroid=(6300.0, 675.0, 0.0) nm
#:         -> Face 0 outside layer bounds
#:       Face 1: centroid=(2550.0, 675.0, 0.0) nm
#:         -> Face 1 outside layer bounds
#:       Face 2: centroid=(5050.0, 575.0, 0.0) nm
#:         -> Face 2 outside layer bounds
#:       Face 3: centroid=(1250.0, 675.0, 0.0) nm
#:         -> Face 3 outside layer bounds
#:       Face 4: centroid=(5000.0, 625.0, 0.0) nm
#:         -> Face 4 outside layer bounds
#:       Face 5: centroid=(50.0, 575.0, 0.0) nm
#:         -> Face 5 outside layer bounds
#:       Face 6: centroid=(5000.0, 525.0, 0.0) nm
#:         -> Face 6 outside layer bounds
#:       Face 7: centroid=(5000.0, 250.0, 0.0) nm
#:         -> Face 7 included in layer
#:       Face 8: centroid=(5000.0, 725.0, 0.0) nm
#:         -> Face 8 outside layer bounds
#:     Found 1 faces in layer
#:     Assigned PET to substrate (1 faces)
#:     Processing layer adhesion: y=5.0e+02 to 5.5e+02
#:     Searching for faces in layer: y=500.0 to 550.0 nm
#:     Total faces in part: 9
#:       Face 0: centroid=(6300.0, 675.0, 0.0) nm
#:         -> Face 0 outside layer bounds
#:       Face 1: centroid=(2550.0, 675.0, 0.0) nm
#:         -> Face 1 outside layer bounds
#:       Face 2: centroid=(5050.0, 575.0, 0.0) nm
#:         -> Face 2 outside layer bounds
#:       Face 3: centroid=(1250.0, 675.0, 0.0) nm
#:         -> Face 3 outside layer bounds
#:       Face 4: centroid=(5000.0, 625.0, 0.0) nm
#:         -> Face 4 outside layer bounds
#:       Face 5: centroid=(50.0, 575.0, 0.0) nm
#:         -> Face 5 outside layer bounds
#:       Face 6: centroid=(5000.0, 525.0, 0.0) nm
#:         -> Face 6 included in layer
#:       Face 7: centroid=(5000.0, 250.0, 0.0) nm
#:         -> Face 7 outside layer bounds
#:       Face 8: centroid=(5000.0, 725.0, 0.0) nm
#:         -> Face 8 outside layer bounds
#:     Found 1 faces in layer
#:     Assigned interlayer to adhesion (1 faces)
#:     Processing layer barrier1: y=5.5e+02 to 6.0e+02
#:     Searching for faces in layer: y=550.0 to 600.0 nm
#:     Total faces in part: 9
#:       Face 0: centroid=(6300.0, 675.0, 0.0) nm
#:         -> Face 0 outside layer bounds
#:       Face 1: centroid=(2550.0, 675.0, 0.0) nm
#:         -> Face 1 outside layer bounds
#:       Face 2: centroid=(5050.0, 575.0, 0.0) nm
#:         -> Face 2 included in layer
#:       Face 3: centroid=(1250.0, 675.0, 0.0) nm
#:         -> Face 3 outside layer bounds
#:       Face 4: centroid=(5000.0, 625.0, 0.0) nm
#:         -> Face 4 outside layer bounds
#:       Face 5: centroid=(50.0, 575.0, 0.0) nm
#:         -> Face 5 included in layer
#:       Face 6: centroid=(5000.0, 525.0, 0.0) nm
#:         -> Face 6 outside layer bounds
#:       Face 7: centroid=(5000.0, 250.0, 0.0) nm
#:         -> Face 7 outside layer bounds
#:       Face 8: centroid=(5000.0, 725.0, 0.0) nm
#:         -> Face 8 outside layer bounds
#:     Found 2 faces in layer
#:     Assigned barrier to barrier1 (2 faces)
#:     Processing layer interlayer: y=6.0e+02 to 6.5e+02
#:     Searching for faces in layer: y=600.0 to 650.0 nm
#:     Total faces in part: 9
#:       Face 0: centroid=(6300.0, 675.0, 0.0) nm
#:         -> Face 0 outside layer bounds
#:       Face 1: centroid=(2550.0, 675.0, 0.0) nm
#:         -> Face 1 outside layer bounds
#:       Face 2: centroid=(5050.0, 575.0, 0.0) nm
#:         -> Face 2 outside layer bounds
#:       Face 3: centroid=(1250.0, 675.0, 0.0) nm
#:         -> Face 3 outside layer bounds
#:       Face 4: centroid=(5000.0, 625.0, 0.0) nm
#:         -> Face 4 included in layer
#:       Face 5: centroid=(50.0, 575.0, 0.0) nm
#:         -> Face 5 outside layer bounds
#:       Face 6: centroid=(5000.0, 525.0, 0.0) nm
#:         -> Face 6 outside layer bounds
#:       Face 7: centroid=(5000.0, 250.0, 0.0) nm
#:         -> Face 7 outside layer bounds
#:       Face 8: centroid=(5000.0, 725.0, 0.0) nm
#:         -> Face 8 outside layer bounds
#:     Found 1 faces in layer
#:     Assigned interlayer to interlayer (1 faces)
#:     Processing layer barrier2: y=6.5e+02 to 7.0e+02
#:     Searching for faces in layer: y=650.0 to 700.0 nm
#:     Total faces in part: 9
#:       Face 0: centroid=(6300.0, 675.0, 0.0) nm
#:         -> Face 0 included in layer
#:       Face 1: centroid=(2550.0, 675.0, 0.0) nm
#:         -> Face 1 included in layer
#:       Face 2: centroid=(5050.0, 575.0, 0.0) nm
#:         -> Face 2 outside layer bounds
#:       Face 3: centroid=(1250.0, 675.0, 0.0) nm
#:         -> Face 3 included in layer
#:       Face 4: centroid=(5000.0, 625.0, 0.0) nm
#:         -> Face 4 outside layer bounds
#:       Face 5: centroid=(50.0, 575.0, 0.0) nm
#:         -> Face 5 outside layer bounds
#:       Face 6: centroid=(5000.0, 525.0, 0.0) nm
#:         -> Face 6 outside layer bounds
#:       Face 7: centroid=(5000.0, 250.0, 0.0) nm
#:         -> Face 7 outside layer bounds
#:       Face 8: centroid=(5000.0, 725.0, 0.0) nm
#:         -> Face 8 outside layer bounds
#:     Found 3 faces in layer
#:     Assigned barrier to barrier2 (3 faces)
#:     Processing layer topcoat: y=7.0e+02 to 7.5e+02
#:     Searching for faces in layer: y=700.0 to 750.0 nm
#:     Total faces in part: 9
#:       Face 0: centroid=(6300.0, 675.0, 0.0) nm
#:         -> Face 0 outside layer bounds
#:       Face 1: centroid=(2550.0, 675.0, 0.0) nm
#:         -> Face 1 outside layer bounds
#:       Face 2: centroid=(5050.0, 575.0, 0.0) nm
#:         -> Face 2 outside layer bounds
#:       Face 3: centroid=(1250.0, 675.0, 0.0) nm
#:         -> Face 3 outside layer bounds
#:       Face 4: centroid=(5000.0, 625.0, 0.0) nm
#:         -> Face 4 outside layer bounds
#:       Face 5: centroid=(50.0, 575.0, 0.0) nm
#:         -> Face 5 outside layer bounds
#:       Face 6: centroid=(5000.0, 525.0, 0.0) nm
#:         -> Face 6 outside layer bounds
#:       Face 7: centroid=(5000.0, 250.0, 0.0) nm
#:         -> Face 7 outside layer bounds
#:       Face 8: centroid=(5000.0, 725.0, 0.0) nm
#:         -> Face 8 included in layer
#:     Found 1 faces in layer
#:     Assigned interlayer to topcoat (1 faces)
#:   Centroid assignment: 9 faces assigned
#: Strategy 1 succeeded
#: === MATERIAL ASSIGNMENT COMPLETED ===
#:   Verifying section assignments:
#:     Total faces: 9
#:     Section assignments: 6
#:       Assignment 0: PET_section -> 4 faces
#:       Assignment 1: interlayer_section -> 4 faces
#:       Assignment 2: barrier_section -> 4 faces
#:       Assignment 3: interlayer_section -> 4 faces
#:       Assignment 4: barrier_section -> 4 faces
#:       Assignment 5: interlayer_section -> 4 faces
#:     Section assignment summary:
#:       Total faces: 9
#:       Assigned faces: 24
#:       Unassigned faces: -15
#:         PET_section: 4 faces (44.4%)
#:         interlayer_section: 12 faces (133.3%)
#:         barrier_section: 8 faces (88.9%)
#:     SUCCESS: All faces have section assignments
#:   Material assignment completed (0.13s)
#: Step 8: Final model verification...
#:   Final model verification:
#:     Model components:
#:       Materials: 4
#:       Sections: 4
#:       Parts: 1
#:       Assembly instances: 1
#:     Model readiness checks:
#:       Materials created: PASS
#:       Sections created: PASS
#:       Geometry created: PASS
#:       Sections assigned: PASS
#:     MODEL IS READY FOR ANALYSIS
#: === MODEL SETUP COMPLETED (0.3s total) ===
#:   Model setup completed (0.3s)
#: Phase 2: Analysis step creation
#: === CREATING ANALYSIS STEP ===
#: Mass diffusion step created successfully (0.01s):
#:   name: Permeation
#:   timePeriod: 86400.0 s (24.0 hours)
#:   initialInc: 1.0 s
#:   maxInc: 3600.0 s (60.0 min)
#:   minInc: 1e-06 s
#:   dcmax: 1000.0
#:   Step verification: SUCCESS
#:   Analysis step creation completed (0.0s)
#: Phase 3: Surface creation
#: === CREATING SURFACES ===
#: Geometry parameters for surface creation:
#:   width: 10000.0 nm
#:   height: 750 nm
#: Creating surfaces:
#:   Creating Top-Surface: Inlet (y=height) at point (5000.0, 750, 0.0)
#:     SUCCESS: Top-Surface created with 1 edges
#:   Creating Bottom-Surface: Outlet (y=0) at point (5000.0, 0.0, 0.0)
#:     SUCCESS: Bottom-Surface created with 1 edges
#:   Creating Left-Surface: Periodic left (x=0) at point (0.0, 375.0, 0.0)
#:     SUCCESS: Left-Surface created with 1 edges
#:   Creating Right-Surface: Periodic right (x=width) at point (10000.0, 375.0, 0.0)
#:     SUCCESS: Right-Surface created with 1 edges
#: Surface creation summary:
#:   Successful surfaces: 4/4
#:     Top-Surface: SUCCESS (1 edges)
#:     Bottom-Surface: SUCCESS (1 edges)
#:     Left-Surface: SUCCESS (1 edges)
#:     Right-Surface: SUCCESS (1 edges)
#:   Surface creation completed (0.0s)
#: Phase 4: Boundary condition application
#: === APPLYING BOUNDARY CONDITIONS (NANOMETER UNITS) ===
#: Boundary condition parameters:
#:   domain height: 750.0 nm
#:   domain width: 10000.0 nm
#:   inlet concentration: 1.0
#:   outlet concentration: 0.0
#: Finding boundary nodes by coordinates:
#:   Total nodes in instance: 0
#:   Found 0 top nodes at y=750.0 nm
#:   Found 0 bottom nodes at y=0.0 nm
#:   ERROR: No top nodes found for inlet BC
#:   ERROR: No bottom nodes found for outlet BC
#: Boundary condition summary:
#:   Successful BCs: 0/2
#:     Inlet: FAILED (0 nodes)
#:     Outlet: FAILED (0 nodes)
#:   WARNING: Boundary conditions failed - will cause zero flux
#:   Boundary condition application completed (0.0s)
#: Generated job name: Job_SS_c100_d10000_o0p25
#: Phase 5: Job submission and execution
#: === SUBMITTING JOB ===
#: Job name: Job_SS_c100_d10000_o0p25
#: Creating ABAQUS job...
#:   Job created successfully (0.02s)
#:   Job description: PECVD barrier permeation analysis - Job_SS_c100_d10000_o0p25
#:   Memory allocation: 90 PERCENTAGE
#: Submitting job for execution...
#: :: WARNING: setvars.bat has already been run. Skipping re-execution.
#:    To force a re-execution of setvars.bat, use the '--force' option.
#:    Using '--force' can result in excessive use of your environment variables.
#:   Job submitted (21.95s)
#:   Job status: None
#: Waiting for job completion...
#:   Job execution completed in 0.0s (0.0 minutes)
#:   Final job status: None
#:   WARNING: Job finished with unexpected status: None
#: Organizing output files...
#: Organizing files for job: Job_SS_c100_d10000_o0p25
#: Moved: Job_SS_c100_d10000_o0p25.odb -> abaqus_files\jobs\Job_SS_c100_d10000_o0p25\Job_SS_c100_d10000_o0p25.odb
#: Moved: Job_SS_c100_d10000_o0p25.sta -> abaqus_files\jobs\Job_SS_c100_d10000_o0p25\Job_SS_c100_d10000_o0p25.sta
#: Moved: Job_SS_c100_d10000_o0p25.msg -> abaqus_files\jobs\Job_SS_c100_d10000_o0p25\Job_SS_c100_d10000_o0p25.msg
#: Moved: Job_SS_c100_d10000_o0p25.dat -> abaqus_files\jobs\Job_SS_c100_d10000_o0p25\Job_SS_c100_d10000_o0p25.dat
#: Moved: Job_SS_c100_d10000_o0p25.prt -> abaqus_files\jobs\Job_SS_c100_d10000_o0p25\Job_SS_c100_d10000_o0p25.prt
#: Moved: Job_SS_c100_d10000_o0p25.com -> abaqus_files\jobs\Job_SS_c100_d10000_o0p25\Job_SS_c100_d10000_o0p25.com
#: Moved: Job_SS_c100_d10000_o0p25.log -> abaqus_files\jobs\Job_SS_c100_d10000_o0p25\Job_SS_c100_d10000_o0p25.log
#: Moved: Job_SS_c100_d10000_o0p25.inp -> abaqus_files\jobs\Job_SS_c100_d10000_o0p25\Job_SS_c100_d10000_o0p25.inp
#:   File organization completed (0.06s)
#:   Organized 8 files
#:     -> Job_SS_c100_d10000_o0p25.odb
#:     -> Job_SS_c100_d10000_o0p25.sta
#:     -> Job_SS_c100_d10000_o0p25.msg
#:     -> Job_SS_c100_d10000_o0p25.dat
#:     -> Job_SS_c100_d10000_o0p25.prt
#:     -> Job_SS_c100_d10000_o0p25.com
#:     -> Job_SS_c100_d10000_o0p25.log
#:     -> Job_SS_c100_d10000_o0p25.inp
#: Starting post-processing...
#: === EXTRACTING AND ANALYZING RESULTS ===
#: Job: Job_SS_c100_d10000_o0p25
#: Looking for ODB at: ./abaqus_files/results/Job_SS_c100_d10000_o0p25.odb
#: ERROR: ODB file not found: ./abaqus_files/results/Job_SS_c100_d10000_o0p25.odb
#:   Post-processing completed (0.00s)
#:   Data extraction and analysis: FAILED
#: === JOB SUBMISSION COMPLETED ===
#: Total job time: 22.0s (0.4 minutes)
#:   Job execution completed (22.1s)
#: === SINGLE SIMULATION COMPLETED ===
#: Total simulation time: 22.4s (0.4 minutes)
#: Completed job: Job_SS_c100_d10000_o0p25
#: Default simulation completed successfully: Job_SS_c100_d10000_o0p25
#: Total main execution time: 22.4s (0.4 minutes)
#: === MAIN EXECUTION COMPLETED ===
print('RT script done')
#: RT script done
