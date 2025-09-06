"""
ABAQUS simulation runner for PECVD barrier coating permeation analysis
Cleaned version with improved error handling and structure
"""

import os
import sys
import json
import time
from pathlib import Path
from file_organizer import ABQFileOrganizer
from material_properties import materials
from geometry_generator import GeometryGenerator

# Logging setup
def setup_logging():
    """Setup logging to file"""
    log_dir = Path('simulation_results') / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'simulation_runner.log'
    return log_file

def log_message(message, log_file=None):
    """Log message to file and console"""
    if log_file is None:
        log_file = setup_logging()
    
    with open(log_file, 'a') as f:
        f.write("{}\n".format(message))
    print(message)

# Initialize logging
LOG_FILE = setup_logging()
if LOG_FILE.exists():
    LOG_FILE.unlink()

log_message("=== SIMULATION RUNNER STARTED ===", LOG_FILE)

# ABAQUS imports
try:
    from abaqus import *
    from abaqusConstants import *
    from regionToolset import Region
    import step
    import load
    import job
    from mesh import ElemType
    ABAQUS_ENV = True
    log_message("ABAQUS environment detected", LOG_FILE)
except ImportError:
    ABAQUS_ENV = False
    log_message("Running in development mode", LOG_FILE)

class SimulationRunner:
    def __init__(self, model_name='PermeationModel'):
        self.model_name = model_name
        self.log_file = LOG_FILE
        
        log_message("Initializing SimulationRunner: {}".format(model_name), self.log_file)
        
        # Setup file organization
        self.file_organizer = ABQFileOrganizer()
        self.file_organizer.create_directory_structure()
        
        # Initialize model and geometry
        if ABAQUS_ENV:
            # Clean up existing model
            if model_name in mdb.models:
                del mdb.models[model_name]
                log_message("Deleted existing model: {}".format(model_name), self.log_file)
            
            self.model = mdb.Model(name=model_name)
            self.geometry = GeometryGenerator(self.model)
        else:
            self.model = None
            self.geometry = GeometryGenerator()
        
        # Simulation parameters (consistent units: nanometers, seconds)
        self.total_time = 86400.0        # 24 hours
        self.initial_increment = 1.0     # 1 second
        self.max_increment = 3600.0      # 1 hour
        self.min_increment = 1e-6        # 1 microsecond
        self.dcmax = 1000.0              # Max concentration change per increment
        
        # Boundary conditions
        self.inlet_concentration = 1.0   # Top boundary
        self.outlet_concentration = 0.0  # Bottom boundary

    def setup_model(self, crack_width=100.0, crack_spacing=10000.0, crack_offset=0.25):
        """Create complete model with geometry, materials, and mesh"""
        log_message("=== SETTING UP MODEL ===", self.log_file)
        log_message("Parameters (nm): width={:.1f}, spacing={:.1f}, offset={:.2f}".format(
            crack_width, crack_spacing, crack_offset), self.log_file)
        
        if not ABAQUS_ENV:
            return None
        
        try:
            # Set geometry parameters
            self.geometry.set_parameters(
                crack_width=crack_width,
                crack_spacing=crack_spacing,
                crack_offset=crack_offset,
                single_sided=True
            )
            
            # Create materials and sections
            self.geometry.create_materials()
            self.geometry.create_sections()
            
            # Create geometry
            part = self.geometry.create_unit_cell_geometry()
            if not part:
                raise ValueError("Failed to create geometry")
            
            log_message("Geometry created: {} faces".format(len(part.faces)), self.log_file)
            
            # Create assembly
            assembly = self.model.rootAssembly
            instance = assembly.Instance(name='UnitCell-1', part=part, dependent=ON)
            
            # Assign materials BEFORE meshing
            self.geometry.assign_materials_to_regions(part, instance)
            
            # Create mesh
            self.geometry.create_mesh(part)
            
            # CRITICAL: Regenerate assembly after meshing
            assembly.regenerate()
            instance = assembly.instances['UnitCell-1']
            
            log_message("Model setup completed", self.log_file)
            return instance
            
        except Exception as e:
            log_message("ERROR in setup_model: {}".format(str(e)), self.log_file)
            raise

    def create_analysis_step(self):
        """Create mass diffusion analysis step"""
        log_message("Creating analysis step...", self.log_file)
        
        if not ABAQUS_ENV:
            return
        
        try:
            self.model.MassDiffusionStep(
                name='Permeation',
                previous='Initial',
                timePeriod=self.total_time,
                initialInc=self.initial_increment,
                maxInc=self.max_increment,
                minInc=self.min_increment,
                dcmax=self.dcmax
            )
            log_message("  Mass diffusion step created", self.log_file)
            
            # Create output requests
            self._create_output_requests()
            
        except Exception as e:
            log_message("ERROR creating analysis step: {}".format(str(e)), self.log_file)
            raise

    def _create_output_requests(self):
        """Setup output requests for mass diffusion"""
        try:
            # Delete defaults
            if 'F-Output-1' in self.model.fieldOutputRequests:
                del self.model.fieldOutputRequests['F-Output-1']
            if 'H-Output-1' in self.model.historyOutputRequests:
                del self.model.historyOutputRequests['H-Output-1']
            
            # Create field output for concentration and flux
            self.model.FieldOutputRequest(
                name='F-Output-1',
                createStepName='Permeation',
                variables=('NNC', 'MFL', 'IVOL', 'COORD')
            )
            log_message("  Output requests configured", self.log_file)
            
        except Exception as e:
            log_message("  WARNING: Could not configure output requests: {}".format(str(e)), self.log_file)

    def apply_boundary_conditions(self, instance):
        """Apply ALL boundary conditions: concentration on top/bottom, periodic on left/right"""
        log_message("=== APPLYING ALL BOUNDARY CONDITIONS ===", self.log_file)
        
        if not ABAQUS_ENV:
            return
        
        try:
            height = self.geometry.total_height
            width = self.geometry.crack_spacing
            assembly = self.model.rootAssembly
            tolerance = 1.0  # nm
            
            # PART 1: CONCENTRATION BCs ON TOP AND BOTTOM
            log_message("Step 1: Applying concentration BCs (top/bottom)...", self.log_file)
            
            # Get TOP nodes (y = height) for INLET
            top_nodes = instance.nodes.getByBoundingBox(
                xMin=-tolerance,
                xMax=width+tolerance,
                yMin=height-tolerance,
                yMax=height+tolerance,
                zMin=-tolerance,
                zMax=tolerance
            )
            
            # Get BOTTOM nodes (y = 0) for OUTLET
            bottom_nodes = instance.nodes.getByBoundingBox(
                xMin=-tolerance,
                xMax=width+tolerance,
                yMin=-tolerance,
                yMax=tolerance,
                zMin=-tolerance,
                zMax=tolerance
            )
            
            # Apply TOP boundary condition (INLET, C=1)
            if top_nodes:
                assembly.Set(nodes=top_nodes, name='TopNodes')
                self.model.ConcentrationBC(
                    name='Inlet',
                    createStepName='Permeation',
                    region=assembly.sets['TopNodes'],
                    magnitude=self.inlet_concentration,
                    distributionType=UNIFORM
                )
                log_message("  TOP (INLET): y={:.1f} nm, C={:.1f}, {} nodes".format(
                    height, self.inlet_concentration, len(top_nodes)), self.log_file)
            
            # Apply BOTTOM boundary condition (OUTLET, C=0)
            if bottom_nodes:
                assembly.Set(nodes=bottom_nodes, name='BottomNodes')
                self.model.ConcentrationBC(
                    name='Outlet',
                    createStepName='Permeation',
                    region=assembly.sets['BottomNodes'],
                    magnitude=self.outlet_concentration,
                    distributionType=UNIFORM
                )
                log_message("  BOTTOM (OUTLET): y=0 nm, C={:.1f}, {} nodes".format(
                    self.outlet_concentration, len(bottom_nodes)), self.log_file)
            
            # PART 2: PERIODIC BCs ON LEFT AND RIGHT (OPTIONAL)
            log_message("Step 2: Handling lateral boundaries...", self.log_file)
            
            # Option A: True periodic BCs (complex but exact)
            apply_periodic = True  # Set to True if you want periodic BCs
            
            if apply_periodic:
                log_message("  Applying periodic boundary conditions on left/right...", self.log_file)
                
                # Find matching node pairs on left and right boundaries
                left_nodes_by_y = {}
                right_nodes_by_y = {}
                
                for node in instance.nodes:
                    x = node.coordinates[0]
                    y = node.coordinates[1]
                    
                    # Left boundary (x = 0)
                    if abs(x - 0.0) < tolerance:
                        y_key = round(y, 3)
                        left_nodes_by_y[y_key] = node
                    
                    # Right boundary (x = width)
                    elif abs(x - width) < tolerance:
                        y_key = round(y, 3)
                        right_nodes_by_y[y_key] = node
                
                # Create equation constraints for matched pairs
                paired_count = 0
                for y_key, left_node in left_nodes_by_y.items():
                    if y_key in right_nodes_by_y:
                        right_node = right_nodes_by_y[y_key]
                        
                        # Create sets for individual nodes
                        left_set_name = 'PBC_L_{}'.format(paired_count)
                        right_set_name = 'PBC_R_{}'.format(paired_count)
                        
                        assembly.Set(nodes=(left_node,), name=left_set_name)
                        assembly.Set(nodes=(right_node,), name=right_set_name)
                        
                        # Create equation: C_right = C_left
                        self.model.Equation(
                            name='PBC_{}'.format(paired_count),
                            terms=(
                                (1.0, right_set_name, 11),   # DOF 11 = concentration
                                (-1.0, left_set_name, 11)
                            )
                        )
                        paired_count += 1
                
                log_message("  Created {} periodic constraints".format(paired_count), self.log_file)
            
            else:
                # Option B: No-flux (insulated) boundaries - often appropriate for symmetric unit cells
                log_message("  Using no-flux (insulated) boundary conditions on left/right", self.log_file)
                log_message("  This is appropriate for symmetric unit cells where flux is parallel to boundaries", self.log_file)
                # No explicit BC needed - ABAQUS defaults to no-flux
            
            # SUMMARY
            log_message("=== BOUNDARY CONDITION SUMMARY ===", self.log_file)
            log_message("  TOP (y={:.1f} nm): C = {:.1f} (INLET)".format(height, self.inlet_concentration), self.log_file)
            log_message("  BOTTOM (y=0 nm): C = {:.1f} (OUTLET)".format(self.outlet_concentration), self.log_file)
            if apply_periodic:
                log_message("  LEFT/RIGHT: Periodic (C_left = C_right)", self.log_file)
            else:
                log_message("  LEFT/RIGHT: No-flux (insulated)", self.log_file)
            log_message("  Flow direction: VERTICAL (top to bottom)", self.log_file)
            
        except Exception as e:
            log_message("ERROR in apply_all_boundary_conditions: {}".format(str(e)), self.log_file)
            import traceback
            log_message(traceback.format_exc(), self.log_file)
            raise

    def verify_boundary_conditions(self):
        """Verify that boundary conditions are correctly applied"""
        log_message("=== VERIFYING BOUNDARY CONDITIONS ===", self.log_file)
        
        if not ABAQUS_ENV:
            return
        
        try:
            # Check all boundary conditions in the model
            if hasattr(self.model, 'boundaryConditions'):
                log_message("Found {} boundary conditions:".format(len(self.model.boundaryConditions)), self.log_file)
                
                for bc_name, bc in self.model.boundaryConditions.items():
                    log_message("  BC Name: {}".format(bc_name), self.log_file)
                    
                    if hasattr(bc, 'magnitude'):
                        log_message("    Magnitude: {}".format(bc.magnitude), self.log_file)
                    
                    if hasattr(bc, 'region'):
                        region = bc.region
                        if hasattr(region, 'nodes'):
                            # Get a sample of node coordinates to verify location
                            sample_nodes = list(region.nodes)[:5]  # First 5 nodes
                            y_coords = [n.coordinates[1] for n in sample_nodes]
                            y_avg = sum(y_coords) / len(y_coords) if y_coords else 0
                            log_message("    Average Y-coordinate of nodes: {:.1f} nm".format(y_avg), self.log_file)
                            
                            if y_avg > self.geometry.total_height * 0.9:
                                log_message("    -> This BC is at the TOP", self.log_file)
                            elif y_avg < self.geometry.total_height * 0.1:
                                log_message("    -> This BC is at the BOTTOM", self.log_file)
                            else:
                                log_message("    -> This BC is in the MIDDLE (unexpected!)", self.log_file)
            else:
                log_message("No boundary conditions found!", self.log_file)
                
        except Exception as e:
            log_message("Error verifying BCs: {}".format(str(e)), self.log_file)

    def submit_job(self, job_name):
        """Submit and monitor ABAQUS job"""
        log_message("Submitting job: {}".format(job_name), self.log_file)
        
        if not ABAQUS_ENV:
            return job_name
        
        try:
            # Check if CAE file already exists and delete it
            cae_file = "{}.cae".format(job_name)
            if os.path.exists(cae_file):
                try:
                    os.remove(cae_file)
                    log_message("  Removed existing CAE file", self.log_file)
                except:
                    # Try alternative name if file is locked
                    timestamp = int(time.time())
                    cae_file = "{}_{}.cae".format(job_name, timestamp)
                    log_message("  Using alternative CAE name: {}".format(cae_file), self.log_file)
            
            # Save model
            mdb.saveAs(pathName=cae_file)
            log_message("  Model saved: {}".format(cae_file), self.log_file)
            
            # Create and submit job
            analysis_job = mdb.Job(
                name=job_name,
                model=self.model_name,
                description='PECVD barrier coating permeation',
                type=ANALYSIS,
                memory=90,
                memoryUnits=PERCENTAGE,
                getMemoryFromAnalysis=True,
                echoPrint=OFF,
                modelPrint=OFF,
                contactPrint=OFF,
                historyPrint=OFF
            )
            
            analysis_job.submit()
            log_message("  Job submitted", self.log_file)
            
            # Wait for completion
            analysis_job.waitForCompletion()
            
            # Check results
            odb_file = "{}.odb".format(job_name)
            if os.path.exists(odb_file):
                odb_size = os.path.getsize(odb_file) / (1024*1024)
                log_message("  Job completed: ODB size = {:.1f} MB".format(odb_size), self.log_file)
                
                # Check status file
                sta_file = "{}.sta".format(job_name)
                if os.path.exists(sta_file):
                    with open(sta_file, 'r') as f:
                        if "COMPLETED SUCCESSFULLY" in f.read().upper():
                            log_message("  Analysis completed successfully", self.log_file)
                
                # Organize output files
                self._organize_job_files(job_name)
                return job_name
            else:
                log_message("  Job failed - no ODB created", self.log_file)
                return None
                
        except Exception as e:
            log_message("ERROR in submit_job: {}".format(str(e)), self.log_file)
            raise

    def _organize_job_files(self, job_name):
        """Move job files to organized structure"""
        try:
            files_to_move = []
            for ext in ['.odb', '.dat', '.msg', '.sta', '.inp', '.prt']:
                file_path = "{}{}".format(job_name, ext)
                if os.path.exists(file_path):
                    files_to_move.append(file_path)
            
            if files_to_move:
                job_dir = Path('abaqus_files') / 'jobs' / job_name
                job_dir.mkdir(parents=True, exist_ok=True)
                
                for file_path in files_to_move:
                    dest = job_dir / Path(file_path).name
                    os.rename(file_path, str(dest))
                
                log_message("  Organized {} files to {}".format(
                    len(files_to_move), job_dir), self.log_file)
                
        except Exception as e:
            log_message("  WARNING: Could not organize files: {}".format(str(e)), self.log_file)

    def run_single_simulation(self, parameters):
        """Run complete simulation with given parameters"""
        log_message("=== RUNNING SIMULATION ===", self.log_file)
        
        crack_width = parameters.get('crack_width', 100.0)
        crack_spacing = parameters.get('crack_spacing', 10000.0)
        crack_offset = parameters.get('crack_offset', 0.25)
        
        log_message("Parameters: w={:.1f}nm, s={:.1f}nm, o={:.2f}".format(
            crack_width, crack_spacing, crack_offset), self.log_file)
        
        try:
            # Setup model
            instance = self.setup_model(crack_width, crack_spacing, crack_offset)
            if not instance:
                raise ValueError("Model setup failed")
            
            # Create analysis step
            self.create_analysis_step()
            
            # Verify orientation
            self.verify_boundary_conditions()

            # Apply boundary conditions
            self.apply_boundary_conditions(instance)
            
            # Generate job name
            job_name = 'Job_c{:.0f}_s{:.0f}_o{:.0f}'.format(
                crack_width, crack_spacing, crack_offset*100
            )
            
            # Submit job
            result = self.submit_job(job_name)
            
            if result:
                log_message("=== SIMULATION COMPLETED ===", self.log_file)
                return result
            else:
                log_message("=== SIMULATION FAILED ===", self.log_file)
                return None
                
        except Exception as e:
            log_message("ERROR in simulation: {}".format(str(e)), self.log_file)
            import traceback
            log_message(traceback.format_exc(), self.log_file)
            return None

    def cleanup(self):
        """Clean up temporary files"""
        try:
            self.file_organizer.cleanup_all_abaqus_files()
            log_message("Cleanup completed", self.log_file)
        except Exception as e:
            log_message("Cleanup error: {}".format(str(e)), self.log_file)


# Main execution
if __name__ == '__main__':
    log_message("=== MAIN EXECUTION ===", LOG_FILE)
    
    try:
        # Initialize runner
        runner = SimulationRunner()
        
        # Default parameters (nanometers)
        default_params = {
            'crack_width': 100.0,      # 100 nm
            'crack_spacing': 10000.0,  # 10 μm
            'crack_offset': 0.25       # 25% offset
        }
        
        # Run simulation
        job_name = runner.run_single_simulation(default_params)
        
        if job_name:
            log_message("SUCCESS: {}".format(job_name), LOG_FILE)
        else:
            log_message("FAILED", LOG_FILE)
            
    except Exception as e:
        log_message("ERROR: {}".format(str(e)), LOG_FILE)
        import traceback
        log_message(traceback.format_exc(), LOG_FILE)
    
    log_message("=== COMPLETED ===", LOG_FILE)