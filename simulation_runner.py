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
        """Apply concentration BCs to top and bottom surfaces"""
        log_message("Applying boundary conditions...", self.log_file)
        
        if not ABAQUS_ENV:
            return
        
        try:
            height = self.geometry.total_height
            width = self.geometry.crack_spacing
            tolerance = 1e-6  # nm
            
            assembly = self.model.rootAssembly
            
            # Method 1: Use getByBoundingBox (more reliable)
            try:
                # Select TOP nodes using bounding box
                top_nodes = instance.nodes.getByBoundingBox(
                    xMin=-tolerance, 
                    xMax=width+tolerance,
                    yMin=height-tolerance, 
                    yMax=height+tolerance,
                    zMin=-tolerance, 
                    zMax=tolerance
                )
                
                # Select BOTTOM nodes using bounding box  
                bottom_nodes = instance.nodes.getByBoundingBox(
                    xMin=-tolerance, 
                    xMax=width+tolerance,
                    yMin=-tolerance, 
                    yMax=tolerance,
                    zMin=-tolerance, 
                    zMax=tolerance
                )
                
                log_message("  Found {} top nodes, {} bottom nodes".format(
                    len(top_nodes), len(bottom_nodes)), self.log_file)
                
                # Apply inlet BC (top)
                if top_nodes:
                    assembly.Set(nodes=top_nodes, name='TopNodes')
                    self.model.ConcentrationBC(
                        name='Inlet',
                        createStepName='Permeation',
                        region=assembly.sets['TopNodes'],
                        magnitude=self.inlet_concentration
                    )
                    log_message("  Inlet BC applied (C=1.0)", self.log_file)
                
                # Apply outlet BC (bottom)
                if bottom_nodes:
                    assembly.Set(nodes=bottom_nodes, name='BottomNodes')
                    self.model.ConcentrationBC(
                        name='Outlet',
                        createStepName='Permeation',
                        region=assembly.sets['BottomNodes'],
                        magnitude=self.outlet_concentration
                    )
                    log_message("  Outlet BC applied (C=0.0)", self.log_file)
                
                log_message("  Flow direction: TOP -> BOTTOM (vertical)", self.log_file)
                
            except Exception as e:
                # Fallback: Try using edges instead of nodes
                log_message("  Node selection failed, trying edge-based BCs", self.log_file)
                
                # Find edges at top and bottom
                top_edge = instance.edges.findAt(((width/2, height, 0.0), ))
                bottom_edge = instance.edges.findAt(((width/2, 0.0, 0.0), ))
                
                if top_edge:
                    assembly.Surface(side1Edges=top_edge, name='TopSurface')
                    # Note: For edges, you might need to use a different BC type
                    # or convert to nodes differently
                
                raise e
                
        except Exception as e:
            log_message("ERROR applying BCs: {}".format(str(e)), self.log_file)
            raise

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