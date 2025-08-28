"""
ABAQUS simulation runner for PECVD barrier coating permeation analysis
Handles job submission, parameter sweeps, and boundary conditions
"""

import os
import sys
import json
import time
from pathlib import Path
from file_organizer import ABQFileOrganizer
from material_properties import materials
from geometry_generator import GeometryGenerator

try:
    # ABAQUS imports
    from abaqus import *
    from abaqusConstants import *
    from regionToolset import Region
    import step
    import load
    import job
    from mesh import ElemType
    ABAQUS_ENV = True
except ImportError:
    ABAQUS_ENV = False
    print("Running in development mode - ABAQUS imports not available")

class SimulationRunner:
    def __init__(self, model_name='PermeationModel'):
        self.model_name = model_name
        self.file_organizer = ABQFileOrganizer()
        self.file_organizer.create_directory_structure()
        if ABAQUS_ENV:
            # Create or get existing model
            if model_name in mdb.models:
                del mdb.models[model_name]
            self.model = mdb.Model(name=model_name)
            
            # Initialize geometry generator with the model
            self.geometry = GeometryGenerator(self.model)
        else:
            self.model = None
            self.geometry = GeometryGenerator()
        
        # Simulation parameters
        self.total_time = 86400.0  # 24 hours (seconds)
        self.initial_increment = 1.0
        self.max_increment = 3600.0  # 1 hour max increment
        
        # Boundary conditions
        self.inlet_concentration = 1.0    # Normalized inlet concentration
        self.outlet_concentration = 0.0   # Outlet (sink condition)
        
    def setup_model(self, crack_width=100e-9, crack_spacing=10e-6, crack_offset=0.25):
        """Create complete model with geometry, materials, and mesh"""
        if not ABAQUS_ENV:
            print("Would setup model with:")
            print("  crack_width: {:.1e}".format(crack_width))
            print("  crack_spacing: {:.1e}".format(crack_spacing))
            print("  crack_offset: {:.2f}".format(crack_offset))
            return
        
        # Set geometry parameters
        self.geometry.set_parameters(
            crack_width=crack_width,
            crack_spacing=crack_spacing, 
            crack_offset=crack_offset,
            single_sided=True
        )
        
        # Create materials first
        self.geometry.create_materials()
        
        # Create geometry
        part = self.geometry.create_unit_cell_geometry()
        
        # Create sections 
        self.geometry.assign_sections(part)
        
        # For now, assign a single material to get the simulation running
        # We'll add complex material assignment later
        pet_section = self.model.sections['PET_section']
        part.SectionAssignment(region=(part.faces,), sectionName='PET_section')
        print("Assigned PET material to entire geometry")
        
        # Create assembly
        assembly = self.model.rootAssembly
        instance = assembly.Instance(name='UnitCell-1', part=part, dependent=ON)
        
        # Create mesh
        self.geometry.create_mesh(part)
        
        return instance
            
    def create_analysis_step(self):
        """Create transient mass diffusion step"""
        if not ABAQUS_ENV:
            print("Would create transient diffusion step")
            return
            
        self.model.MassDiffusionStep(
            name='Permeation',
            previous='Initial',
            timePeriod=self.total_time,
            initialInc=self.initial_increment,
            maxInc=self.max_increment,
            minInc=1e-6,
            dcmax=1000.0  # Maximum allowable concentration change per increment
        )

    def create_surfaces(self, instance):
        """Create named surfaces for boundary conditions"""
        if not ABAQUS_ENV:
            print("Would create boundary surfaces")
            return
            
        # Get geometry parameters
        height = self.geometry.total_height
        width = self.geometry.crack_spacing
        
        # Get assembly for surface creation
        assembly = self.model.rootAssembly
        
        # Create surfaces by selecting edges
        # Top surface (y = total_height) - inlet
        top_edges = instance.edges.findAt(((width/2, height, 0.0), ))
        assembly.Surface(side1Edges=top_edges, name='Top-Surface')
        
        # Bottom surface (y = 0) - outlet
        bottom_edges = instance.edges.findAt(((width/2, 0.0, 0.0), ))
        assembly.Surface(side1Edges=bottom_edges, name='Bottom-Surface')
        
        # Left surface (x = 0) - periodic
        left_edges = instance.edges.findAt(((0.0, height/2, 0.0), ))
        assembly.Surface(side1Edges=left_edges, name='Left-Surface')
        
        # Right surface (x = width) - periodic  
        right_edges = instance.edges.findAt(((width, height/2, 0.0), ))
        assembly.Surface(side1Edges=right_edges, name='Right-Surface')

    def apply_boundary_conditions(self, instance):
        """Apply concentration boundary conditions and periodic constraints"""
        if not ABAQUS_ENV:
            print("Would apply boundary conditions:")
            print("  Top (inlet): C = {:.1f}".format(self.inlet_concentration))
            print("  Bottom (outlet): C = {:.1f}".format(self.outlet_concentration))
            return
            
        # Get geometry parameters
        height = self.geometry.total_height
        width = self.geometry.crack_spacing
        
        # Find edges by coordinate - use a simpler approach
        # Top edge (y = height)
        try:
            top_edges = instance.edges.findAt(((width/2, height, 0.0), ), ((width/4, height, 0.0), ))
            top_region = Region(side1Edges=top_edges)
            self.model.ConcentrationBC(
                name='Inlet',
                createStepName='Permeation',
                region=top_region,
                magnitude=self.inlet_concentration
            )
            print("Applied inlet BC to top edge")
        except:
            print("Warning: Could not find top edge for inlet BC")
        
        # Bottom edge (y = 0)
        try:
            bottom_edges = instance.edges.findAt(((width/2, 0.0, 0.0), ), ((width/4, 0.0, 0.0), ))
            bottom_region = Region(side1Edges=bottom_edges)
            self.model.ConcentrationBC(
                name='Outlet',
                createStepName='Permeation',
                region=bottom_region,
                magnitude=self.outlet_concentration
            )
            print("Applied outlet BC to bottom edge")
        except:
            print("Warning: Could not find bottom edge for outlet BC")    

    def submit_job(self, job_name='Permeation_Job'):
        """Submit ABAQUS job for analysis"""
        if not ABAQUS_ENV:
            print("Would submit job: {}".format(job_name))
            return job_name
            
        # Create job
        analysis_job = mdb.Job(
            name=job_name,
            model=self.model_name,
            description='PECVD barrier permeation analysis'
        )
        
        # Submit job
        analysis_job.submit()
        
        # Wait for completion
        analysis_job.waitForCompletion()
        
        print("Organizing output files for job: {}".format(job_name))
        moved_files = self.file_organizer.organize_job(job_name)
        print("Organized {} files".format(len(moved_files)))

        return job_name

    def run_single_simulation(self, parameters):
        """Run single simulation with given parameters"""
        crack_width = parameters.get('crack_width', 100e-9)
        crack_spacing = parameters.get('crack_spacing', 10e-6) 
        crack_offset = parameters.get('crack_offset', 0.25)
        
        print("Running single-sided simulation:")
        print("  Crack width: {:.1e} m".format(crack_width))
        print("  Crack spacing: {:.1e} m".format(crack_spacing))
        print("  Crack offset: {:.2f}".format(crack_offset))
        
        # Setup model
        instance = self.setup_model(crack_width, crack_spacing, crack_offset)
        self.create_analysis_step()
        self.create_surfaces(instance)
        self.apply_boundary_conditions(instance)
        
        # Generate job name
        job_name = 'Job_SS_c{:.0e}_d{:.0e}_o{:.2f}'.format(
            crack_width, crack_spacing, crack_offset
        ).replace('.', 'p').replace('+', '').replace('-', 'n')
        
        # Submit job
        return self.submit_job(job_name)

    def run_parameter_sweep(self, sweep_config):
        """Run parameter sweep based on configuration"""
        if not ABAQUS_ENV:
            print("Would run parameter sweep with config:")
            print(sweep_config)
            return []
            
        results = []
        
        # Generate parameter combinations
        crack_widths = sweep_config.get('crack_widths', [100e-9])
        crack_spacings = sweep_config.get('crack_spacings', [10e-6])
        crack_offsets = sweep_config.get('crack_offsets', [0.0, 0.25, 0.5])
        
        total_runs = len(crack_widths) * len(crack_spacings) * len(crack_offsets)
        run_count = 0
        
        for c_width in crack_widths:
            for c_spacing in crack_spacings:
                for c_offset in crack_offsets:
                    run_count += 1
                    print("Run {}/{}: w={:.1e}, s={:.1e}, o={:.2f}".format(
                        run_count, total_runs, c_width, c_spacing, c_offset
                    ))
                    
                    parameters = {
                        'crack_width': c_width,
                        'crack_spacing': c_spacing,
                        'crack_offset': c_offset
                    }
                    
                    job_name = self.run_single_simulation(parameters)
                    results.append({
                        'parameters': parameters,
                        'job_name': job_name
                    })
        
        return results

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            return json.load(f)
        
    def cleanup_simulation_files(self):
        """Clean up all ABAQUS files in current directory"""
        if self.file_organizer:
            return self.file_organizer.cleanup_all_abaqus_files()
        return []

    def get_job_results_path(self, job_name):
        """Get path to job results directory"""
        if self.file_organizer:
            return self.file_organizer.abq_dir / 'jobs' / job_name
        return Path('.')

    def archive_job(self, job_name):
        """Archive completed job to reduce clutter"""
        if self.file_organizer:
            return self.file_organizer.archive_completed_job(job_name)
        return None

# Create global instance
runner = SimulationRunner()

# Command line interface
if __name__ == '__main__':
    print("Starting PECVD barrier permeation simulation...")
    print("Units: nanometers, seconds")
    
    # Run default case - now in nanometers
    default_params = {
        'crack_width': 100,      # 100 nm crack width
        'crack_spacing': 10000,  # 10 μm = 10,000 nm crack spacing
        'crack_offset': 0.25     # 25% offset between layers
    }
    job_name = runner.run_single_simulation(default_params)
    print("Default simulation completed: {}".format(job_name))
