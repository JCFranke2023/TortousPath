"""
ABAQUS simulation runner for PECVD barrier coating permeation analysis
Handles job submission, parameter sweeps, and boundary conditions
"""

import os
import sys
import json
import time
from material_properties import materials
from geometry_generator import geometry

try:
    # ABAQUS imports
    from abaqus import *
    from abaqusConstants import *
    import step
    import load
    import job
    ABAQUS_ENV = True
except ImportError:
    ABAQUS_ENV = False
    print("Running in development mode - ABAQUS imports not available")

class SimulationRunner:
    def __init__(self, model_name='PermeationModel'):
        self.model_name = model_name
        if ABAQUS_ENV:
            # Create or get existing model
            if model_name in mdb.models:
                del mdb.models[model_name]
            self.model = mdb.Model(name=model_name)
        
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
        
        # Set geometry parameters (single-sided only for now)
        geometry.set_parameters(
            crack_width=crack_width,
            crack_spacing=crack_spacing, 
            crack_offset=crack_offset,
            single_sided=True
        )
        
        # Create geometry and materials
        geometry.create_materials()
        part = geometry.create_unit_cell_geometry()
        geometry.assign_sections(part)
        
        # Create assembly
        assembly = self.model.rootAssembly
        instance = assembly.Instance(name='UnitCell-1', part=part, dependent=ON)
        
        # Create mesh
        geometry.create_mesh(part)
        
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
            minInc=1e-6
        )
        
        # Request field and history outputs
        self.model.fieldOutputRequests['F-Output-1'].setValues(
            variables=('CONC', 'MFL'),  # Concentration and mass flux
            timeInterval=3600.0  # Output every hour
        )
        
        self.model.HistoryOutputRequest(
            name='H-Flux',
            createStepName='Permeation', 
            variables=('MFL',),
            region=None  # Will be set to outlet surface
        )

    def apply_boundary_conditions(self, instance):
        """Apply concentration boundary conditions and periodic constraints"""
        if not ABAQUS_ENV:
            print("Would apply boundary conditions:")
            print("  Top (inlet): C = {:.1f}".format(self.inlet_concentration))
            print("  Bottom (outlet): C = {:.1f}".format(self.outlet_concentration))
            return
            
        # Get surfaces
        assembly = self.model.rootAssembly
        
        # Inlet boundary condition (top surface)
        top_surface = instance.surfaces['Top-Surface']
        self.model.ConcentrationBC(
            name='Inlet',
            createStepName='Permeation',
            region=top_surface,
            magnitude=self.inlet_concentration
        )
        
        # Outlet boundary condition (bottom surface)
        bottom_surface = instance.surfaces['Bottom-Surface']
        self.model.ConcentrationBC(
            name='Outlet',
            createStepName='Permeation',
            region=bottom_surface,
            magnitude=self.outlet_concentration
        )
        
        # Periodic boundary conditions (left/right surfaces)
        left_surface = instance.surfaces['Left-Surface'] 
        right_surface = instance.surfaces['Right-Surface']
        
        # Create equation constraint for periodic conditions
        self.model.Equation(
            name='Periodic-Conc',
            terms=((1.0, left_surface, 8), (-1.0, right_surface, 8))  # DOF 8 = concentration
        )

    def create_surfaces(self, instance):
        """Create named surfaces for boundary conditions"""
        if not ABAQUS_ENV:
            print("Would create boundary surfaces")
            return
            
        assembly = self.model.rootAssembly
        part = instance.part
        
        # Get total height for surface identification
        height = geometry.total_height
        width = geometry.crack_spacing
        
        # Top surface (y = total_height) - inlet
        top_edges = part.edges.findAt(((width/2, height, 0.0), ))
        assembly.Surface(side1Edges=top_edges, name='Top-Surface')
        
        # Bottom surface (y = 0) - outlet
        bottom_edges = part.edges.findAt(((width/2, 0.0, 0.0), ))
        assembly.Surface(side1Edges=bottom_edges, name='Bottom-Surface')
        
        # Left surface (x = 0) - periodic
        left_edges = part.edges.findAt(((0.0, height/2, 0.0), ))
        assembly.Surface(side1Edges=left_edges, name='Left-Surface')
        
        # Right surface (x = width) - periodic  
        right_edges = part.edges.findAt(((width, height/2, 0.0), ))
        assembly.Surface(side1Edges=right_edges, name='Right-Surface')

    def submit_job(self, job_name='Permeation_Job'):
        """Submit ABAQUS job for analysis"""
        if not ABAQUS_ENV:
            print("Would submit job: {}".format(job_name))
            return job_name
            
        # Create job
        job = mdb.Job(
            name=job_name,
            model=self.model_name,
            description='PECVD barrier permeation analysis'
        )
        
        # Submit job
        job.submit()
        
        # Wait for completion
        job.waitForCompletion()
        
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

# Create global instance
runner = SimulationRunner()

# Command line interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run PECVD barrier permeation simulation')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--sweep', help='Parameter sweep configuration file')
    
    args = parser.parse_args()
    
    if args.sweep:
        # Run parameter sweep
        config = runner.load_config(args.sweep)
        results = runner.run_parameter_sweep(config)
        print("Parameter sweep completed. {} jobs submitted.".format(len(results)))
        
    elif args.config:
        # Run single simulation
        config = runner.load_config(args.config)
        job_name = runner.run_single_simulation(config)
        print("Single simulation completed: {}".format(job_name))
        
    else:
        # Run default case
        default_params = {
            'crack_width': 100e-9,
            'crack_spacing': 10e-6,
            'crack_offset': 0.25
        }
        job_name = runner.run_single_simulation(default_params)
        print("Default simulation completed: {}".format(job_name))
        