# -*- coding: utf-8 -*-
"""
ABAQUS simulation runner for PECVD barrier coating permeation analysis
Handles job submission, parameter sweeps, and boundary conditions
Complete version with comprehensive file logging and robust error handling
"""

import os
import sys
import json
import time
from pathlib import Path
from file_organizer import ABQFileOrganizer
from material_properties import materials
from geometry_generator import GeometryGenerator

# Create logging directory and file
def setup_logging():
    """Setup logging to file"""
    log_dir = Path('simulation_results') / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'simulation_runner.log'
    return log_file

def log_message(message, log_file=None):
    """Log message to file"""
    if log_file is None:
        log_file = setup_logging()
    
    with open(log_file, 'a') as f:
        f.write("{}\n".format(message))
    
    # Also print to console for immediate feedback
    print(message)

# Initialize logging
LOG_FILE = setup_logging()
# Clear previous log
if LOG_FILE.exists():
    LOG_FILE.unlink()

log_message("=== SIMULATION RUNNER STARTED ===", LOG_FILE)

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
    log_message("ABAQUS environment detected", LOG_FILE)
except ImportError:
    ABAQUS_ENV = False
    log_message("Running in development mode - ABAQUS imports not available", LOG_FILE)

class SimulationRunner:
    def __init__(self, model_name='PermeationModel'):
        self.model_name = model_name
        self.log_file = LOG_FILE
        
        log_message("SimulationRunner initialized with model: {}".format(model_name), self.log_file)
        
        # Setup file organization
        self.file_organizer = ABQFileOrganizer()
        self.file_organizer.create_directory_structure()
        log_message("File organization structure created", self.log_file)
        
        # Initialize ABAQUS model and geometry generator
        if ABAQUS_ENV:
            # Create or get existing model
            if model_name in mdb.models:
                log_message("Deleting existing model: {}".format(model_name), self.log_file)
                del mdb.models[model_name]
            self.model = mdb.Model(name=model_name)
            log_message("Created new ABAQUS model: {}".format(model_name), self.log_file)
            
            # Initialize geometry generator with the model
            self.geometry = GeometryGenerator(self.model)
            log_message("GeometryGenerator initialized with model", self.log_file)
        else:
            self.model = None
            self.geometry = GeometryGenerator()
            log_message("GeometryGenerator initialized without model (dev mode)", self.log_file)
        
        # Simulation parameters
        self.total_time = 86400.0  # 24 hours (seconds)
        self.initial_increment = 1.0
        self.max_increment = 3600.0  # 1 hour max increment
        self.min_increment = 1e-6
        self.dcmax = 1000.0  # Maximum allowable concentration change per increment
        
        # Boundary conditions
        self.inlet_concentration = 1.0    # Normalized inlet concentration
        self.outlet_concentration = 0.0   # Outlet (sink condition)
        
        # Log simulation parameters
        log_message("Simulation parameters initialized:", self.log_file)
        log_message("  total_time: {} s ({:.1f} hours)".format(self.total_time, self.total_time/3600), self.log_file)
        log_message("  initial_increment: {} s".format(self.initial_increment), self.log_file)
        log_message("  max_increment: {} s ({:.1f} min)".format(self.max_increment, self.max_increment/60), self.log_file)
        log_message("  min_increment: {} s".format(self.min_increment), self.log_file)
        log_message("  dcmax: {}".format(self.dcmax), self.log_file)
        log_message("  inlet_concentration: {}".format(self.inlet_concentration), self.log_file)
        log_message("  outlet_concentration: {}".format(self.outlet_concentration), self.log_file)
        
    def setup_model(self, crack_width=100.0, crack_spacing=10000.0, crack_offset=0.25):
        """Create complete model with geometry, materials, and mesh - ALL NANOMETER UNITS"""
        log_message("=== SETTING UP MODEL (NANOMETER UNITS) ===", self.log_file)
        log_message("Input parameters (all in nanometers):", self.log_file)
        log_message("  crack_width: {:.1f} nm".format(crack_width), self.log_file)
        log_message("  crack_spacing: {:.1f} nm ({:.1f} um)".format(crack_spacing, crack_spacing/1000), self.log_file)
        log_message("  crack_offset: {:.2f}".format(crack_offset), self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would setup model in ABAQUS environment", self.log_file)
            return None
        
        setup_start_time = time.time()
        
        try:
            # Step 1: Set geometry parameters (all in nanometers)
            log_message("Step 1: Setting geometry parameters...", self.log_file)
            step_start = time.time()
            
            self.geometry.set_parameters(
                crack_width=crack_width,  # nm
                crack_spacing=crack_spacing,  # nm
                crack_offset=crack_offset,
                single_sided=True
            )
            
            step_time = time.time() - step_start
            log_message("  Geometry parameters set successfully ({:.2f}s)".format(step_time), self.log_file)
            
            # Step 2: Create materials
            log_message("Step 2: Creating materials...", self.log_file)
            step_start = time.time()
            
            self.geometry.create_materials()
            self._verify_materials_created()
            
            step_time = time.time() - step_start
            log_message("  Materials created successfully ({:.2f}s)".format(step_time), self.log_file)
            
            # Step 3: Create sections
            log_message("Step 3: Creating sections...", self.log_file)
            step_start = time.time()
            
            self.geometry.assign_sections()
            self._verify_sections_created()
            
            step_time = time.time() - step_start
            log_message("  Sections created successfully ({:.2f}s)".format(step_time), self.log_file)
            
            # Step 4: Create geometry with partitions
            log_message("Step 4: Creating unit cell geometry...", self.log_file)
            step_start = time.time()
            
            part = self.geometry.create_unit_cell_geometry()
            if part:
                log_message("  Unit cell geometry created: {} faces, {} edges, {} vertices".format(
                    len(part.faces), len(part.edges), len(part.vertices)), self.log_file)
                self._log_geometry_details(part)
            else:
                log_message("  ERROR: Unit cell geometry creation failed", self.log_file)
                return None
            
            step_time = time.time() - step_start
            log_message("  Geometry creation completed ({:.2f}s)".format(step_time), self.log_file)
            
            # Step 5: Create assembly instance
            log_message("Step 5: Creating assembly instance...", self.log_file)
            step_start = time.time()
            
            assembly = self.model.rootAssembly
            instance = assembly.Instance(name='UnitCell-1', part=part, dependent=ON)
            log_message("  Assembly instance created: UnitCell-1", self.log_file)
            
            step_time = time.time() - step_start
            log_message("  Assembly creation completed ({:.2f}s)".format(step_time), self.log_file)
            
            # Step 6: Create mesh
            log_message("Step 6: Creating mesh...", self.log_file)
            step_start = time.time()
            
            self.geometry.create_mesh(part)
            log_message("  Mesh created successfully", self.log_file)
            
            step_time = time.time() - step_start
            log_message("  Mesh creation completed ({:.2f}s)".format(step_time), self.log_file)
            
            # Step 7: Assign materials to regions
            log_message("Step 7: Assigning materials to regions...", self.log_file)
            step_start = time.time()
            
            self.geometry.assign_materials_to_regions(part, instance)
            self._verify_section_assignments(part)
            
            step_time = time.time() - step_start
            log_message("  Material assignment completed ({:.2f}s)".format(step_time), self.log_file)
            
            # Step 8: Final model verification
            log_message("Step 8: Final model verification...", self.log_file)
            self._verify_complete_model(part, instance)
            
            setup_time = time.time() - setup_start_time
            log_message("=== MODEL SETUP COMPLETED ({:.1f}s total) ===".format(setup_time), self.log_file)
            return instance
            
        except Exception as e:
            setup_time = time.time() - setup_start_time
            log_message("ERROR in setup_model after {:.1f}s: {}".format(setup_time, str(e)), self.log_file)
            import traceback
            log_message("Full traceback:", self.log_file)
            log_message(traceback.format_exc(), self.log_file)
            raise

    def _verify_materials_created(self):
        """Verify that all materials were created successfully"""
        log_message("  Verifying materials creation:", self.log_file)
        
        expected_materials = list(materials.diffusivities.keys())
        created_materials = list(self.model.materials.keys())
        
        log_message("    Expected materials: {}".format(expected_materials), self.log_file)
        log_message("    Created materials: {}".format(created_materials), self.log_file)
        
        missing_materials = set(expected_materials) - set(created_materials)
        if missing_materials:
            log_message("    WARNING: Missing materials: {}".format(list(missing_materials)), self.log_file)
        else:
            log_message("    All materials created successfully", self.log_file)
            
        # Verify material properties
        for mat_name in created_materials:
            mat = self.model.materials[mat_name]
            diffusivity = materials.get_diffusivity(mat_name)
            solubility = materials.get_solubility(mat_name)
            log_message("    {}: D={:.1e} m²/s, S={:.1e} mol/m³/Pa".format(
                mat_name, diffusivity, solubility), self.log_file)
    
    def _verify_sections_created(self):
        """Verify that all sections were created successfully"""
        log_message("  Verifying sections creation:", self.log_file)
        
        expected_sections = [mat_name + '_section' for mat_name in materials.diffusivities.keys()]
        created_sections = list(self.model.sections.keys())
        
        log_message("    Expected sections: {}".format(expected_sections), self.log_file)
        log_message("    Created sections: {}".format(created_sections), self.log_file)
        
        missing_sections = set(expected_sections) - set(created_sections)
        if missing_sections:
            log_message("    WARNING: Missing sections: {}".format(list(missing_sections)), self.log_file)
        else:
            log_message("    All sections created successfully", self.log_file)
    
    def _log_geometry_details(self, part):
        """Log detailed geometry information - FIXED VERSION"""
        log_message("  Geometry details:", self.log_file)
        log_message("    Total faces: {}".format(len(part.faces)), self.log_file)
        log_message("    Total edges: {}".format(len(part.edges)), self.log_file)
        log_message("    Total vertices: {}".format(len(part.vertices)), self.log_file)
        
        # Log edge information (first 10 edges) - coordinates in nm
        log_message("    Edges (first 10, coordinates in nm):", self.log_file)
        for i, edge in enumerate(part.edges[:10]):
            try:
                vertices = edge.getVertices()
                if len(vertices) >= 2:
                    v1 = part.vertices[vertices[0]]
                    v2 = part.vertices[vertices[1]]
                    coord1 = v1.pointOn[0]
                    coord2 = v2.pointOn[0]
                    log_message("      Edge {}: ({:.1f},{:.1f}) to ({:.1f},{:.1f}) nm".format(
                        i, coord1[0], coord1[1], coord2[0], coord2[1]), self.log_file)
            except Exception as e:
                log_message("      Edge {}: error getting coordinates - {}".format(i, str(e)), self.log_file)
        
        if len(part.edges) > 10:
            log_message("      ... ({} more edges)".format(len(part.edges) - 10), self.log_file)    

    def _verify_mesh_created(self, part):
        """Verify that mesh was created successfully"""
        log_message("  Verifying mesh creation:", self.log_file)
        
        try:
            # Check if mesh exists
            mesh = part.getMesh()
            
            if mesh:
                num_elements = len(mesh.elements)
                num_nodes = len(mesh.nodes)
                
                log_message("    Mesh created successfully:", self.log_file)
                log_message("      Elements: {}".format(num_elements), self.log_file)
                log_message("      Nodes: {}".format(num_nodes), self.log_file)
                
                # Verify element types
                if num_elements > 0:
                    first_element = mesh.elements[0]
                    element_type = first_element.type
                    log_message("      Element type: {}".format(element_type), self.log_file)
                    
                    if element_type != DC2D4:
                        log_message("      WARNING: Expected DC2D4 elements, got {}".format(element_type), self.log_file)
                    else:
                        log_message("      Element type correct: DC2D4 (mass diffusion)", self.log_file)
                        
                # Check mesh quality metrics if available
                try:
                    # Basic mesh statistics
                    min_size = min([elem.getSize() for elem in mesh.elements[:100]])  # Sample first 100
                    max_size = max([elem.getSize() for elem in mesh.elements[:100]])
                    log_message("      Element size range: {:.1e} to {:.1e}".format(min_size, max_size), self.log_file)
                except Exception as e:
                    log_message("      Could not determine element sizes: {}".format(str(e)), self.log_file)
                    
            else:
                log_message("    ERROR: No mesh found", self.log_file)
                
        except Exception as e:
            log_message("    ERROR verifying mesh: {}".format(str(e)), self.log_file)
    
    def _verify_section_assignments(self, part):
        """Verify that sections have been assigned to all elements"""
        log_message("  Verifying section assignments:", self.log_file)
        
        try:
            total_faces = len(part.faces)
            section_assignments = part.sectionAssignments
            
            log_message("    Total faces: {}".format(total_faces), self.log_file)
            log_message("    Section assignments: {}".format(len(section_assignments)), self.log_file)
            
            assigned_faces = 0
            section_counts = {}
            
            for i, assignment in enumerate(section_assignments):
                section_name = assignment.sectionName
                region = assignment.region
                
                # Count faces in this assignment
                if region and len(region) > 0:
                    faces_in_region = len(region[0]) if hasattr(region[0], '__len__') else 1
                    assigned_faces += faces_in_region
                    
                    if section_name not in section_counts:
                        section_counts[section_name] = 0
                    section_counts[section_name] += faces_in_region
                    
                    log_message("      Assignment {}: {} -> {} faces".format(
                        i, section_name, faces_in_region), self.log_file)
            
            log_message("    Section assignment summary:", self.log_file)
            log_message("      Total faces: {}".format(total_faces), self.log_file)
            log_message("      Assigned faces: {}".format(assigned_faces), self.log_file)
            log_message("      Unassigned faces: {}".format(total_faces - assigned_faces), self.log_file)
            
            for section_name, count in section_counts.items():
                percentage = (count / total_faces * 100) if total_faces > 0 else 0
                log_message("        {}: {} faces ({:.1f}%)".format(
                    section_name, count, percentage), self.log_file)
            
            if assigned_faces < total_faces:
                log_message("    WARNING: {} faces remain unassigned!".format(
                    total_faces - assigned_faces), self.log_file)
            else:
                log_message("    SUCCESS: All faces have section assignments", self.log_file)
                
        except Exception as e:
            log_message("    ERROR verifying section assignments: {}".format(str(e)), self.log_file)
    
    def _verify_complete_model(self, part, instance):
        """Perform final verification of complete model - FIXED VERSION"""
        log_message("  Final model verification:", self.log_file)
        
        try:
            # Check model components
            log_message("    Model components:", self.log_file)
            log_message("      Materials: {}".format(len(self.model.materials)), self.log_file)
            log_message("      Sections: {}".format(len(self.model.sections)), self.log_file)
            log_message("      Parts: {}".format(len(self.model.parts)), self.log_file)
            log_message("      Assembly instances: {}".format(len(self.model.rootAssembly.instances)), self.log_file)
            
            # Check if model is ready for analysis
            readiness_checks = {
                'Materials created': len(self.model.materials) > 0,
                'Sections created': len(self.model.sections) > 0,
                'Geometry created': len(part.faces) > 0,
                'Sections assigned': len(part.sectionAssignments) > 0
            }
            
            log_message("    Model readiness checks:", self.log_file)
            all_ready = True
            for check_name, check_result in readiness_checks.items():
                status = "PASS" if check_result else "FAIL"
                log_message("      {}: {}".format(check_name, status), self.log_file)
                if not check_result:
                    all_ready = False
            
            if all_ready:
                log_message("    MODEL IS READY FOR ANALYSIS", self.log_file)
            else:
                log_message("    WARNING: MODEL MAY NOT BE READY FOR ANALYSIS", self.log_file)
                
        except Exception as e:
            log_message("    ERROR in final verification: {}".format(str(e)), self.log_file)

    def create_analysis_step(self):
        """Create transient mass diffusion step"""
        log_message("=== CREATING ANALYSIS STEP ===", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create transient diffusion step in ABAQUS environment", self.log_file)
            return
        
        try:
            step_start_time = time.time()
            
            step_obj = self.model.MassDiffusionStep(
                name='Permeation',
                previous='Initial',
                timePeriod=self.total_time,
                initialInc=self.initial_increment,
                maxInc=self.max_increment,
                minInc=self.min_increment,
                dcmax=self.dcmax
            )
            
            step_time = time.time() - step_start_time
            
            log_message("Mass diffusion step created successfully ({:.2f}s):".format(step_time), self.log_file)
            log_message("  name: Permeation", self.log_file)
            log_message("  timePeriod: {} s ({:.1f} hours)".format(self.total_time, self.total_time/3600), self.log_file)
            log_message("  initialInc: {} s".format(self.initial_increment), self.log_file)
            log_message("  maxInc: {} s ({:.1f} min)".format(self.max_increment, self.max_increment/60), self.log_file)
            log_message("  minInc: {} s".format(self.min_increment), self.log_file)
            log_message("  dcmax: {}".format(self.dcmax), self.log_file)
            
            # Verify step was created
            if 'Permeation' in self.model.steps:
                log_message("  Step verification: SUCCESS", self.log_file)
            else:
                log_message("  Step verification: FAILED", self.log_file)
                
        except Exception as e:
            log_message("ERROR creating analysis step: {}".format(str(e)), self.log_file)
            import traceback
            log_message("Full traceback:", self.log_file)
            log_message(traceback.format_exc(), self.log_file)
            raise

    def create_surfaces(self, instance):
        """Create named surfaces for boundary conditions"""
        log_message("=== CREATING SURFACES ===", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create boundary surfaces in ABAQUS environment", self.log_file)
            return
        
        try:
            # Get geometry parameters
            height = self.geometry.total_height
            width = self.geometry.crack_spacing
            
            log_message("Geometry parameters for surface creation:", self.log_file)
            log_message("  width: {:.1f} nm".format(width, width), self.log_file)
            log_message("  height: {:.0f} nm".format(height, height), self.log_file)
            
            assembly = self.model.rootAssembly
            surfaces_created = {}
            
            # Define surface locations
            surface_definitions = {
                'Top-Surface': {'point': (width/2, height, 0.0), 'description': 'Inlet (y=height)'},
                'Bottom-Surface': {'point': (width/2, 0.0, 0.0), 'description': 'Outlet (y=0)'},
                'Left-Surface': {'point': (0.0, height/2, 0.0), 'description': 'Periodic left (x=0)'},
                'Right-Surface': {'point': (width, height/2, 0.0), 'description': 'Periodic right (x=width)'}
            }
            
            log_message("Creating surfaces:", self.log_file)
            
            for surface_name, surface_info in surface_definitions.items():
                point = surface_info['point']
                description = surface_info['description']
                
                try:
                    log_message("  Creating {}: {} at point {}".format(
                        surface_name, description, point), self.log_file)
                    
                    edges = instance.edges.findAt((point, ))
                    
                    if edges:
                        assembly.Surface(side1Edges=edges, name=surface_name)
                        surfaces_created[surface_name] = len(edges)
                        log_message("    SUCCESS: {} created with {} edges".format(
                            surface_name, len(edges)), self.log_file)
                    else:
                        log_message("    WARNING: No edges found at point {} for {}".format(
                            point, surface_name), self.log_file)
                        surfaces_created[surface_name] = 0
                        
                except Exception as e:
                    log_message("    ERROR creating {}: {}".format(surface_name, str(e)), self.log_file)
                    surfaces_created[surface_name] = 0
            
            # Summary
            log_message("Surface creation summary:", self.log_file)
            successful_surfaces = 0
            for count in surfaces_created.values():
                if count > 0:
                    successful_surfaces += 1
            log_message("  Successful surfaces: {}/{}".format(
                successful_surfaces, len(surface_definitions)), self.log_file)
            
            for surface_name, edge_count in surfaces_created.items():
                status = "SUCCESS" if edge_count > 0 else "FAILED"
                log_message("    {}: {} ({} edges)".format(surface_name, status, edge_count), self.log_file)
                
        except Exception as e:
            log_message("ERROR in create_surfaces: {}".format(str(e)), self.log_file)
            import traceback
            log_message("Full traceback:", self.log_file)
            log_message(traceback.format_exc(), self.log_file)
            raise

    def apply_boundary_conditions(self, instance):
        """Apply concentration boundary conditions - NANOMETER UNITS"""
        log_message("=== APPLYING BOUNDARY CONDITIONS (NANOMETER UNITS) ===", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would apply boundary conditions in ABAQUS environment", self.log_file)
            return
        
        try:
            # Get geometry parameters (all in nanometers)
            height = self.geometry.total_height  # nm
            width = self.geometry.crack_spacing  # nm
            
            log_message("Boundary condition parameters:", self.log_file)
            log_message("  domain height: {:.1f} nm".format(height), self.log_file)
            log_message("  domain width: {:.1f} nm".format(width), self.log_file)
            log_message("  inlet concentration: {}".format(self.inlet_concentration), self.log_file)
            log_message("  outlet concentration: {}".format(self.outlet_concentration), self.log_file)
            
            assembly = self.model.rootAssembly
            bc_results = {}
            
            log_message("Finding boundary nodes by coordinates:", self.log_file)
            log_message("  Total nodes in instance: {}".format(len(instance.nodes)), self.log_file)
            
            # Find boundary nodes
            top_nodes = []
            bottom_nodes = []
            tolerance = 1e-6  # nm tolerance
            
            for node in instance.nodes:
                coords = node.coordinates
                y_coord = coords[1]  # y-coordinate in nm
                
                # Check if node is at top boundary (y = height)
                if abs(y_coord - height) < tolerance:
                    top_nodes.append(node)
                
                # Check if node is at bottom boundary (y = 0)  
                elif abs(y_coord - 0.0) < tolerance:
                    bottom_nodes.append(node)
            
            log_message("  Found {} top nodes at y={:.1f} nm".format(len(top_nodes), height), self.log_file)
            log_message("  Found {} bottom nodes at y=0.0 nm".format(len(bottom_nodes)), self.log_file)
            
            # Apply inlet BC to top nodes
            if top_nodes:
                try:
                    assembly.Set(nodes=tuple(top_nodes), name='TopNodes')
                    top_region = assembly.sets['TopNodes']
                    
                    self.model.ConcentrationBC(
                        name='Inlet',
                        createStepName='Permeation',
                        region=top_region,
                        magnitude=self.inlet_concentration
                    )
                    bc_results['Inlet'] = len(top_nodes)
                    log_message("  Inlet BC applied successfully to {} nodes".format(len(top_nodes)), self.log_file)
                    
                except Exception as e:
                    bc_results['Inlet'] = 0
                    log_message("  ERROR applying inlet BC: {}".format(str(e)), self.log_file)
            else:
                bc_results['Inlet'] = 0
                log_message("  ERROR: No top nodes found for inlet BC", self.log_file)
            
            # Apply outlet BC to bottom nodes
            if bottom_nodes:
                try:
                    assembly.Set(nodes=tuple(bottom_nodes), name='BottomNodes')
                    bottom_region = assembly.sets['BottomNodes']
                    
                    self.model.ConcentrationBC(
                        name='Outlet',
                        createStepName='Permeation',
                        region=bottom_region,
                        magnitude=self.outlet_concentration
                    )
                    bc_results['Outlet'] = len(bottom_nodes)
                    log_message("  Outlet BC applied successfully to {} nodes".format(len(bottom_nodes)), self.log_file)
                    
                except Exception as e:
                    bc_results['Outlet'] = 0
                    log_message("  ERROR applying outlet BC: {}".format(str(e)), self.log_file)
            else:
                bc_results['Outlet'] = 0
                log_message("  ERROR: No bottom nodes found for outlet BC", self.log_file)
            
            # Summary
            log_message("Boundary condition summary:", self.log_file)
            successful_bcs = 0
            for count in bc_results.values():
                if count > 0:
                    successful_bcs += 1
            
            log_message("  Successful BCs: {}/2".format(successful_bcs), self.log_file)
            
            for bc_name, node_count in bc_results.items():
                status = "SUCCESS" if node_count > 0 else "FAILED"
                log_message("    {}: {} ({} nodes)".format(bc_name, status, node_count), self.log_file)
            
            if successful_bcs == 2:
                log_message("  All boundary conditions applied successfully!", self.log_file)
            else:
                log_message("  WARNING: Boundary conditions failed - will cause zero flux", self.log_file)
                
        except Exception as e:
            log_message("ERROR in apply_boundary_conditions: {}".format(str(e)), self.log_file)
            import traceback
            log_message("Full traceback:", self.log_file)
            log_message(traceback.format_exc(), self.log_file)
            raise

    def submit_job(self, job_name='Permeation_Job'):
        """Submit ABAQUS job for analysis"""
        log_message("=== SUBMITTING JOB ===", self.log_file)
        log_message("Job name: {}".format(job_name), self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would submit job in ABAQUS environment", self.log_file)
            return job_name
        
        try:
            job_start_time = time.time()
            
            # Create job
            log_message("Creating ABAQUS job...", self.log_file)
            analysis_job = mdb.Job(
                name=job_name,
                model=self.model_name,
                description='PECVD barrier permeation analysis - {}'.format(job_name),
                type=ANALYSIS,
                atTime=None,
                waitMinutes=0,
                waitHours=0,
                queue=None,
                memory=90,
                memoryUnits=PERCENTAGE,
                getMemoryFromAnalysis=True,
                explicitPrecision=SINGLE,
                nodalOutputPrecision=SINGLE,
                echoPrint=OFF,
                modelPrint=OFF,
                contactPrint=OFF,
                historyPrint=OFF,
                userSubroutine='',
                scratch='',
                resultsFormat=ODB
            )
            
            creation_time = time.time() - job_start_time
            log_message("  Job created successfully ({:.2f}s)".format(creation_time), self.log_file)
            log_message("  Job description: {}".format(analysis_job.description), self.log_file)
            log_message("  Memory allocation: {} {}".format(
                analysis_job.memory, 'PERCENTAGE' if analysis_job.memoryUnits == PERCENTAGE else 'MB'), self.log_file)
            
            # Submit job
            log_message("Submitting job for execution...", self.log_file)
            submit_start = time.time()
            analysis_job.submit()
            submit_time = time.time() - submit_start
            
            log_message("  Job submitted ({:.2f}s)".format(submit_time), self.log_file)
            log_message("  Job status: {}".format(analysis_job.status), self.log_file)
            
            # Wait for completion with periodic status updates
            log_message("Waiting for job completion...", self.log_file)
            wait_start = time.time()
            
            # Check status every 30 seconds
            last_status = None
            status_check_count = 0
            
            while analysis_job.status in [SUBMITTED, RUNNING]:
                time.sleep(30)
                status_check_count += 1
                current_time = time.time() - wait_start
                
                if analysis_job.status != last_status:
                    log_message("  Status update: {} (after {:.1f}s)".format(
                        analysis_job.status, current_time), self.log_file)
                    last_status = analysis_job.status
                
                # Log periodic updates every 5 minutes
                if status_check_count % 10 == 0:
                    log_message("  Still running... ({:.1f} minutes elapsed)".format(
                        current_time / 60), self.log_file)
            
            # Final wait (this blocks until completion)
            analysis_job.waitForCompletion()
            
            execution_time = time.time() - wait_start
            total_time = time.time() - job_start_time
            
            log_message("  Job execution completed in {:.1f}s ({:.1f} minutes)".format(
                execution_time, execution_time/60), self.log_file)
            log_message("  Final job status: {}".format(analysis_job.status), self.log_file)
            
            # Check completion status
            if analysis_job.status == COMPLETED:
                log_message("  Job completed successfully!", self.log_file)
            elif analysis_job.status == ABORTED:
                log_message("  WARNING: Job aborted!", self.log_file)
            else:
                log_message("  WARNING: Job finished with unexpected status: {}".format(
                    analysis_job.status), self.log_file)
            
            # Organize output files
            log_message("Organizing output files...", self.log_file)
            organize_start = time.time()
            
            moved_files = self.file_organizer.organize_job(job_name)
            
            organize_time = time.time() - organize_start
            log_message("  File organization completed ({:.2f}s)".format(organize_time), self.log_file)
            log_message("  Organized {} files".format(len(moved_files)), self.log_file)
            
            # List organized files
            for moved_file in moved_files:
                log_message("    -> {}".format(moved_file.name), self.log_file)

            # Extract and analyze results
            log_message("Starting post-processing...", self.log_file)
            post_start = time.time()
            
            extraction_success = self.extract_and_analyze_results(job_name)
            
            post_time = time.time() - post_start
            log_message("  Post-processing completed ({:.2f}s)".format(post_time), self.log_file)
            
            if extraction_success:
                log_message("  Data extraction and analysis: SUCCESS", self.log_file)
            else:
                log_message("  Data extraction and analysis: FAILED", self.log_file)
            
            log_message("=== JOB SUBMISSION COMPLETED ===", self.log_file)
            log_message("Total job time: {:.1f}s ({:.1f} minutes)".format(total_time, total_time/60), self.log_file)
            
            return job_name
            
        except Exception as e:
            total_time = time.time() - job_start_time
            log_message("ERROR in submit_job after {:.1f}s: {}".format(total_time, str(e)), self.log_file)
            import traceback
            log_message("Full traceback:", self.log_file)
            log_message(traceback.format_exc(), self.log_file)
            raise

    def run_single_simulation(self, parameters):
        """Run single simulation with given parameters - ALL NANOMETER UNITS"""
        log_message("=== STARTING SINGLE SIMULATION ===", self.log_file)
        
        crack_width = parameters.get('crack_width', 100.0)  # nm
        crack_spacing = parameters.get('crack_spacing', 10000.0)  # nm
        crack_offset = parameters.get('crack_offset', 0.25)
        
        log_message("Simulation parameters (all nanometers):", self.log_file)
        log_message("  crack_width: {:.1f} nm".format(crack_width), self.log_file)
        log_message("  crack_spacing: {:.1f} nm ({:.1f} um)".format(crack_spacing, crack_spacing/1000), self.log_file)
        log_message("  crack_offset: {:.2f} ({:.1%} of spacing)".format(crack_offset, crack_offset), self.log_file)
        
        # Calculate expected permeation characteristics
        log_message("Expected simulation characteristics:", self.log_file)
        crack_area_fraction = crack_width / crack_spacing
        log_message("  crack area fraction: {:.3%}".format(crack_area_fraction), self.log_file)
        
        simulation_start = time.time()
        
        try:
            # Setup model
            log_message("Phase 1: Model setup", self.log_file)
            setup_start = time.time()
            
            instance = self.setup_model(crack_width, crack_spacing, crack_offset)
            
            setup_time = time.time() - setup_start
            log_message("  Model setup completed ({:.1f}s)".format(setup_time), self.log_file)
            
            if not instance:
                log_message("ERROR: Model setup failed - aborting simulation", self.log_file)
                return None
            
            # Create analysis step
            log_message("Phase 2: Analysis step creation", self.log_file)
            step_start = time.time()
            
            self.create_analysis_step()
            
            step_time = time.time() - step_start
            log_message("  Analysis step creation completed ({:.1f}s)".format(step_time), self.log_file)
            
            # Create surfaces
            log_message("Phase 3: Surface creation", self.log_file)
            surface_start = time.time()
            
            self.create_surfaces(instance)
            
            surface_time = time.time() - surface_start
            log_message("  Surface creation completed ({:.1f}s)".format(surface_time), self.log_file)
            
            # Apply boundary conditions
            log_message("Phase 4: Boundary condition application", self.log_file)
            bc_start = time.time()
            
            self.apply_boundary_conditions(instance)
            
            bc_time = time.time() - bc_start
            log_message("  Boundary condition application completed ({:.1f}s)".format(bc_time), self.log_file)
            
            # Generate job name
            job_name = 'Job_SS_c{:.0f}_d{:.0f}_o{:.2f}'.format(
                crack_width, crack_spacing, crack_offset
            ).replace('.', 'p')
            
            log_message("Generated job name: {}".format(job_name), self.log_file)
            
            # Submit job
            log_message("Phase 5: Job submission and execution", self.log_file)
            job_start = time.time()
            
            result_job_name = self.submit_job(job_name)
            
            job_time = time.time() - job_start
            log_message("  Job execution completed ({:.1f}s)".format(job_time), self.log_file)
            
            simulation_time = time.time() - simulation_start
            
            log_message("=== SINGLE SIMULATION COMPLETED ===", self.log_file)
            log_message("Total simulation time: {:.1f}s ({:.1f} minutes)".format(
                simulation_time, simulation_time/60), self.log_file)
            log_message("Completed job: {}".format(result_job_name), self.log_file)
            
            return result_job_name
            
        except Exception as e:
            simulation_time = time.time() - simulation_start
            log_message("ERROR in run_single_simulation after {:.1f}s: {}".format(
                simulation_time, str(e)), self.log_file)
            import traceback
            log_message("Full traceback:", self.log_file)
            log_message(traceback.format_exc(), self.log_file)
            raise

    def extract_and_analyze_results(self, job_name):
        """Extract data from ODB and trigger analysis"""
        log_message("=== EXTRACTING AND ANALYZING RESULTS ===", self.log_file)
        log_message("Job: {}".format(job_name), self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would extract and analyze results in ABAQUS environment", self.log_file)
            return True
        
        try:
            extraction_start = time.time()
            
            # Construct ODB path
            odb_path = "./abaqus_files/results/{}.odb".format(job_name)
            log_message("Looking for ODB at: {}".format(odb_path), self.log_file)
            
            if not os.path.exists(odb_path):
                log_message("ERROR: ODB file not found: {}".format(odb_path), self.log_file)
                return False
            
            # Get ODB file size for reference
            odb_size = os.path.getsize(odb_path)
            log_message("ODB file found: {:.1f} MB".format(odb_size / (1024*1024)), self.log_file)
            
            # Run extractor
            log_message("Running ABAQUS data extractor...", self.log_file)
            from abaqus_data_extractor import extract_raw_data
            
            result_file = extract_raw_data(job_name, odb_path)
            
            extraction_time = time.time() - extraction_start
            
            if result_file:
                log_message("Data extraction successful ({:.2f}s): {}".format(
                    extraction_time, result_file), self.log_file)
                
                # Check extracted data file size
                if os.path.exists(result_file):
                    data_size = os.path.getsize(result_file)
                    log_message("  Extracted data size: {:.1f} KB".format(data_size / 1024), self.log_file)
                
                # Trigger Python analysis
                analysis_success = self._trigger_python_analysis(job_name)
                
                return analysis_success
            else:
                log_message("Data extraction failed ({:.2f}s)".format(extraction_time), self.log_file)
                return False
                
        except Exception as e:
            extraction_time = time.time() - extraction_start
            log_message("ERROR during extraction after {:.2f}s: {}".format(
                extraction_time, str(e)), self.log_file)
            import traceback
            log_message("Full traceback:", self.log_file)
            log_message(traceback.format_exc(), self.log_file)
            return False

    def _trigger_python_analysis(self, job_name):
        """Trigger Python analysis as subprocess"""
        log_message("Triggering Python analysis...", self.log_file)
        
        try:
            analysis_start = time.time()
            import subprocess
            
            # Run Python analyzer
            cmd = ["python", "python_data_analyzer.py", "--job", job_name]
            log_message("  Running command: {}".format(" ".join(cmd)), self.log_file)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            analysis_time = time.time() - analysis_start
            
            if result.returncode == 0:
                log_message("  Python analysis completed successfully ({:.2f}s)".format(analysis_time), self.log_file)
                if result.stdout:
                    # Log first few lines of output
                    output_lines = result.stdout.strip().split('\n')
                    log_message("  Analysis output (first 5 lines):", self.log_file)
                    for i, line in enumerate(output_lines[:5]):
                        log_message("    {}".format(line), self.log_file)
                    if len(output_lines) > 5:
                        log_message("    ... ({} more lines)".format(len(output_lines) - 5), self.log_file)
                return True
            else:
                log_message("  Python analysis failed ({:.2f}s) with return code: {}".format(
                    analysis_time, result.returncode), self.log_file)
                if result.stderr:
                    log_message("  Analysis error: {}".format(result.stderr), self.log_file)
                return False
                    
        except subprocess.TimeoutExpired:
            log_message("  ERROR: Python analysis timed out after 5 minutes", self.log_file)
            return False
        except Exception as e:
            log_message("  ERROR triggering Python analysis: {}".format(str(e)), self.log_file)
            return False

    def run_parameter_sweep(self, sweep_config):
        """Run parameter sweep based on configuration"""
        log_message("=== STARTING PARAMETER SWEEP ===", self.log_file)
        log_message("Sweep configuration:", self.log_file)
        for key, value in sweep_config.items():
            if isinstance(value, list):
                log_message("  {}: {} values - {}".format(key, len(value), 
                    ["{:.1e}".format(v) if isinstance(v, float) and v < 1 else str(v) for v in value[:3]] + 
                    (["..."] if len(value) > 3 else [])), self.log_file)
            else:
                log_message("  {}: {}".format(key, value), self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would run parameter sweep in ABAQUS environment", self.log_file)
            return []
            
        results = []
        sweep_start = time.time()
        
        # Generate parameter combinations
        crack_widths = sweep_config.get('crack_widths', [100])
        crack_spacings = sweep_config.get('crack_spacings', [10000])
        crack_offsets = sweep_config.get('crack_offsets', [0.0, 0.25, 0.5])
        
        total_runs = len(crack_widths) * len(crack_spacings) * len(crack_offsets)
        run_count = 0
        
        log_message("Parameter sweep details:", self.log_file)
        log_message("  Total simulations: {}".format(total_runs), self.log_file)
        log_message("  crack_widths: {} values".format(len(crack_widths)), self.log_file)
        log_message("  crack_spacings: {} values".format(len(crack_spacings)), self.log_file)
        log_message("  crack_offsets: {} values".format(len(crack_offsets)), self.log_file)
        
        # Estimate total time
        estimated_time_per_run = 5 * 60  # 5 minutes per run estimate
        estimated_total_time = total_runs * estimated_time_per_run
        log_message("  Estimated total time: {:.1f} hours".format(estimated_total_time / 3600), self.log_file)
        
        successful_runs = 0
        failed_runs = 0
        
        for c_width in crack_widths:
            for c_spacing in crack_spacings:
                for c_offset in crack_offsets:
                    run_count += 1
                    
                    log_message("=== PARAMETER SWEEP RUN {}/{} ===".format(run_count, total_runs), self.log_file)
                    log_message("Parameters: w={:.1e} m, s={:.1e} m, o={:.2f}".format(
                        c_width, c_spacing, c_offset), self.log_file)
                    
                    parameters = {
                        'crack_width': c_width,
                        'crack_spacing': c_spacing,
                        'crack_offset': c_offset
                    }
                    
                    run_start_time = time.time()
                    
                    try:
                        job_name = self.run_single_simulation(parameters)
                        run_end_time = time.time()
                        run_time = run_end_time - run_start_time
                        
                        results.append({
                            'run_number': run_count,
                            'parameters': parameters,
                            'job_name': job_name,
                            'run_time': run_time,
                            'success': True
                        })
                        
                        successful_runs += 1
                        elapsed_time = time.time() - sweep_start
                        
                        log_message("Run {}/{} completed successfully in {:.1f}s".format(
                            run_count, total_runs, run_time), self.log_file)
                        log_message("  Job: {}".format(job_name), self.log_file)
                        log_message("  Elapsed time: {:.1f}s ({:.1f} min)".format(
                            elapsed_time, elapsed_time/60), self.log_file)
                        
                        # Update time estimate
                        avg_time_per_run = elapsed_time / run_count
                        remaining_runs = total_runs - run_count
                        estimated_remaining = avg_time_per_run * remaining_runs
                        log_message("  Estimated remaining time: {:.1f} min".format(
                            estimated_remaining/60), self.log_file)
                        
                    except Exception as e:
                        run_end_time = time.time()
                        run_time = run_end_time - run_start_time
                        
                        results.append({
                            'run_number': run_count,
                            'parameters': parameters,
                            'job_name': None,
                            'run_time': run_time,
                            'success': False,
                            'error': str(e)
                        })
                        
                        failed_runs += 1
                        elapsed_time = time.time() - sweep_start
                        
                        log_message("Run {}/{} FAILED in {:.1f}s: {}".format(
                            run_count, total_runs, run_time, str(e)), self.log_file)
                        log_message("  Elapsed time: {:.1f}s ({:.1f} min)".format(
                            elapsed_time, elapsed_time/60), self.log_file)
        
        sweep_end_time = time.time()
        total_sweep_time = sweep_end_time - sweep_start
        
        log_message("=== PARAMETER SWEEP COMPLETED ===", self.log_file)
        log_message("Sweep statistics:", self.log_file)
        log_message("  Total runs: {}".format(total_runs), self.log_file)
        log_message("  Successful: {} ({:.1%})".format(successful_runs, successful_runs/total_runs), self.log_file)
        log_message("  Failed: {} ({:.1%})".format(failed_runs, failed_runs/total_runs), self.log_file)
        log_message("  Total time: {:.1f}s ({:.1f} hours)".format(total_sweep_time, total_sweep_time/3600), self.log_file)
        log_message("  Average time per run: {:.1f}s".format(total_sweep_time/total_runs), self.log_file)
        
        if successful_runs > 0:
            successful_times = [r['run_time'] for r in results if r['success']]
            avg_successful_time = sum(successful_times) / len(successful_times)
            log_message("  Average time per successful run: {:.1f}s".format(avg_successful_time), self.log_file)
        
        # Save sweep results
        sweep_results_file = Path('simulation_results') / 'logs' / 'parameter_sweep_results.json'
        try:
            with open(sweep_results_file, 'w') as f:
                json.dump({
                    'sweep_config': sweep_config,
                    'total_runs': total_runs,
                    'successful_runs': successful_runs,
                    'failed_runs': failed_runs,
                    'total_time': total_sweep_time,
                    'results': results
                }, f, indent=2)
            log_message("Sweep results saved to: {}".format(sweep_results_file), self.log_file)
        except Exception as e:
            log_message("Could not save sweep results: {}".format(str(e)), self.log_file)
        
        return results

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        log_message("Loading configuration from: {}".format(config_file), self.log_file)
        
        try:
            config_start = time.time()
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            config_time = time.time() - config_start
            log_message("Configuration loaded successfully ({:.3f}s):".format(config_time), self.log_file)
            
            for key, value in config.items():
                if isinstance(value, list) and len(value) > 5:
                    log_message("  {}: {} items - {}...".format(
                        key, len(value), str(value[:3])), self.log_file)
                else:
                    log_message("  {}: {}".format(key, value), self.log_file)
            
            return config
        except Exception as e:
            log_message("ERROR loading configuration: {}".format(str(e)), self.log_file)
            raise

    def cleanup_simulation_files(self):
        """Clean up all ABAQUS files in current directory"""
        log_message("=== CLEANING UP SIMULATION FILES ===", self.log_file)
        
        try:
            cleanup_start = time.time()
            
            if self.file_organizer:
                moved_files = self.file_organizer.cleanup_all_abaqus_files()
                
                cleanup_time = time.time() - cleanup_start
                log_message("File cleanup completed ({:.2f}s):".format(cleanup_time), self.log_file)
                log_message("  Files organized: {}".format(len(moved_files)), self.log_file)
                
                return moved_files
            else:
                log_message("No file organizer available", self.log_file)
                return []
                
        except Exception as e:
            log_message("ERROR during file cleanup: {}".format(str(e)), self.log_file)
            return []

    def get_job_results_path(self, job_name):
        """Get path to job results directory"""
        if self.file_organizer:
            results_path = self.file_organizer.abq_dir / 'jobs' / job_name
            log_message("Job results path: {}".format(results_path), self.log_file)
            return results_path
        return Path('.')

    def archive_job(self, job_name):
        """Archive completed job to reduce clutter"""
        log_message("Archiving job: {}".format(job_name), self.log_file)
        
        try:
            if self.file_organizer:
                archive_path = self.file_organizer.archive_completed_job(job_name)
                
                if archive_path:
                    log_message("Job archived successfully: {}".format(archive_path), self.log_file)
                else:
                    log_message("Job archival failed", self.log_file)
                
                return archive_path
            else:
                log_message("No file organizer available for archival", self.log_file)
                return None
                
        except Exception as e:
            log_message("ERROR archiving job: {}".format(str(e)), self.log_file)
            return None

# Create global instance
log_message("Creating global SimulationRunner instance...", LOG_FILE)
try:
    runner = SimulationRunner()
    log_message("SimulationRunner instance created successfully", LOG_FILE)
except Exception as e:
    log_message("ERROR creating SimulationRunner instance: {}".format(str(e)), LOG_FILE)
    import traceback
    log_message("Full traceback:", LOG_FILE)
    log_message(traceback.format_exc(), LOG_FILE)
    raise

# Command line interface
if __name__ == '__main__':
    log_message("=== MAIN EXECUTION STARTED ===", LOG_FILE)
    log_message("Starting PECVD barrier permeation simulation...", LOG_FILE)
    log_message("Units: nanometers, seconds", LOG_FILE)
    
    try:
        main_start_time = time.time()
        
        # Run default case - ALL IN NANOMETERS
        default_params = {
            'crack_width': 100.0,      # 100 nm crack width
            'crack_spacing': 10000.0,  # 10 μm = 10,000 nm crack spacing  
            'crack_offset': 0.25       # 25% offset between layers
        }
        
        log_message("Running default simulation with parameters:", LOG_FILE)
        for key, value in default_params.items():
            if 'width' in key or 'spacing' in key:
                log_message("  {}: {:.1f} nm".format(key, value), LOG_FILE)
            else:
                log_message("  {}: {}".format(key, value), LOG_FILE)
        
        job_name = runner.run_single_simulation(default_params)
        
        main_time = time.time() - main_start_time
        
        if job_name:
            log_message("Default simulation completed successfully: {}".format(job_name), LOG_FILE)
            log_message("Total main execution time: {:.1f}s ({:.1f} minutes)".format(
                main_time, main_time/60), LOG_FILE)
        else:
            log_message("Default simulation FAILED", LOG_FILE)
            log_message("Total main execution time: {:.1f}s".format(main_time), LOG_FILE)
            
    except Exception as e:
        main_time = time.time() - main_start_time
        log_message("ERROR in main execution after {:.1f}s: {}".format(main_time, str(e)), LOG_FILE)
        import traceback
        log_message("Full traceback:", LOG_FILE)
        log_message(traceback.format_exc(), LOG_FILE)
        raise
    
    log_message("=== MAIN EXECUTION COMPLETED ===", LOG_FILE)
