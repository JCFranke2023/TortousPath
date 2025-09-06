"""
ABAQUS geometry generator for PECVD barrier coating permeation simulation
Creates 2D representative unit cell with periodic crack patterns
Cleaned version with proper crack material assignment
"""

import os
import sys
from pathlib import Path
from material_properties import materials

# Create logging directory and file
def setup_logging():
    """Setup logging to file"""
    log_dir = Path('simulation_results') / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'geometry_generator.log'
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

log_message("=== GEOMETRY GENERATOR STARTED ===", LOG_FILE)

try:
    # ABAQUS imports (will work when running in ABAQUS environment)
    from abaqus import *
    from abaqusConstants import *
    import part
    import material
    import section
    import assembly
    import mesh
    from mesh import ElemType
    ABAQUS_ENV = True
    log_message("ABAQUS environment detected", LOG_FILE)
except ImportError:
    # Development environment - create mock objects
    ABAQUS_ENV = False
    log_message("Running in development mode - ABAQUS imports not available", LOG_FILE)

class GeometryGenerator:
    def __init__(self, model=None):
        self.model = model
        self.log_file = LOG_FILE
        
        log_message("GeometryGenerator initialized", self.log_file)
        
        # Geometry parameters (will be set via configuration)
        self.crack_width = 100.0      # c: crack width (nm)
        self.crack_spacing = 10000.0  # d: crack spacing (nm) 
        self.crack_offset = 0.25      # o: offset fraction (0-1, periodic)
        self.single_sided = False     # coating configuration
        
        # Layer thicknesses from materials (bottom to top) - all in nm
        self.h0_substrate = materials.get_thickness('h0')   # PET substrate
        self.h1_adhesion = materials.get_thickness('h1')    # Adhesion promoter
        self.h2_barrier1 = materials.get_thickness('h2')    # Barrier 1
        self.h3_interlayer = materials.get_thickness('h3')  # Interlayer
        self.h4_barrier2 = materials.get_thickness('h4')    # Barrier 2
        self.h5_topcoat = materials.get_thickness('h5')     # Top coat

        # Calculate total height (all layers)
        self.total_height = (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + 
                            self.h3_interlayer + self.h4_barrier2 + self.h5_topcoat)
        
        log_message("Layer thicknesses initialized (nm):", self.log_file)
        log_message("  h0_substrate: {:.1f}".format(self.h0_substrate), self.log_file)
        log_message("  h1_adhesion: {:.1f}".format(self.h1_adhesion), self.log_file)
        log_message("  h2_barrier1: {:.1f}".format(self.h2_barrier1), self.log_file)
        log_message("  h3_interlayer: {:.1f}".format(self.h3_interlayer), self.log_file)
        log_message("  h4_barrier2: {:.1f}".format(self.h4_barrier2), self.log_file)
        log_message("  h5_topcoat: {:.1f}".format(self.h5_topcoat), self.log_file)
        log_message("  total_height: {:.1f}".format(self.total_height), self.log_file)

    def set_model(self, model):
        """Set the ABAQUS model"""
        self.model = model
        log_message("Model set: {}".format(model.name if model else None), self.log_file)

    def set_parameters(self, crack_width=None, crack_spacing=None, crack_offset=None, 
                       single_sided=None, thicknesses=None):
        """Update geometry parameters - ALL NANOMETER UNITS"""
        log_message("Setting parameters (nanometer units):", self.log_file)
        
        if crack_width is not None:
            self.crack_width = crack_width  # nm
            log_message("  crack_width: {:.1f} nm".format(crack_width), self.log_file)
        if crack_spacing is not None:
            self.crack_spacing = crack_spacing  # nm
            log_message("  crack_spacing: {:.1f} nm ({:.1f} um)".format(
                crack_spacing, crack_spacing/1000), self.log_file)
        if crack_offset is not None:
            self.crack_offset = crack_offset
            log_message("  crack_offset: {:.2f}".format(crack_offset), self.log_file)
        if single_sided is not None:
            self.single_sided = single_sided
            log_message("  single_sided: {}".format(single_sided), self.log_file)
        if thicknesses is not None:
            for key, value in thicknesses.items():
                setattr(self, key, value)
                log_message("  {}: {:.1f} nm".format(key, value), self.log_file)
        
        # Recalculate total height
        self.total_height = (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + 
                            self.h3_interlayer + self.h4_barrier2 + self.h5_topcoat)
        log_message("  total_height: {:.1f} nm".format(self.total_height), self.log_file)

    def create_materials(self):
        """Create ABAQUS materials with diffusion properties"""
        log_message("Creating materials...", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create materials in ABAQUS environment", self.log_file)
            return
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
            
        # Create materials with diffusion properties
        for mat_name, diffusivity in materials.diffusivities.items():
            try:
                mat = self.model.Material(name=mat_name)
                mat.Diffusivity(table=((diffusivity, ),))
                mat.Solubility(table=((materials.get_solubility(mat_name), ),))
                log_message("  Created material {}: D={:.1e} nm²/s, S={:.1e} mol/nm³/Pa".format(
                    mat_name, diffusivity, materials.get_solubility(mat_name)), self.log_file)
            except Exception as e:
                log_message("  ERROR creating material {}: {}".format(mat_name, str(e)), self.log_file)

    def create_sections(self):
        """Create sections for each material"""
        log_message("Creating sections...", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create material sections in ABAQUS environment", self.log_file)
            return
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
            
        # Create sections for each material
        for mat_name in materials.diffusivities.keys():
            section_name = mat_name + '_section'
            try:
                self.model.HomogeneousSolidSection(
                    name=section_name, 
                    material=mat_name,
                    thickness=None
                )
                log_message("  Created section: {}".format(section_name), self.log_file)
            except Exception as e:
                log_message("  ERROR creating section {}: {}".format(section_name, str(e)), self.log_file)

    def create_unit_cell_geometry(self):
        """Create 2D unit cell with periodic crack pattern - NANOMETER UNITS"""
        log_message("Creating unit cell geometry (nanometer units)...", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create unit cell geometry in ABAQUS environment", self.log_file)
            return None
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Create 2D deformable part
        part_name = 'UnitCell'
        width = self.crack_spacing  # nm
        
        log_message("  Creating part: {} (width={:.1f} nm, height={:.1f} nm)".format(
            part_name, width, self.total_height), self.log_file)
        
        try:
            # Create base rectangular geometry
            sketch = self.model.ConstrainedSketch(name=part_name+'_sketch', sheetSize=1.0)
            sketch.rectangle(point1=(0.0, 0.0), point2=(width, self.total_height))
            
            # Create base part
            part = self.model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
            part.BaseShell(sketch=sketch)
            
            log_message("  Base geometry created successfully", self.log_file)
            
            # Create partitions for layers and cracks
            self._create_layer_partitions(part, width)
            self._create_crack_partitions(part, width)
            
            log_message("  Unit cell geometry completed with {} faces".format(len(part.faces)), self.log_file)
            return part
            
        except Exception as e:
            log_message("  ERROR creating unit cell geometry: {}".format(str(e)), self.log_file)
            raise

    def _create_layer_partitions(self, part, width):
        """Create horizontal partitions to define different material layers"""
        log_message("  Creating layer partitions...", self.log_file)
        
        # Layer interface positions (y-coordinates from bottom)
        y_positions = []
        y_current = 0.0
        
        # Build y-coordinates for each layer interface
        layer_thicknesses = [
            self.h0_substrate,    # PET substrate
            self.h1_adhesion,     # Adhesion promoter
            self.h2_barrier1,     # Barrier 1
            self.h3_interlayer,   # Interlayer
            self.h4_barrier2,     # Barrier 2
            # h5_topcoat is last, no partition needed at top
        ]
        
        for i, thickness in enumerate(layer_thicknesses):
            y_current += thickness
            y_positions.append(y_current)
            log_message("    Layer {} interface at y={:.1f} nm".format(i, y_current), self.log_file)
        
        # Create horizontal partition lines at layer interfaces
        for i, y_pos in enumerate(y_positions):
            try:
                sketch = self.model.ConstrainedSketch(name='layer_interface_{}'.format(i), sheetSize=1.0)
                sketch.Line(point1=(0.0, y_pos), point2=(width, y_pos))
                part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)
                log_message("    Created layer partition at y={:.1f} nm".format(y_pos), self.log_file)
            except Exception as e:
                log_message("    ERROR creating layer partition at y={:.1f}: {}".format(
                    y_pos, str(e)), self.log_file)

    def _create_crack_partitions(self, part, width):
        """Create vertical partitions to define crack regions in barrier layers"""
        log_message("  Creating crack partitions...", self.log_file)
        
        # Barrier 1 crack parameters
        if self.h2_barrier1 > 0:
            y1_bottom = self.h0_substrate + self.h1_adhesion
            y1_top = y1_bottom + self.h2_barrier1
            x1_offset = 0.0  # First barrier has no offset
            log_message("    Barrier1: y={:.1f} to {:.1f} nm, offset={:.1f} nm".format(
                y1_bottom, y1_top, x1_offset), self.log_file)
            self._create_vertical_crack_partitions(part, width, x1_offset, y1_bottom, y1_top, "Barrier1")
        
        # Barrier 2 crack parameters
        if self.h4_barrier2 > 0:
            y2_bottom = self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer
            y2_top = y2_bottom + self.h4_barrier2
            x2_offset = self.crack_offset * self.crack_spacing
            log_message("    Barrier2: y={:.1f} to {:.1f} nm, offset={:.1f} nm".format(
                y2_bottom, y2_top, x2_offset), self.log_file)
            self._create_vertical_crack_partitions(part, width, x2_offset, y2_bottom, y2_top, "Barrier2")

    def _create_vertical_crack_partitions(self, part, width, x_offset, y_bottom, y_top, layer_name):
        """Create vertical partition lines for cracks within a specific layer"""
        log_message("    Creating vertical partitions for {}".format(layer_name), self.log_file)
        
        # Calculate crack boundaries with periodic wrapping
        crack_x1 = x_offset % width
        crack_x2 = (x_offset + self.crack_width) % width
        
        log_message("      Crack: x1={:.1f}, x2={:.1f} nm (width={:.1f} nm)".format(
            crack_x1, crack_x2, self.crack_width), self.log_file)
        
        # Handle periodic wrapping case
        if crack_x2 < crack_x1:
            crack_positions = [crack_x2, crack_x1]
            log_message("      Crack wraps around domain", self.log_file)
        else:
            crack_positions = [crack_x1, crack_x2]
        
        # Create vertical partitions
        tolerance = 1e-3  # 0.001 nm tolerance
        for i, x_pos in enumerate(crack_positions):
            # Skip partitions at domain boundaries
            if tolerance < x_pos < (width - tolerance):
                try:
                    sketch = self.model.ConstrainedSketch(name='{}_crack_{}'.format(layer_name, i), sheetSize=1.0)
                    sketch.Line(point1=(x_pos, y_bottom), point2=(x_pos, y_top))
                    part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)
                    log_message("      Created partition at x={:.1f} nm".format(x_pos), self.log_file)
                except Exception as e:
                    log_message("      ERROR creating partition at x={:.1f}: {}".format(x_pos, str(e)), self.log_file)
            else:
                log_message("      Skipped boundary partition at x={:.1f} nm".format(x_pos), self.log_file)

    def assign_materials_to_regions(self, part, instance):
        """Assign materials to regions with proper crack identification"""
        log_message("=== ASSIGNING MATERIALS WITH CRACK DETECTION ===", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would assign materials in ABAQUS environment", self.log_file)
            return
        
        # Calculate layer boundaries and crack positions
        boundaries = self._calculate_layer_boundaries()
        crack_positions = self._calculate_crack_positions()
        
        # Get section names
        sections = {
            'PET': 'PET_section',
            'interlayer': 'interlayer_section', 
            'barrier': 'barrier_section',
            'air_crack': 'air_crack_section'
        }
        
        # Assign materials based on face location
        self._assign_materials_with_cracks(part, boundaries, crack_positions, sections)
        
        log_message("=== MATERIAL ASSIGNMENT COMPLETED ===", self.log_file)

    def _calculate_layer_boundaries(self):
        """Calculate y-coordinates of layer boundaries"""
        boundaries = {}
        y_current = 0.0
        
        boundaries['substrate'] = (y_current, y_current + self.h0_substrate)
        y_current += self.h0_substrate
        
        boundaries['adhesion'] = (y_current, y_current + self.h1_adhesion)
        y_current += self.h1_adhesion
        
        boundaries['barrier1'] = (y_current, y_current + self.h2_barrier1)
        y_current += self.h2_barrier1
        
        boundaries['interlayer'] = (y_current, y_current + self.h3_interlayer)
        y_current += self.h3_interlayer
        
        boundaries['barrier2'] = (y_current, y_current + self.h4_barrier2)
        y_current += self.h4_barrier2
        
        boundaries['topcoat'] = (y_current, y_current + self.h5_topcoat)
        
        log_message("Layer boundaries (nm):", self.log_file)
        for name, (y_min, y_max) in boundaries.items():
            log_message("  {}: y={:.1f} to {:.1f}".format(name, y_min, y_max), self.log_file)
        
        return boundaries

    def _calculate_crack_positions(self):
        """Calculate x-coordinate ranges for cracks in barrier layers"""
        log_message("Calculating crack positions (nm):", self.log_file)
        
        width = self.crack_spacing
        crack_positions = {}
        
        # Barrier 1 cracks (no offset)
        if self.h2_barrier1 > 0:
            x1_start = 0.0
            x1_end = self.crack_width
            crack_positions['barrier1'] = {
                'x_ranges': [(x1_start, x1_end)],
                'y_range': (self.h0_substrate + self.h1_adhesion, 
                           self.h0_substrate + self.h1_adhesion + self.h2_barrier1)
            }
            log_message("  Barrier1 crack: x={:.1f} to {:.1f} nm".format(x1_start, x1_end), self.log_file)
        
        # Barrier 2 cracks (with offset)
        if self.h4_barrier2 > 0:
            x2_offset = self.crack_offset * self.crack_spacing
            x2_start = x2_offset % width
            x2_end = (x2_offset + self.crack_width) % width
            
            # Handle periodic wrapping
            if x2_end < x2_start:
                # Crack wraps around
                crack_positions['barrier2'] = {
                    'x_ranges': [(0, x2_end), (x2_start, width)],
                    'y_range': (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer,
                               self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer + self.h4_barrier2)
                }
                log_message("  Barrier2 crack (wrapped): x=0 to {:.1f} and x={:.1f} to {:.1f} nm".format(
                    x2_end, x2_start, width), self.log_file)
            else:
                crack_positions['barrier2'] = {
                    'x_ranges': [(x2_start, x2_end)],
                    'y_range': (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer,
                               self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer + self.h4_barrier2)
                }
                log_message("  Barrier2 crack: x={:.1f} to {:.1f} nm".format(x2_start, x2_end), self.log_file)
        
        return crack_positions

    def _assign_materials_with_cracks(self, part, boundaries, crack_positions, sections):
        """Assign materials to faces with crack detection"""
        log_message("Assigning materials to {} faces...".format(len(part.faces)), self.log_file)
        
        tolerance = 1.0  # 1 nm tolerance
        assignment_count = {}
        
        for i, face in enumerate(part.faces):
            try:
                # Get face centroid
                centroid_result = face.getCentroid()
                if hasattr(centroid_result, '__len__') and len(centroid_result) > 0:
                    coords = centroid_result[0]
                    x_cent, y_cent = coords[0], coords[1]
                    
                    # Determine material based on location
                    material = None
                    layer = None
                    
                    # Check each layer
                    if boundaries['substrate'][0] <= y_cent <= boundaries['substrate'][1]:
                        material = 'PET'
                        layer = 'substrate'
                    
                    elif boundaries['adhesion'][0] <= y_cent <= boundaries['adhesion'][1]:
                        material = 'interlayer'
                        layer = 'adhesion'
                    
                    elif boundaries['barrier1'][0] <= y_cent <= boundaries['barrier1'][1]:
                        # Check if in crack
                        is_crack = False
                        if 'barrier1' in crack_positions:
                            for x_range in crack_positions['barrier1']['x_ranges']:
                                if x_range[0] - tolerance <= x_cent <= x_range[1] + tolerance:
                                    is_crack = True
                                    break
                        material = 'air_crack' if is_crack else 'barrier'
                        layer = 'barrier1_crack' if is_crack else 'barrier1'
                    
                    elif boundaries['interlayer'][0] <= y_cent <= boundaries['interlayer'][1]:
                        material = 'interlayer'
                        layer = 'interlayer'
                    
                    elif boundaries['barrier2'][0] <= y_cent <= boundaries['barrier2'][1]:
                        # Check if in crack
                        is_crack = False
                        if 'barrier2' in crack_positions:
                            for x_range in crack_positions['barrier2']['x_ranges']:
                                if x_range[0] - tolerance <= x_cent <= x_range[1] + tolerance:
                                    is_crack = True
                                    break
                        material = 'air_crack' if is_crack else 'barrier'
                        layer = 'barrier2_crack' if is_crack else 'barrier2'
                    
                    elif boundaries['topcoat'][0] <= y_cent <= boundaries['topcoat'][1]:
                        material = 'interlayer'
                        layer = 'topcoat'
                    
                    # Assign material
                    if material:
                        region = (face,)
                        part.SectionAssignment(region=region, sectionName=sections[material])
                        
                        # Track assignments
                        if layer not in assignment_count:
                            assignment_count[layer] = 0
                        assignment_count[layer] += 1
                        
                        # Log only crack and barrier assignments for clarity
                        if 'crack' in layer or 'barrier' in layer:
                            log_message("  Face {} at ({:.1f},{:.1f}): {} -> {}".format(
                                i, x_cent, y_cent, layer, material), self.log_file)
                    
            except Exception as e:
                log_message("  Face {}: Error - {}".format(i, str(e)), self.log_file)
        
        # Summary
        log_message("Material assignment summary:", self.log_file)
        for layer, count in sorted(assignment_count.items()):
            log_message("  {}: {} faces".format(layer, count), self.log_file)

    def create_mesh(self, part, element_size=None):
        """Create mesh with appropriate element type for diffusion"""
        log_message("Creating mesh...", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create mesh with DC2D4 elements in ABAQUS environment", self.log_file)
            return
        
        try:
            # Set element type for mass diffusion
            part.setElementType(regions=(part.faces,), 
                            elemTypes=(ElemType(elemCode=DC2D4),))
            log_message("  Element type: DC2D4 (mass diffusion)", self.log_file)
            
            # Calculate element size
            if element_size is None:
                element_size = min(self.crack_width/2, self.crack_spacing/20)
                element_size = max(1.0, element_size)  # Minimum 1 nm
            
            log_message("  Element size: {:.1f} nm".format(element_size), self.log_file)
            
            # Seed and generate mesh
            part.seedPart(size=element_size)
            part.generateMesh()
            
            # Access mesh information correctly
            # In ABAQUS, after generateMesh(), the mesh exists but is accessed through part.elements and part.nodes
            num_elements = len(part.elements)
            num_nodes = len(part.nodes)
            
            log_message("  Mesh created: {} elements, {} nodes".format(
                num_elements, num_nodes), self.log_file)
            
        except Exception as e:
            log_message("  ERROR creating mesh: {}".format(str(e)), self.log_file)
            raise

        def print_geometry_summary(self):
            """Print summary of geometry configuration"""
            log_message("=== GEOMETRY SUMMARY ===", self.log_file)
            log_message("Domain: {:.1f} x {:.1f} nm".format(self.crack_spacing, self.total_height), self.log_file)
            log_message("Crack width: {:.1f} nm".format(self.crack_width), self.log_file)
            log_message("Crack spacing: {:.1f} nm ({:.1f} um)".format(
                self.crack_spacing, self.crack_spacing/1000), self.log_file)
            log_message("Crack offset: {:.2%}".format(self.crack_offset), self.log_file)
            log_message("Crack area fraction: {:.3%}".format(self.crack_width/self.crack_spacing), self.log_file)
            log_message("=========================", self.log_file)
