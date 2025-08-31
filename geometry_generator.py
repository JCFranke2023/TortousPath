"""
ABAQUS geometry generator for PECVD barrier coating permeation simulation
Creates 2D representative unit cell with periodic crack patterns
Now includes comprehensive file logging and robust face detection
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
        self.crack_width = 100    # c: crack width (m)
        self.crack_spacing = 10000    # d: crack spacing (m) 
        self.crack_offset = 0.25      # o: offset fraction (0-1, periodic)
        self.single_sided = False     # coating configuration
        
        # Layer thicknesses from materials (bottom to top)
        self.h0_substrate = materials.get_thickness('h0')  # PET substrate
        self.h1_adhesion = materials.get_thickness('h1')   # Adhesion promoter
        self.h2_barrier1 = materials.get_thickness('h2')   # Barrier 1
        self.h3_interlayer = materials.get_thickness('h3') # Interlayer
        self.h4_barrier2 = materials.get_thickness('h4')   # Barrier 2
        self.h5_topcoat = materials.get_thickness('h5')    # Top coat

        # Calculate total height (all layers)
        self.total_height = (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + 
                             self.h3_interlayer + self.h4_barrier2 + self.h5_topcoat)
        
        log_message("Layer thicknesses initialized:", self.log_file)
        log_message("  h0_substrate: {:.1e}".format(self.h0_substrate), self.log_file)
        log_message("  h1_adhesion: {:.1e}".format(self.h1_adhesion), self.log_file)
        log_message("  h2_barrier1: {:.1e}".format(self.h2_barrier1), self.log_file)
        log_message("  h3_interlayer: {:.1e}".format(self.h3_interlayer), self.log_file)
        log_message("  h4_barrier2: {:.1e}".format(self.h4_barrier2), self.log_file)
        log_message("  h5_topcoat: {:.1e}".format(self.h5_topcoat), self.log_file)
        log_message("  total_height: {:.1e}".format(self.total_height), self.log_file)

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
            log_message("  crack_spacing: {:.1f} nm ({:.1f} um)".format(crack_spacing, crack_spacing/1000), self.log_file)
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
        
        # Recalculate total height (all in nanometers)
        old_height = self.total_height
        self.total_height = (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + 
                            self.h3_interlayer + self.h4_barrier2 + self.h5_topcoat)
        log_message("  total_height updated: {:.1f} -> {:.1f} nm".format(old_height, self.total_height), self.log_file)

    def create_materials(self):
        """Create ABAQUS materials"""
        log_message("Creating materials...", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create materials in ABAQUS environment", self.log_file)
            return
        
        if self.model is None:
            error_msg = "Model not set. Call set_model() first."
            log_message("ERROR: {}".format(error_msg), self.log_file)
            raise ValueError(error_msg)
            
        # Create materials with diffusion properties
        for mat_name, diffusivity in materials.diffusivities.items():
            try:
                mat = self.model.Material(name=mat_name)
                mat.Diffusivity(table=((diffusivity, ),))
                mat.Solubility(table=((materials.get_solubility(mat_name), ),))
                log_message("  Created material {}: D={:.1e}, S={:.1e}".format(
                    mat_name, diffusivity, materials.get_solubility(mat_name)), self.log_file)
            except Exception as e:
                log_message("  ERROR creating material {}: {}".format(mat_name, str(e)), self.log_file)

    def assign_sections(self):
        """Create sections for each material"""
        log_message("Creating sections...", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create material sections in ABAQUS environment", self.log_file)
            return
        
        if self.model is None:
            error_msg = "Model not set. Call set_model() first."
            log_message("ERROR: {}".format(error_msg), self.log_file)
            raise ValueError(error_msg)
            
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
            return
        
        if self.model is None:
            error_msg = "Model not set. Call set_model() first."
            log_message("ERROR: {}".format(error_msg), self.log_file)
            raise ValueError(error_msg)
        
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
        """Create horizontal partitions to define different material layers - NANOMETER UNITS"""
        log_message("  Creating layer partitions (nanometer coordinates)...", self.log_file)
        
        # Layer interface positions (y-coordinates from bottom, all in nm)
        y_positions = []
        y_current = 0.0
        
        # Build y-coordinates for each layer interface
        layer_thicknesses = [
            self.h0_substrate,    # PET substrate bottom
            self.h1_adhesion,     # Adhesion promoter
            self.h2_barrier1,     # Barrier 1 (has cracks)
            self.h3_interlayer,   # Interlayer
            self.h4_barrier2,     # Barrier 2 (has cracks)
            self.h5_topcoat       # Top coat
        ]
        
        for i, thickness in enumerate(layer_thicknesses[:-1]):  # Don't partition at very top
            y_current += thickness
            y_positions.append(y_current)
            log_message("    Layer {} interface at y={:.1f} nm (thickness={:.1f} nm)".format(
                i, y_current, thickness), self.log_file)
        
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
        """Create partitions to define crack regions in barrier layers - NANOMETER UNITS"""
        log_message("  Creating crack partitions (nanometer coordinates)...", self.log_file)
        
        # Barrier 1 crack parameters (only if layer thickness > 0)
        if self.h2_barrier1 > 0:
            y1_bottom = self.h0_substrate + self.h1_adhesion
            y1_top = y1_bottom + self.h2_barrier1
            x1_offset = 0.0  # First barrier has no offset
            log_message("    Barrier1 cracks: y={:.1f} to {:.1f} nm, offset={:.1f} nm".format(
                y1_bottom, y1_top, x1_offset), self.log_file)
            self._create_vertical_crack_partitions(part, width, x1_offset, y1_bottom, y1_top, "Barrier1")
        
        # Barrier 2 crack parameters (only if layer thickness > 0)
        if self.h4_barrier2 > 0:
            y2_bottom = self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer
            y2_top = y2_bottom + self.h4_barrier2
            x2_offset = self.crack_offset * self.crack_spacing  # Second barrier is offset
            log_message("    Barrier2 cracks: y={:.1f} to {:.1f} nm, offset={:.1f} nm".format(
                y2_bottom, y2_top, x2_offset), self.log_file)
            self._create_vertical_crack_partitions(part, width, x2_offset, y2_bottom, y2_top, "Barrier2")

    def _create_vertical_crack_partitions(self, part, width, x_offset, y_bottom, y_top, layer_name):
        """Create vertical partition lines for cracks within a specific layer - NANOMETER UNITS"""
        log_message("    Creating vertical crack partitions for {} (nm coordinates)".format(layer_name), self.log_file)
        
        # Calculate crack boundaries with periodic wrapping (all in nm)
        crack_x1 = x_offset % width
        crack_x2 = (x_offset + self.crack_width) % width
        
        log_message("      Crack boundaries: x1={:.1f}, x2={:.1f} nm (width={:.1f} nm)".format(
            crack_x1, crack_x2, self.crack_width), self.log_file)
        
        # Handle periodic wrapping case
        if crack_x2 < crack_x1:
            # Crack wraps around - create two partitions
            crack_positions = [crack_x2, crack_x1]
            log_message("      Crack wraps around domain - two partitions needed", self.log_file)
        else:
            # Normal case - crack within bounds
            crack_positions = [crack_x1, crack_x2]
            log_message("      Normal crack within domain", self.log_file)
        
        # Create vertical partitions only within the barrier layer
        for i, x_pos in enumerate(crack_positions):
            # Skip partitions exactly at domain boundaries (tolerance in nm)
            if 1e-3 < x_pos < (width - 1e-3):  # 0.001 nm tolerance
                try:
                    sketch = self.model.ConstrainedSketch(name='{}_crack_{}'.format(layer_name, i), sheetSize=1.0)
                    sketch.Line(point1=(x_pos, y_bottom), point2=(x_pos, y_top))
                    part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)
                    log_message("      Created {} crack partition at x={:.1f} nm".format(
                        layer_name, x_pos), self.log_file)
                except Exception as e:
                    log_message("      ERROR creating {} crack partition at x={:.1f}: {}".format(
                        layer_name, x_pos, str(e)), self.log_file)
            else:
                log_message("      Skipped partition at boundary x={:.1f} nm".format(x_pos), self.log_file)

    def assign_materials_to_regions(self, part, instance):
        """Assign different materials to different regions - FIXED CENTROID VERSION"""
        log_message("=== STARTING MATERIAL ASSIGNMENT (FIXED CENTROID) ===", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would assign materials to layer regions in ABAQUS environment", self.log_file)
            return
        
        if self.model is None:
            error_msg = "Model not set. Call set_model() first."
            log_message("ERROR: {}".format(error_msg), self.log_file)
            raise ValueError(error_msg)
        
        log_message("Total faces in part: {}".format(len(part.faces)), self.log_file)
        
        # Skip problematic centroid examination - it causes tuple errors
        log_message("Skipping face centroid examination (causes ABAQUS API errors)", self.log_file)
        
        # Get material sections
        sections = {
            'PET': 'PET_section',
            'interlayer': 'interlayer_section', 
            'barrier': 'barrier_section',
            'air_crack': 'air_crack_section'
        }
        
        # Verify all sections exist
        log_message("Checking available sections in model:", self.log_file)
        available_sections = list(self.model.sections.keys())
        for section_name in available_sections:
            log_message("  - {}".format(section_name), self.log_file)
        
        # Calculate layer boundaries
        layer_boundaries = self._calculate_layer_boundaries()
        
        # Try assignment strategies
        success = self._try_assignment_strategies(part, instance, layer_boundaries, sections)
        
        if not success:
            log_message("All assignment strategies failed - trying fallback", self.log_file)
            self._fallback_assignment(part, sections)
        
        log_message("=== MATERIAL ASSIGNMENT COMPLETED ===", self.log_file)

    def _calculate_layer_boundaries(self):
        """Calculate y-coordinates of layer boundaries - NANOMETER UNITS"""
        log_message("Calculating layer boundaries (nanometer coordinates):", self.log_file)
        
        boundaries = {}
        y_current = 0.0
        
        # Calculate boundaries for each layer (all in nm)
        boundaries['substrate'] = (y_current, y_current + self.h0_substrate)
        log_message("  substrate: y={:.1f} to {:.1f} nm".format(boundaries['substrate'][0], boundaries['substrate'][1]), self.log_file)
        y_current += self.h0_substrate
        
        boundaries['adhesion'] = (y_current, y_current + self.h1_adhesion)
        log_message("  adhesion: y={:.1f} to {:.1f} nm".format(boundaries['adhesion'][0], boundaries['adhesion'][1]), self.log_file)
        y_current += self.h1_adhesion
        
        boundaries['barrier1'] = (y_current, y_current + self.h2_barrier1)
        log_message("  barrier1: y={:.1f} to {:.1f} nm".format(boundaries['barrier1'][0], boundaries['barrier1'][1]), self.log_file)
        y_current += self.h2_barrier1
        
        boundaries['interlayer'] = (y_current, y_current + self.h3_interlayer)
        log_message("  interlayer: y={:.1f} to {:.1f} nm".format(boundaries['interlayer'][0], boundaries['interlayer'][1]), self.log_file)
        y_current += self.h3_interlayer
        
        boundaries['barrier2'] = (y_current, y_current + self.h4_barrier2)
        log_message("  barrier2: y={:.1f} to {:.1f} nm".format(boundaries['barrier2'][0], boundaries['barrier2'][1]), self.log_file)
        y_current += self.h4_barrier2
        
        boundaries['topcoat'] = (y_current, y_current + self.h5_topcoat)
        log_message("  topcoat: y={:.1f} to {:.1f} nm".format(boundaries['topcoat'][0], boundaries['topcoat'][1]), self.log_file)
        
        return boundaries

    def _try_assignment_strategies(self, part, instance, boundaries, sections):
        """Try different strategies to assign materials"""
        log_message("Trying different assignment strategies...", self.log_file)
        
        # Strategy 1: Centroid-based assignment
        log_message("Strategy 1: Centroid-based assignment", self.log_file)
        success1 = self._assign_by_centroids(part, boundaries, sections)
        
        if success1:
            log_message("Strategy 1 succeeded", self.log_file)
            return True
        
        # Strategy 2: Assign all faces to PET first, then reassign
        log_message("Strategy 2: Sequential reassignment", self.log_file)
        success2 = self._assign_sequentially(part, boundaries, sections)
        
        if success2:
            log_message("Strategy 2 succeeded", self.log_file)
            return True
        
        log_message("Both strategies failed", self.log_file)
        return False

    def _assign_by_centroids(self, part, boundaries, sections):
        """Assign materials based on face centroids"""
        log_message("  Assigning materials by centroid analysis...", self.log_file)
        
        # Material assignment for each layer
        layer_materials = {
            'substrate': 'PET',
            'adhesion': 'interlayer', 
            'barrier1': 'barrier',
            'interlayer': 'interlayer',
            'barrier2': 'barrier',
            'topcoat': 'interlayer'
        }
        
        total_assigned = 0
        
        for layer_name, material_name in layer_materials.items():
            if layer_name not in boundaries:
                log_message("    Skipping layer {} (not in boundaries)".format(layer_name), self.log_file)
                continue
                
            y_bottom, y_top = boundaries[layer_name]
            layer_thickness = y_top - y_bottom
            
            # Skip layers with zero thickness
            if abs(layer_thickness) < 1e-12:
                log_message("    Skipping layer {} (zero thickness)".format(layer_name), self.log_file)
                continue
            
            log_message("    Processing layer {}: y={:.1e} to {:.1e}".format(
                layer_name, y_bottom, y_top), self.log_file)
            
            # Find faces in this layer
            layer_faces = self._find_faces_in_layer(part, y_bottom, y_top, self.crack_spacing)
            
            if layer_faces:
                section_name = sections[material_name]
                try:
                    region = (layer_faces,)
                    part.SectionAssignment(region=region, sectionName=section_name)
                    total_assigned += len(layer_faces)
                    log_message("    Assigned {} to {} ({} faces)".format(
                        material_name, layer_name, len(layer_faces)), self.log_file)
                except Exception as e:
                    log_message("    ERROR assigning {}: {}".format(layer_name, str(e)), self.log_file)
                    return False
            else:
                log_message("    No faces found in layer {}".format(layer_name), self.log_file)
        
        log_message("  Centroid assignment: {} faces assigned".format(total_assigned), self.log_file)
        return total_assigned > 0

    def _assign_sequentially(self, part, boundaries, sections):
        """Assign all faces to PET first, then selectively reassign"""
        log_message("  Sequential assignment strategy...", self.log_file)
        
        try:
            # Step 1: Assign ALL faces to PET
            all_faces = tuple(part.faces)
            part.SectionAssignment(region=(all_faces,), sectionName=sections['PET'])
            log_message("    Step 1: Assigned all {} faces to PET".format(len(all_faces)), self.log_file)
            
            # Step 2: Reassign non-substrate faces
            layer_materials = {
                'adhesion': 'interlayer',
                'barrier1': 'barrier', 
                'interlayer': 'interlayer',
                'barrier2': 'barrier',
                'topcoat': 'interlayer'
            }
            
            for layer_name, material_name in layer_materials.items():
                if layer_name not in boundaries:
                    continue
                    
                y_bottom, y_top = boundaries[layer_name]
                if abs(y_top - y_bottom) < 1e-12:
                    continue
                
                layer_faces = self._find_faces_in_layer(part, y_bottom, y_top, self.crack_spacing)
                
                if layer_faces:
                    try:
                        region = (layer_faces,)
                        part.SectionAssignment(region=region, sectionName=sections[material_name])
                        log_message("    Reassigned {} faces to {} in layer {}".format(
                            len(layer_faces), material_name, layer_name), self.log_file)
                    except Exception as e:
                        log_message("    ERROR reassigning {}: {}".format(layer_name, str(e)), self.log_file)
            
            return True
            
        except Exception as e:
            log_message("  Sequential assignment failed: {}".format(str(e)), self.log_file)
            return False

    def _fallback_assignment(self, part, sections):
        """Fallback: assign all faces to PET to avoid unassigned elements"""
        log_message("Fallback assignment: assigning all faces to PET", self.log_file)
        
        try:
            all_faces = tuple(part.faces)
            part.SectionAssignment(region=(all_faces,), sectionName=sections['PET'])
            log_message("  Fallback successful: {} faces assigned to PET".format(len(all_faces)), self.log_file)
        except Exception as e:
            log_message("  Even fallback failed: {}".format(str(e)), self.log_file)

    def _find_faces_in_layer(self, part, y_bottom, y_top, width):
        """Find all faces within a specific layer - FIXED CENTROID METHOD"""
        log_message("    Searching for faces in layer: y={:.1f} to {:.1f} nm".format(
            y_bottom, y_top), self.log_file)
        
        layer_faces = []
        
        log_message("    Total faces in part: {}".format(len(part.faces)), self.log_file)
        
        for i, face in enumerate(part.faces):
            try:
                # FIXED: Handle ABAQUS getCentroid() API properly
                centroid_result = face.getCentroid()
                
                # ABAQUS returns centroid in format: ((x, y, z),)
                if hasattr(centroid_result, '__len__') and len(centroid_result) > 0:
                    if hasattr(centroid_result[0], '__len__') and len(centroid_result[0]) >= 3:
                        # Extract coordinates from nested tuple
                        coords = centroid_result[0]
                        x_cent, y_cent, z_cent = coords[0], coords[1], coords[2]
                        
                        log_message("      Face {}: centroid=({:.1f}, {:.1f}, {:.1f}) nm".format(
                            i, x_cent, y_cent, z_cent), self.log_file)
                        
                        # Check if face centroid is within layer bounds (tolerance in nm)
                        tolerance = 1e-3  # 0.001 nm tolerance
                        if (y_bottom - tolerance) <= y_cent <= (y_top + tolerance):
                            layer_faces.append(face)
                            log_message("        -> Face {} included in layer".format(i), self.log_file)
                        else:
                            log_message("        -> Face {} outside layer bounds".format(i), self.log_file)
                    else:
                        log_message("      Face {}: unexpected centroid format".format(i), self.log_file)
                else:
                    log_message("      Face {}: centroid result is empty or invalid".format(i), self.log_file)
                    
            except Exception as e:
                log_message("      Face {}: centroid error - {}".format(i, str(e)), self.log_file)
                
                # Try alternative method using face vertices as fallback
                try:
                    vertices = face.getVertices()
                    if len(vertices) > 0:
                        # Get first vertex as approximation
                        vertex = part.vertices[vertices[0]]
                        coords = vertex.pointOn[0]
                        y_approx = coords[1]
                        
                        log_message("      Face {}: using vertex approximation y={:.1f} nm".format(i, y_approx), self.log_file)
                        
                        tolerance = 1e-3  # nm tolerance
                        if (y_bottom - tolerance) <= y_approx <= (y_top + tolerance):
                            layer_faces.append(face)
                            log_message("        -> Face {} included (vertex method)".format(i), self.log_file)
                        else:
                            log_message("        -> Face {} outside bounds (vertex method)".format(i), self.log_file)
                            
                except Exception as e2:
                    log_message("      Face {}: all methods failed - {}".format(i, str(e2)), self.log_file)
        
        log_message("    Found {} faces in layer".format(len(layer_faces)), self.log_file)
        return tuple(layer_faces)

    def create_mesh(self, part, element_size=None):
        """Create mesh with appropriate element type for diffusion - NANOMETER UNITS"""
        log_message("Creating mesh (nanometer units)...", self.log_file)
        
        if not ABAQUS_ENV:
            log_message("Would create mesh with DC2D4 elements in ABAQUS environment", self.log_file)
            return
        
        try:
            # Set element type for mass diffusion
            part.setElementType(regions=(part.faces,), 
                            elemTypes=(ElemType(elemCode=DC2D4),))
            log_message("  Element type set to DC2D4", self.log_file)
            
            # Calculate appropriate element size (in nanometers)
            if element_size is None:
                # Use crack width or a reasonable fraction of spacing
                element_size = min(self.crack_width/2, self.crack_spacing/20)
                # But don't go below 1 nm for reasonable mesh density
                element_size = max(1.0, element_size)
            
            log_message("  Seeding part with element size: {:.1f} nm".format(element_size), self.log_file)
            log_message("  Geometry dimensions: width={:.1f} nm, height={:.1f} nm".format(
                self.crack_spacing, self.total_height), self.log_file)
            
            # Seed part
            part.seedPart(size=element_size)
            log_message("  Part seeded successfully", self.log_file)
            
            # Generate mesh
            part.generateMesh()
            log_message("  Mesh generated successfully", self.log_file)
            
        except Exception as e:
            log_message("  ERROR creating mesh: {}".format(str(e)), self.log_file)
            raise

    def get_layer_info(self):
        """Return layer information for debugging/reporting"""
        layer_info = {
            'total_height': self.total_height,
            'layers': [
                {'name': 'substrate', 'thickness': self.h0_substrate, 'material': 'PET'},
                {'name': 'adhesion', 'thickness': self.h1_adhesion, 'material': 'interlayer'},
                {'name': 'barrier1', 'thickness': self.h2_barrier1, 'material': 'barrier', 'has_cracks': True},
                {'name': 'interlayer', 'thickness': self.h3_interlayer, 'material': 'interlayer'},
                {'name': 'barrier2', 'thickness': self.h4_barrier2, 'material': 'barrier', 'has_cracks': True},
                {'name': 'topcoat', 'thickness': self.h5_topcoat, 'material': 'interlayer'}
            ],
            'crack_parameters': {
                'width': self.crack_width,
                'spacing': self.crack_spacing,
                'offset': self.crack_offset
            }
        }
        return layer_info

    def print_geometry_summary(self):
        """Print summary of geometry configuration - NANOMETER UNITS"""
        info = self.get_layer_info()
        
        log_message("=== GEOMETRY SUMMARY (NANOMETER UNITS) ===", self.log_file)
        log_message("Total height: {:.1f} nm".format(info['total_height']), self.log_file)
        log_message("Crack spacing: {:.1f} nm ({:.1f} um)".format(self.crack_spacing, self.crack_spacing/1000), self.log_file)
        log_message("Crack width: {:.1f} nm".format(self.crack_width), self.log_file)
        log_message("Crack offset: {:.2f}".format(self.crack_offset), self.log_file)
        
        log_message("Layer stack (bottom to top):", self.log_file)
        for layer in info['layers']:
            if layer['thickness'] > 0:
                log_message("  {}: {:.1f} nm ({}){}".format(
                    layer['name'], layer['thickness'], layer['material'],
                    " - with cracks" if layer.get('has_cracks') else ""
                ), self.log_file)
        
        log_message("=== END SUMMARY ===", self.log_file)
        