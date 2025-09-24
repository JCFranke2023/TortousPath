"""
ABAQUS geometry generator for PECVD barrier coating permeation simulation
Creates 2D representative unit cell with periodic crack patterns
Cleaned version with proper crack material assignment
"""

import os
import sys
from pathlib import Path
from material_properties import materials

# Logging setup with configurable output directory
def setup_logging(log_dir=None, log_name='geometry_generator.log'):
    """Setup logging to file"""
    if log_dir:
        log_path = Path(log_dir)
    else:
        log_path = Path('.')
    
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / log_name
    return log_file

def log_message(message, log_file=None, verbose=True):
    """Log message to file and optionally to console"""
    if log_file:
        with open(log_file, 'a') as f:
            f.write("{}\n".format(message))
    
    if verbose:
        print(message)

# Default log file (will be overridden when class is instantiated)
DEFAULT_LOG_FILE = None

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
    self.log("ABAQUS environment detected", LOG_FILE)
except ImportError:
    # Development environment - create mock objects
    ABAQUS_ENV = False
    self.log("Running in development mode - ABAQUS imports not available", LOG_FILE)

class GeometryGenerator:
    def __init__(self, model=None, log_dir=None, verbose=True):
        """
        Initialize geometry generator
        
        Args:
            model: ABAQUS model object
            log_dir: Directory for log files
            verbose: Whether to print log messages
        """
        self.model = model
        self.verbose = verbose
        
        # Setup logging
        if log_dir:
            self.log_file = setup_logging(log_dir, 'geometry_generator.log')
        else:
            self.log_file = setup_logging(None, 'geometry_generator.log')
        
        # Clear log file for new session
        if self.log_file.exists():
            self.log_file.unlink()
        
        self.log("=== GEOMETRY GENERATOR INITIALIZED ===")
        
        # Layer thicknesses from materials (bottom to top) - all in nm
        self.h0_dense_pet = materials.get_thickness('h0')  # Dense PET substrate
        self.h1_pet = materials.get_thickness('h1')        # PET layer
        self.h2_adhesion = materials.get_thickness('h2')   # Adhesion promoter
        self.h3_barrier1 = materials.get_thickness('h3')   # Barrier 1
        self.h4_interlayer = materials.get_thickness('h4') # Interlayer
        self.h5_barrier2 = materials.get_thickness('h5')   # Barrier 2
        self.h6_topcoat = materials.get_thickness('h6')    # Top coat

        # Calculate total height (all 7 layers)
        self.total_height = (self.h0_dense_pet + self.h1_pet + self.h2_adhesion + 
                            self.h3_barrier1 + self.h4_interlayer + 
                            self.h5_barrier2 + self.h6_topcoat)
        
        self.log("Layer thicknesses initialized (nm):")
        self.log("  h0_substrate: {:.1f}".format(self.h0_substrate))
        self.log("  h1_adhesion: {:.1f}".format(self.h1_adhesion))
        self.log("  h2_barrier1: {:.1f}".format(self.h2_barrier1))
        self.log("  h3_interlayer: {:.1f}".format(self.h3_interlayer))
        self.log("  h4_barrier2: {:.1f}".format(self.h4_barrier2))
        self.log("  h5_topcoat: {:.1f}".format(self.h5_topcoat))
        self.log("  total_height: {:.1f}".format(self.total_height))
    
    def log(self, message):
        """Convenience method for logging"""
        self.log(message, self.verbose)

    def set_model(self, model):
        """Set the ABAQUS model"""
        self.model = model
        self.log("Model set: {}".format(model.name if model else None))

    def set_parameters(self, crack_width=None, crack_spacing=None, crack_offset=None, 
                       single_sided=None, thicknesses=None):
        """Update geometry parameters - ALL NANOMETER UNITS"""
        self.log("Setting parameters (nanometer units):")
        
        if crack_width is not None:
            self.crack_width = crack_width  # nm
            self.log("  crack_width: {:.1f} nm".format(crack_width))
        if crack_spacing is not None:
            self.crack_spacing = crack_spacing  # nm
            self.log("  crack_spacing: {:.1f} nm ({:.1f} um)".format(
                crack_spacing, crack_spacing/1000))
        if crack_offset is not None:
            self.crack_offset = crack_offset
            self.log("  crack_offset: {:.2f}".format(crack_offset))
        if single_sided is not None:
            self.single_sided = single_sided
            self.log("  single_sided: {}".format(single_sided))
        if thicknesses is not None:
            for key, value in thicknesses.items():
                setattr(self, key, value)
                self.log("  {}: {:.1f} nm".format(key, value))
        
        # Recalculate total height
        self.total_height = (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + 
                            self.h3_interlayer + self.h4_barrier2 + self.h5_topcoat)
        self.log("  total_height: {:.1f} nm".format(self.total_height))

    def get_geometry_config(self):
        """
        Get current geometry configuration as dictionary
        
        Returns:
            Dictionary with all geometry parameters
        """
        config = {
            'crack_parameters': {
                'width': self.crack_width,
                'spacing': self.crack_spacing,
                'offset': self.crack_offset,
                'single_sided': self.single_sided
            },
            'layer_thicknesses': {
                'h0_substrate': self.h0_substrate,
                'h1_adhesion': self.h1_adhesion,
                'h2_barrier1': self.h2_barrier1,
                'h3_interlayer': self.h3_interlayer,
                'h4_barrier2': self.h4_barrier2,
                'h5_topcoat': self.h5_topcoat,
                'total': self.total_height
            },
            'derived_metrics': {
                'crack_fraction': self.crack_width / self.crack_spacing,
                'crack_density_per_mm': 1000000 / self.crack_spacing,
                'offset_distance_nm': self.crack_offset * self.crack_spacing,
                'domain_width_nm': self.crack_spacing,
                'domain_height_nm': self.total_height
            }
        }
        return config
    
    def save_geometry_config(self, filepath):
        """Save geometry configuration to JSON file"""
        import json
        config = self.get_geometry_config()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.log("Geometry configuration saved to: {}".format(filepath))

    def create_materials(self):
        """Create ABAQUS materials with diffusion properties"""
        self.log("Creating materials...")
        
        if not ABAQUS_ENV:
            self.log("Would create materials in ABAQUS environment")
            return
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
            
        # Create materials with diffusion properties
        for mat_name, diffusivity in materials.diffusivities.items():
            try:
                mat = self.model.Material(name=mat_name)
                mat.Diffusivity(table=((diffusivity, ),))
                mat.Solubility(table=((materials.get_solubility(mat_name), ),))
                self.log("  Created material {}: D={:.1e} nm²/s, S={:.1e} mol/nm³/Pa".format(
                    mat_name, diffusivity, materials.get_solubility(mat_name)))
            except Exception as e:
                self.log("  ERROR creating material {}: {}".format(mat_name, str(e)))

    def create_sections(self):
        """Create sections for each material"""
        self.log("Creating sections...")
        
        if not ABAQUS_ENV:
            self.log("Would create material sections in ABAQUS environment")
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
                self.log("  Created section: {}".format(section_name))
            except Exception as e:
                self.log("  ERROR creating section {}: {}".format(section_name, str(e)))

    def create_unit_cell_geometry(self):
        """Create 2D unit cell with periodic crack pattern - NANOMETER UNITS"""
        self.log("Creating unit cell geometry (nanometer units)...")
        
        if not ABAQUS_ENV:
            self.log("Would create unit cell geometry in ABAQUS environment")
            return None
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Create 2D deformable part
        part_name = 'UnitCell'
        width = self.crack_spacing  # nm
        
        self.log("  Creating part: {} (width={:.1f} nm, height={:.1f} nm)".format(
            part_name, width, self.total_height))
        
        try:
            # Create base rectangular geometry
            sketch = self.model.ConstrainedSketch(name=part_name+'_sketch', sheetSize=1.0)
            sketch.rectangle(point1=(0.0, 0.0), point2=(width, self.total_height))
            
            # Create base part
            part = self.model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
            part.BaseShell(sketch=sketch)
            
            self.log("  Base geometry created successfully")
            
            # Create partitions for layers and cracks
            self._create_layer_partitions(part, width)
            self._create_crack_partitions(part, width)
            
            self.log("  Unit cell geometry completed with {} faces".format(len(part.faces)))
            return part
            
        except Exception as e:
            self.log("  ERROR creating unit cell geometry: {}".format(str(e)))
            raise

    def _create_layer_partitions(self, part, width):
        """Create horizontal partitions to define different material layers"""
        self.log("  Creating layer partitions...")
        
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
            self.log("    Layer {} interface at y={:.1f} nm".format(i, y_current))
        
        # Create horizontal partition lines at layer interfaces
        for i, y_pos in enumerate(y_positions):
            try:
                sketch = self.model.ConstrainedSketch(name='layer_interface_{}'.format(i), sheetSize=1.0)
                sketch.Line(point1=(0.0, y_pos), point2=(width, y_pos))
                part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)
                self.log("    Created layer partition at y={:.1f} nm".format(y_pos))
            except Exception as e:
                self.log("    ERROR creating layer partition at y={:.1f}: {}".format(
                    y_pos, str(e)))

    def _create_crack_partitions(self, part, width):
        """Create vertical partitions to define crack regions in barrier layers"""
        self.log("  Creating crack partitions...")
        
        # Barrier 1 crack parameters
        if self.h2_barrier1 > 0:
            y1_bottom = self.h0_substrate + self.h1_adhesion
            y1_top = y1_bottom + self.h2_barrier1
            x1_offset = 0.0  # First barrier has no offset
            self.log("    Barrier1: y={:.1f} to {:.1f} nm, offset={:.1f} nm".format(
                y1_bottom, y1_top, x1_offset))
            self._create_vertical_crack_partitions(part, width, x1_offset, y1_bottom, y1_top, "Barrier1")
        
        # Barrier 2 crack parameters
        if self.h4_barrier2 > 0:
            y2_bottom = self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer
            y2_top = y2_bottom + self.h4_barrier2
            x2_offset = self.crack_offset * self.crack_spacing
            self.log("    Barrier2: y={:.1f} to {:.1f} nm, offset={:.1f} nm".format(
                y2_bottom, y2_top, x2_offset))
            self._create_vertical_crack_partitions(part, width, x2_offset, y2_bottom, y2_top, "Barrier2")

    def _create_vertical_crack_partitions(self, part, width, x_offset, y_bottom, y_top, layer_name):
        """Create vertical partition lines for cracks within a specific layer"""
        self.log("    Creating vertical partitions for {}".format(layer_name))
        
        # Calculate crack boundaries with periodic wrapping
        crack_x1 = x_offset % width
        crack_x2 = (x_offset + self.crack_width) % width
        
        self.log("      Crack: x1={:.1f}, x2={:.1f} nm (width={:.1f} nm)".format(
            crack_x1, crack_x2, self.crack_width))
        
        # Handle periodic wrapping case
        if crack_x2 < crack_x1:
            crack_positions = [crack_x2, crack_x1]
            self.log("      Crack wraps around domain")
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
                    self.log("      Created partition at x={:.1f} nm".format(x_pos))
                except Exception as e:
                    self.log("      ERROR creating partition at x={:.1f}: {}".format(x_pos, str(e)))
            else:
                self.log("      Skipped boundary partition at x={:.1f} nm".format(x_pos))

    def assign_materials_to_regions(self, part, instance):
        """Assign materials to regions with proper crack identification"""
        self.log("=== ASSIGNING MATERIALS WITH CRACK DETECTION ===")
        
        if not ABAQUS_ENV:
            self.log("Would assign materials in ABAQUS environment")
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
        
        self.log("=== MATERIAL ASSIGNMENT COMPLETED ===")

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
        
        self.log("Layer boundaries (nm):")
        for name, (y_min, y_max) in boundaries.items():
            self.log("  {}: y={:.1f} to {:.1f}".format(name, y_min, y_max))
        
        return boundaries

    def _calculate_crack_positions(self):
        """Calculate x-coordinate ranges for cracks in barrier layers"""
        self.log("Calculating crack positions (nm):")
        
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
            self.log("  Barrier1 crack: x={:.1f} to {:.1f} nm".format(x1_start, x1_end))
        
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
                self.log("  Barrier2 crack (wrapped): x=0 to {:.1f} and x={:.1f} to {:.1f} nm".format(
                    x2_end, x2_start, width))
            else:
                crack_positions['barrier2'] = {
                    'x_ranges': [(x2_start, x2_end)],
                    'y_range': (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer,
                               self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer + self.h4_barrier2)
                }
                self.log("  Barrier2 crack: x={:.1f} to {:.1f} nm".format(x2_start, x2_end))
        
        return crack_positions

    def _assign_materials_with_cracks(self, part, boundaries, crack_positions, sections):
        """Assign materials to faces with crack detection"""
        self.log("Assigning materials to {} faces...".format(len(part.faces)))
        
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
                            self.log("  Face {} at ({:.1f},{:.1f}): {} -> {}".format(
                                i, x_cent, y_cent, layer, material))
                    
            except Exception as e:
                self.log("  Face {}: Error - {}".format(i, str(e)))
        
        # Summary
        self.log("Material assignment summary:")
        for layer, count in sorted(assignment_count.items()):
            self.log("  {}: {} faces".format(layer, count))

    def create_mesh(self, part, element_size=None):
        """Create mesh with appropriate element type for diffusion"""
        self.log("Creating mesh...")
        
        if not ABAQUS_ENV:
            self.log("Would create mesh with DC2D4 elements in ABAQUS environment")
            return
        
        try:
            # Set element type for mass diffusion
            part.setElementType(regions=(part.faces,), 
                            elemTypes=(ElemType(elemCode=DC2D4),))
            self.log("  Element type: DC2D4 (mass diffusion)")
            
            # Calculate element size
            if element_size is None:
                element_size = min(self.crack_width/2, self.crack_spacing/20)
                element_size = max(1.0, element_size)  # Minimum 1 nm
            
            self.log("  Element size: {:.1f} nm".format(element_size))
            
            # Seed and generate mesh
            part.seedPart(size=element_size)
            part.generateMesh()
            
            # Access mesh information correctly
            # In ABAQUS, after generateMesh(), the mesh exists but is accessed through part.elements and part.nodes
            num_elements = len(part.elements)
            num_nodes = len(part.nodes)
            
            self.log("  Mesh created: {} elements, {} nodes".format(
                num_elements, num_nodes))
            
        except Exception as e:
            self.log("  ERROR creating mesh: {}".format(str(e)))
            raise

    def validate_geometry(self):
        """
        Validate geometry parameters and check for potential issues
        
        Returns:
            Tuple (is_valid, warnings_list)
        """
        warnings = []
        is_valid = True
        
        # Check crack width vs spacing
        if self.crack_width >= self.crack_spacing:
            warnings.append("ERROR: Crack width ({:.1f} nm) >= spacing ({:.1f} nm)".format(
                self.crack_width, self.crack_spacing))
            is_valid = False
        
        # Check if crack width is very small
        if self.crack_width < 10:
            warnings.append("WARNING: Very small crack width ({:.1f} nm) may cause mesh issues".format(
                self.crack_width))
        
        # Check aspect ratios
        crack_aspect = self.crack_spacing / self.crack_width
        if crack_aspect > 1000:
            warnings.append("WARNING: High aspect ratio ({:.0f}) may affect mesh quality".format(
                crack_aspect))
        
        # Check layer thicknesses
        min_thickness = min(self.h1_adhesion, self.h2_barrier1, 
                           self.h3_interlayer, self.h4_barrier2, self.h5_topcoat)
        if min_thickness < 10:
            warnings.append("WARNING: Very thin layer ({:.1f} nm) detected".format(min_thickness))
        
        # Check total height vs crack spacing
        if self.total_height > 10 * self.crack_spacing:
            warnings.append("WARNING: Domain height much larger than crack spacing")
        
        # Report validation results
        if is_valid:
            self.log("Geometry validation: PASSED")
        else:
            self.log("Geometry validation: FAILED")
        
        for warning in warnings:
            self.log("  {}".format(warning))
        
        return is_valid, warnings

    def print_geometry_summary(self):
            """Print summary of geometry configuration"""
            self.log("=== GEOMETRY SUMMARY ===")
            self.log("Domain: {:.1f} x {:.1f} nm".format(self.crack_spacing, self.total_height))
            self.log("Crack width: {:.1f} nm".format(self.crack_width))
            self.log("Crack spacing: {:.1f} nm ({:.1f} um)".format(
                self.crack_spacing, self.crack_spacing/1000))
            self.log("Crack offset: {:.2%}".format(self.crack_offset))
            self.log("Crack area fraction: {:.3%}".format(self.crack_width/self.crack_spacing))
            self.log("=========================")

    def check_mesh_quality(self, part):
        """
        Check mesh quality and report statistics
        
        Returns:
            Dictionary with mesh quality metrics
        """
        if not ABAQUS_ENV:
            self.log("Would check mesh quality in ABAQUS environment")
            return {}
        
        try:
            mesh_stats = {
                'total_elements': len(part.elements),
                'total_nodes': len(part.nodes),
                'element_type': 'DC2D4',
                'checks': {}
            }
            
            # Calculate element size statistics
            element_sizes = []
            for element in part.elements[:min(100, len(part.elements))]:  # Sample first 100
                nodes = element.getNodes()
                if len(nodes) == 4:  # Quad element
                    # Calculate approximate element size
                    x_coords = [n.coordinates[0] for n in nodes]
                    y_coords = [n.coordinates[1] for n in nodes]
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    size = (width + height) / 2
                    element_sizes.append(size)
            
            if element_sizes:
                mesh_stats['element_size'] = {
                    'min': min(element_sizes),
                    'max': max(element_sizes),
                    'avg': sum(element_sizes) / len(element_sizes)
                }
            
            # Check aspect ratios near cracks
            crack_elements = self._identify_crack_elements(part)
            mesh_stats['crack_region_elements'] = len(crack_elements)
            
            # Report
            self.log("Mesh Quality Check:")
            self.log("  Total elements: {}".format(mesh_stats['total_elements']))
            self.log("  Total nodes: {}".format(mesh_stats['total_nodes']))
            if 'element_size' in mesh_stats:
                self.log("  Element size (nm): min={:.1f}, max={:.1f}, avg={:.1f}".format(
                    mesh_stats['element_size']['min'],
                    mesh_stats['element_size']['max'],
                    mesh_stats['element_size']['avg']
                ))
            self.log("  Elements in crack regions: {}".format(mesh_stats['crack_region_elements']))
            
            return mesh_stats
            
        except Exception as e:
            self.log("Error checking mesh quality: {}".format(str(e)))
            return {}
    
    def _identify_crack_elements(self, part):
        """Identify elements in crack regions"""
        crack_elements = []
        
        try:
            # Calculate crack x-positions
            crack_positions = self._calculate_crack_positions()
            tolerance = self.crack_width * 1.5
            
            for element in part.elements:
                centroid = element.centroid
                if hasattr(centroid, '__len__'):
                    x_cent = centroid[0]
                    y_cent = centroid[1]
                    
                    # Check if element is in a crack region
                    for barrier, crack_info in crack_positions.items():
                        y_min, y_max = crack_info['y_range']
                        if y_min <= y_cent <= y_max:
                            for x_range in crack_info['x_ranges']:
                                if x_range[0] - tolerance <= x_cent <= x_range[1] + tolerance:
                                    crack_elements.append(element)
                                    break
        except:
            pass  # Silent fail for mesh checking
        
        return crack_elements

# Standalone execution for testing
if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Test geometry generator')
    parser.add_argument('--crack_width', type=float, default=100.0)
    parser.add_argument('--crack_spacing', type=float, default=10000.0)
    parser.add_argument('--crack_offset', type=float, default=0.25)
    parser.add_argument('--output', help='Save configuration to file')
    parser.add_argument('--validate', action='store_true', help='Validate geometry only')
    
    args = parser.parse_args()
    
    # Create generator
    generator = GeometryGenerator(verbose=True)
    
    # Set parameters
    generator.set_parameters(
        crack_width=args.crack_width,
        crack_spacing=args.crack_spacing,
        crack_offset=args.crack_offset
    )
    
    # Print summary
    generator.print_geometry_summary()
    
    # Validate
    if args.validate:
        is_valid, warnings = generator.validate_geometry()
        if not is_valid:
            sys.exit(1)
    
    # Save configuration if requested
    if args.output:
        generator.save_geometry_config(args.output)
        print(f"Configuration saved to: {args.output}")
