"""
ABAQUS geometry generator for PECVD barrier coating permeation simulation
Creates 2D representative unit cell with periodic crack patterns
"""

import os
import sys
from material_properties import materials

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
except ImportError:
    # Development environment - create mock objects
    ABAQUS_ENV = False
    print("Running in development mode - ABAQUS imports not available")

class GeometryGenerator:
    def __init__(self, model=None):
        self.model = model  # Model will be passed from simulation_runner
        
        # Geometry parameters (will be set via configuration)
        self.crack_width = 100e-9     # c: crack width (m)
        self.crack_spacing = 10e-6    # d: crack spacing (m) 
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

    def set_model(self, model):
        """Set the ABAQUS model"""
        self.model = model

    def set_parameters(self, crack_width=None, crack_spacing=None, crack_offset=None, 
                      single_sided=None, thicknesses=None):
        """Update geometry parameters"""
        if crack_width is not None:
            self.crack_width = crack_width
        if crack_spacing is not None:
            self.crack_spacing = crack_spacing
        if crack_offset is not None:
            self.crack_offset = crack_offset
        if single_sided is not None:
            self.single_sided = single_sided
        if thicknesses is not None:
            for key, value in thicknesses.items():
                setattr(self, key, value)
        
        # Recalculate total height
        self.total_height = (self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + 
                             self.h3_interlayer + self.h4_barrier2 + self.h5_topcoat)

    def create_unit_cell_geometry(self):
        """Create 2D unit cell with periodic crack pattern"""
        if not ABAQUS_ENV:
            print("Would create unit cell geometry with:")
            print("  Crack width: {:.1e} m".format(self.crack_width))
            print("  Crack spacing: {:.1e} m".format(self.crack_spacing))
            print("  Crack offset: {:.2f}".format(self.crack_offset))
            print("  Single sided: {}".format(self.single_sided))
            return
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Create 2D deformable part
        part_name = 'UnitCell'
        width = self.crack_spacing
        
        # Create base rectangular geometry
        sketch = self.model.ConstrainedSketch(name=part_name+'_sketch', sheetSize=1.0)
        sketch.rectangle(point1=(0.0, 0.0), point2=(width, self.total_height))
        
        # Create base part
        part = self.model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        part.BaseShell(sketch=sketch)
        
        # Create partitions for layers and cracks
        self._create_layer_partitions(part, width)
        self._create_crack_partitions(part, width)
        
        print("Created multilayer unit cell: {:.1e} x {:.1e} m".format(width, self.total_height))
        
        return part

    def _create_layer_partitions(self, part, width):
        """Create horizontal partitions to define different material layers"""
        if not ABAQUS_ENV:
            return
            
        print("Creating layer partitions...")
        
        # Layer interface positions (y-coordinates from bottom)
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
        
        for thickness in layer_thicknesses[:-1]:  # Don't partition at very top
            y_current += thickness
            y_positions.append(y_current)
        
        # Create horizontal partition lines at layer interfaces
        for i, y_pos in enumerate(y_positions):
            try:
                sketch = self.model.ConstrainedSketch(name='layer_interface_{}'.format(i), sheetSize=1.0)
                sketch.Line(point1=(0.0, y_pos), point2=(width, y_pos))
                part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)
                print("Created layer partition at y={:.1e}".format(y_pos))
            except Exception as e:
                print("Warning: Could not create layer partition at y={:.1e}: {}".format(y_pos, str(e)))

    def _create_crack_partitions(self, part, width):
        """Create partitions to define crack regions in barrier layers only"""
        if not ABAQUS_ENV:
            return
            
        print("Creating crack partitions...")
        
        # Barrier 1 crack parameters (only if layer thickness > 0)
        if self.h2_barrier1 > 0:
            y1_bottom = self.h0_substrate + self.h1_adhesion
            y1_top = y1_bottom + self.h2_barrier1
            x1_offset = 0.0  # First barrier has no offset
            self._create_vertical_crack_partitions(part, width, x1_offset, y1_bottom, y1_top, "Barrier1")
        
        # Barrier 2 crack parameters (only if layer thickness > 0)
        if self.h4_barrier2 > 0:
            y2_bottom = self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer
            y2_top = y2_bottom + self.h4_barrier2
            x2_offset = self.crack_offset * self.crack_spacing  # Second barrier is offset
            self._create_vertical_crack_partitions(part, width, x2_offset, y2_bottom, y2_top, "Barrier2")

    def _create_vertical_crack_partitions(self, part, width, x_offset, y_bottom, y_top, layer_name):
        """Create vertical partition lines for cracks within a specific layer"""
        # Calculate crack boundaries with periodic wrapping
        crack_x1 = x_offset % width
        crack_x2 = (x_offset + self.crack_width) % width
        
        # Handle periodic wrapping case
        if crack_x2 < crack_x1:
            # Crack wraps around - create two partitions
            crack_positions = [crack_x2, crack_x1]
        else:
            # Normal case - crack within bounds
            crack_positions = [crack_x1, crack_x2]
        
        # Create vertical partitions only within the barrier layer
        for i, x_pos in enumerate(crack_positions):
            # Skip partitions exactly at domain boundaries
            if 1e-12 < x_pos < (width - 1e-12):
                try:
                    sketch = self.model.ConstrainedSketch(name='{}_crack_{}'.format(layer_name, i), sheetSize=1.0)
                    sketch.Line(point1=(x_pos, y_bottom), point2=(x_pos, y_top))
                    part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)
                    print("Created {} crack partition at x={:.1e} (y={:.1e} to {:.1e})".format(
                        layer_name, x_pos, y_bottom, y_top))
                except Exception as e:
                    print("Warning: Could not create {} crack partition: {}".format(layer_name, str(e)))

    def create_materials(self):
        """Create ABAQUS materials"""
        if not ABAQUS_ENV:
            print("Would create materials:")
            for mat_name in materials.diffusivities.keys():
                diff = materials.get_diffusivity(mat_name)
                print("  {}: D = {:.1e} m²/s".format(mat_name, diff))
            return
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
            
        # Create materials with diffusion properties
        for mat_name, diffusivity in materials.diffusivities.items():
            mat = self.model.Material(name=mat_name)
            mat.Diffusivity(table=((diffusivity, ),))
            mat.Solubility(table=((materials.get_solubility(mat_name), ),))

    def assign_sections(self, part=None):
        """Create sections for each material (part parameter not used)"""
        if not ABAQUS_ENV:
            print("Would create material sections")
            return
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
            
        # Create sections for each material
        for mat_name in materials.diffusivities.keys():
            section_name = mat_name + '_section'
            self.model.HomogeneousSolidSection(
                name=section_name, 
                material=mat_name,
                thickness=None
            )
            print("Created section: {}".format(section_name))

    def assign_materials_to_regions(self, part, instance):
        """Assign different materials to different regions of the geometry"""
        if not ABAQUS_ENV:
            print("Would assign materials to layer regions")
            return
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        print("Assigning materials to regions...")
        
        # Get material sections
        sections = {
            'PET': 'PET_section',
            'interlayer': 'interlayer_section', 
            'barrier': 'barrier_section',
            'air_crack': 'air_crack_section'
        }
        
        # Calculate layer boundaries for region selection
        layer_boundaries = self._calculate_layer_boundaries()
        
        # Assign materials to each layer
        self._assign_layer_materials(part, instance, layer_boundaries, sections)

    def _calculate_layer_boundaries(self):
        """Calculate y-coordinates of layer boundaries"""
        boundaries = {}
        y_current = 0.0
        
        # Calculate boundaries for each layer
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
        
        return boundaries

    def _assign_layer_materials(self, part, instance, boundaries, sections):
        """Assign materials to specific layer regions"""
        width = self.crack_spacing
        
        # Material assignment for each layer
        layer_materials = {
            'substrate': 'PET',
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
            y_mid = (y_bottom + y_top) / 2
            
            # Skip layers with zero thickness
            if abs(y_top - y_bottom) < 1e-12:
                continue
        
            section_name = sections[material_name]
            
            # For barrier layers, need to assign different materials to crack vs solid regions
            if layer_name in ['barrier1', 'barrier2']:
                self._assign_barrier_materials(part, instance, layer_name, y_mid, width, sections)
            else:
                # For non-barrier layers, assign uniform material
                try:
                    # Find faces in this layer
                    sample_point = (width/2, y_mid, 0.0)
                    faces = part.faces.findAt((sample_point,))
                    if faces:
                        region = (faces,)
                        part.SectionAssignment(region=region, sectionName=section_name)
                        print("Assigned {} to {}".format(material_name, layer_name))
                except Exception as e:
                    print("Warning: Could not assign material to {}: {}".format(layer_name, str(e)))

    def _assign_barrier_materials(self, part, instance, layer_name, y_mid, width, sections):
        """Assign materials to barrier layer regions (solid barrier vs air cracks)"""
        # Determine crack boundaries for this barrier layer
        if layer_name == 'barrier1':
            x_offset = 0.0
        else:  # barrier2
            x_offset = self.crack_offset * self.crack_spacing
        
        crack_x1 = x_offset % width
        crack_x2 = (x_offset + self.crack_width) % width
        
        # Handle periodic wrapping
        if crack_x2 < crack_x1:
            # Crack wraps around domain
            crack_regions = [(0.0, crack_x2), (crack_x1, width)]
        else:
            # Normal crack within domain
            crack_regions = [(crack_x1, crack_x2)]
        
        # Try to assign air material to crack regions and barrier material to solid regions
        try:
            # Find all faces in this barrier layer
            all_faces = []
            for x_sample in [width/4, width/2, 3*width/4]:
                sample_point = (x_sample, y_mid, 0.0)
                try:
                    face = part.faces.findAt((sample_point,))
                    if face and face not in all_faces:
                        all_faces.extend(face)
                except:
                    pass
            
            if all_faces:
                # For now, assign barrier material to all faces
                # TODO: Implement more sophisticated crack vs solid detection
                region = tuple(all_faces)
                part.SectionAssignment(region=region, sectionName=sections['barrier'])
                print("Assigned barrier material to {} (simplified assignment)".format(layer_name))
            
        except Exception as e:
            print("Warning: Could not assign materials to {}: {}".format(layer_name, str(e)))

    def create_mesh(self, part, element_size=None):
        """Create mesh with appropriate element type for diffusion"""
        if not ABAQUS_ENV:
            print("Would create mesh with DC2D4 elements")
            return
        
        # Set element type for mass diffusion
        part.setElementType(regions=(part.faces,), 
                           elemTypes=(ElemType(elemCode=DC2D4),))
        
        # Calculate appropriate element size (in meters)
        if element_size is None:
            # Use crack width or a reasonable fraction of spacing
            # Ensure good resolution of crack features
            element_size = min(self.crack_width/2, self.crack_spacing/20)
            # But don't go below 1 nm for reasonable mesh density
            element_size = max(1e-9, element_size)
        
        print("Seeding part with element size: {:.1e} m".format(element_size))
        print("Geometry dimensions: width={:.1e} m, height={:.1e} m".format(
            self.crack_spacing, self.total_height))
        print("Crack width: {:.1e} m".format(self.crack_width))
        
        # Seed part with refined crack regions
        self._create_refined_mesh(part, element_size)
        
        # Generate mesh
        part.generateMesh()
        print("Mesh generated successfully")

    def _create_refined_mesh(self, part, base_element_size):
        """Create refined mesh with finer elements in crack regions"""
        if not ABAQUS_ENV:
            return
        
        try:
            # Global seed first
            part.seedPart(size=base_element_size)
            
            # Create finer seeds along crack edges if possible
            crack_element_size = base_element_size / 2
            
            # This would need more sophisticated edge detection
            # For now, use uniform seeding
            print("Using uniform mesh seeding (crack refinement not implemented)")
            
        except Exception as e:
            print("Warning: Could not create refined mesh: {}".format(str(e)))
            # Fall back to basic uniform seeding
            part.seedPart(size=base_element_size)

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
        """Print summary of geometry configuration"""
        info = self.get_layer_info()
        
        print("=== Geometry Summary ===")
        print("Total height: {:.1e} m".format(info['total_height']))
        print("Crack spacing: {:.1e} m".format(self.crack_spacing))
        print("Crack width: {:.1e} m".format(self.crack_width))
        print("Crack offset: {:.2f}".format(self.crack_offset))
        
        print("\nLayer stack (bottom to top):")
        y_pos = 0.0
        for layer in info['layers']:
            if layer['thickness'] > 0:
                print("  {}: {:.1e} m ({}){}".format(
                    layer['name'], layer['thickness'], layer['material'],
                    " - with cracks" if layer.get('has_cracks') else ""
                ))
                y_pos += layer['thickness']
        
        print("=== End Summary ===")
        