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
        
        # Create 2D deformable part - simplified approach
        part_name = 'UnitCell'
        
        # Create simple rectangular geometry for now
        sketch = self.model.ConstrainedSketch(name=part_name+'_sketch', sheetSize=1.0)
        width = self.crack_spacing
        sketch.rectangle(point1=(0.0, 0.0), point2=(width, self.total_height))
        
        # Create base part
        part = self.model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        part.BaseShell(sketch=sketch)
        
        print("Created basic unit cell geometry: {:.1e} x {:.1e} m".format(width, self.total_height))
        
        return part

    def _create_crack_partitions(self, part, width):
        """Create partitions to define crack regions"""
        if not ABAQUS_ENV:
            return
            
        print("Creating partitions for crack regions...")
        
        # Barrier 1 crack parameters
        y1_bottom = self.h0_substrate + self.h1_adhesion
        y1_top = y1_bottom + self.h2_barrier1
        x1_offset = 0.0
        
        # Barrier 2 crack parameters
        y2_bottom = self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer
        y2_top = y2_bottom + self.h4_barrier2
        x2_offset = self.crack_offset * self.crack_spacing
        
        # Create vertical partition lines for crack boundaries
        self._create_vertical_partitions(part, width, x1_offset, "Crack1")
        self._create_vertical_partitions(part, width, x2_offset, "Crack2")
        
        # Create horizontal partition lines at layer interfaces
        self._create_horizontal_partitions(part, width)

    def _create_vertical_partitions(self, part, width, x_offset, crack_name):
        """Create vertical partition lines for a crack"""
        crack_x1 = x_offset % width
        crack_x2 = (x_offset + self.crack_width) % width
        
        # Skip if crack is too small
        if abs(crack_x2 - crack_x1) < 1e-12:
            print("Skipping {} - too narrow".format(crack_name))
            return
            
        try:
            # Create vertical lines at crack boundaries
            for i, x_pos in enumerate([crack_x1, crack_x2]):
                if 1e-12 < x_pos < (width - 1e-12):  # Don't partition at boundaries
                    sketch = self.model.ConstrainedSketch(name=crack_name + '_vert_{}'.format(i), sheetSize=1.0)
                    sketch.Line(point1=(x_pos, 0.0), point2=(x_pos, self.total_height))
                    part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)
                    print("Created vertical partition for {} at x={:.1e}".format(crack_name, x_pos))
        except Exception as e:
            print("Warning: Could not create vertical partitions for {}: {}".format(crack_name, str(e)))

    def _create_horizontal_partitions(self, part, width):
        """Create horizontal partition lines at layer interfaces"""
        try:
            # Layer interface positions
            y_positions = [
                self.h0_substrate,
                self.h0_substrate + self.h1_adhesion,
                self.h0_substrate + self.h1_adhesion + self.h2_barrier1,
                self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer,
                self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer + self.h4_barrier2
            ]
            
            for i, y_pos in enumerate(y_positions):
                sketch = self.model.ConstrainedSketch(name='layer_interface_{}'.format(i), sheetSize=1.0)
                sketch.Line(point1=(0.0, y_pos), point2=(width, y_pos))
                part.PartitionFaceBySketch(faces=part.faces, sketch=sketch)
                print("Created horizontal partition at y={:.1e}".format(y_pos))
        except Exception as e:
            print("Warning: Could not create horizontal partitions: {}".format(str(e)))

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

    def assign_sections(self, part):
        """Create sections and assign to part regions"""
        if not ABAQUS_ENV:
            print("Would assign material sections to part regions")
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
        
        # This will need to be implemented after partitioning the geometry
        # into different regions for each layer and material type
        # For now, sections are created but not assigned - 
        # assignment should be done after partitioning in simulation_runner

    def create_mesh(self, part, element_size=None):
        """Create mesh with appropriate element type for diffusion"""
        if not ABAQUS_ENV:
            print("Would create mesh with DC2D4 elements")
            return
        
        # Set element type for mass diffusion
        part.setElementType(regions=(part.faces,), 
                           elemTypes=(ElemType(elemCode=DC2D4),))
        
        # Calculate appropriate element size (in nanometers)
        if element_size is None:
            # Use crack width or a reasonable fraction of spacing
            # Ensure good resolution of crack features
            element_size = min(self.crack_width/2, self.crack_spacing/20)
            # But don't go below 10 nm for reasonable mesh density
            element_size = max(10.0, element_size)
        
        print("Seeding part with element size: {:.1f} nm".format(element_size))
        print("Geometry dimensions: width={:.1f} nm, height={:.1f} nm".format(
            self.crack_spacing, self.total_height))
        print("Crack width: {:.1f} nm".format(self.crack_width))
        
        # Seed part
        part.seedPart(size=element_size)
        
        # Generate mesh
        part.generateMesh()
        print("Mesh generated successfully")
