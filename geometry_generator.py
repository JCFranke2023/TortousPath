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
    ABAQUS_ENV = True
except ImportError:
    # Development environment - create mock objects
    ABAQUS_ENV = False
    print("Running in development mode - ABAQUS imports not available")

class GeometryGenerator:
    def __init__(self, model_name='PermeationModel'):
        self.model_name = model_name
        if ABAQUS_ENV:
            self.model = mdb.models[model_name]
        
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

    def create_unit_cell_geometry(self):
        """Create 2D unit cell with periodic crack pattern"""
        if not ABAQUS_ENV:
            print("Would create unit cell geometry with:")
            print("  Crack width: {:.1e} m".format(self.crack_width))
            print("  Crack spacing: {:.1e} m".format(self.crack_spacing))
            print("  Crack offset: {:.2f}".format(self.crack_offset))
            print("  Single sided: {}".format(self.single_sided))
            return
        
        # Create 2D deformable part
        part_name = 'UnitCell'
        sketch = self.model.ConstrainedSketch(name=part_name+'_sketch', sheetSize=1.0)
                    
        # Unit cell width (crack spacing - center to center distance)
        width = self.crack_spacing
        
        # Create outer boundary
        sketch.rectangle(point1=(0.0, 0.0), point2=(width, self.total_height))
        
        # Add crack geometry
        self._add_cracks_to_sketch(sketch, width, self.total_height)
        
        # Create part
        part = self.model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        part.BaseShell(sketch=sketch)
        
        return part

    def _add_cracks_to_sketch(self, sketch, width, height):
        """Add through-thickness crack rectangles to sketch"""
        if not ABAQUS_ENV:
            return
            
        # Barrier 1 crack - through full barrier thickness
        y1_bottom = self.h0_substrate + self.h1_adhesion
        y1_top = y1_bottom + self.h2_barrier1
        x1_offset = 0.0  # Reference crack (no offset)
        
        # Barrier 2 crack - through full barrier thickness  
        y2_bottom = self.h0_substrate + self.h1_adhesion + self.h2_barrier1 + self.h3_interlayer
        y2_top = y2_bottom + self.h4_barrier2
        x2_offset = self.crack_offset * self.crack_spacing
        
        # Create crack rectangles
        crack_definitions = [
            (y1_bottom, y1_top, x1_offset),  # Barrier 1 crack
            (y2_bottom, y2_top, x2_offset)   # Barrier 2 crack
        ]
        
        for y_bottom, y_top, x_offset in crack_definitions:
            crack_x1 = x_offset % width
            crack_x2 = (x_offset + self.crack_width) % width
            
            if crack_x1 < crack_x2:
                # Normal case - crack doesn't wrap around
                crack_points = [
                    (crack_x1, y_bottom),
                    (crack_x2, y_bottom), 
                    (crack_x2, y_top),
                    (crack_x1, y_top)
                ]
                # Create rectangle
                for i in range(4):
                    p1 = crack_points[i]
                    p2 = crack_points[(i+1) % 4]
                    sketch.Line(point1=p1, point2=p2)
            else:
                # Crack wraps around unit cell boundary - create two rectangles
                # Left part (from crack_x1 to width)
                sketch.Line(point1=(crack_x1, y_bottom), point2=(width, y_bottom))
                sketch.Line(point1=(width, y_bottom), point2=(width, y_top))  
                sketch.Line(point1=(width, y_top), point2=(crack_x1, y_top))
                sketch.Line(point1=(crack_x1, y_top), point2=(crack_x1, y_bottom))
                
                # Right part (from 0 to crack_x2)
                sketch.Line(point1=(0.0, y_bottom), point2=(crack_x2, y_bottom))
                sketch.Line(point1=(crack_x2, y_bottom), point2=(crack_x2, y_top))
                sketch.Line(point1=(crack_x2, y_top), point2=(0.0, y_top))
                sketch.Line(point1=(0.0, y_top), point2=(0.0, y_bottom))

    def create_materials(self):
        """Create ABAQUS materials"""
        if not ABAQUS_ENV:
            print("Would create materials:")
            for mat_name in materials.diffusivities.keys():
                diff = materials.get_diffusivity(mat_name)
                print("  {}: D = {:.1e} m²/s".format(mat_name, diff))
            return
            
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
            
        # Create sections for each material
        for mat_name in materials.diffusivities.keys():
            section_name = mat_name + '_section'
            self.model.HomogeneousSolidSection(
                name=section_name, 
                material=mat_name,
                thickness=None
            )

    def create_mesh(self, part, element_size=None):
        """Create mesh with appropriate element type for diffusion"""
        if not ABAQUS_ENV:
            print("Would create mesh with DC2D4 elements")
            return
            
        # Set element type for mass diffusion
        part.setElementType(regions=(part.faces,), 
                           elemTypes=(ElemType(elemCode=DC2D4),))
        
        # Seed part
        if element_size is None:
            element_size = min(self.crack_width, self.crack_spacing * 0.1)
        part.seedPart(size=element_size)
        
        # Generate mesh
        part.generateMesh()

# Create global instance
geometry = GeometryGenerator()
