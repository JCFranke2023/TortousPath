"""
Material properties for PECVD barrier coating permeation simulation
Water vapor diffusion at 37°C, 100% RH
"""

import math

# Temperature for calculations (Kelvin)
TEMPERATURE = 37 + 273.15  # 310.15 K

class MaterialProperties:
    def __init__(self):
        # Water vapor diffusivity (m²/s) at 37°C
        self.diffusivities = {
            'PET': 1.0e-12,           # PET substrate (typical range 1e-13 to 1e-11)
            'interlayer': 5.0e-11,    # Silicon organic interlayers (permeable)
            'barrier': 1.0e-16,       # PECVD barrier (effectively impermeable)
            'air_crack': 2.4e-5       # Water vapor in air-filled cracks
        }
        
        # Solubility coefficients (mol/m³/Pa) - if needed for concentration boundary conditions
        self.solubilities = {
            'PET': 1.0e-6,
            'interlayer': 2.0e-6,
            'barrier': 1.0e-9,
            'air_crack': 4.0e-2  # Ideal gas approximation
        }
        
        # Layer thickness defaults (meters)
        self.thicknesses = {
            'h0': 500e-9,    # PET substrate (50 μm)
            'h1': 50-9,   # Adhesion promoter (500 nm)
            'h2': 50-9,   # Barrier 1 (100 nm)
            'h3': 50-9,   # Interlayer (500 nm)  
            'h4': 50-9,   # Barrier 2 (100 nm)
            'h5': 50-9    # Top coat (500 nm)
        }
    
    def get_diffusivity(self, material):
        """Get diffusivity for specified material"""
        return self.diffusivities.get(material, 1.0e-12)
    
    def get_solubility(self, material):
        """Get solubility for specified material"""
        return self.solubilities.get(material, 1.0e-6)
    
    def get_thickness(self, layer):
        """Get default thickness for specified layer"""
        return self.thicknesses.get(layer, 1.0e-6)

# Create global instance
materials = MaterialProperties()
