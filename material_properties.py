"""
Material properties for PECVD barrier coating permeation simulation
Water vapor diffusion at 37°C, 100% RH
Units: nanometers, seconds
"""

import math

# Temperature for calculations (Kelvin)
TEMPERATURE = 37 + 273.15  # 310.15 K

class MaterialProperties:
    def __init__(self):
        # Water vapor diffusivity (nm²/s) at 37°C - converted from m²/s
        self.diffusivities = {
            'PET': 1e5,         # 1.44e-7 m²/s = 1.44e11 nm²/s
            'interlayer': 1e5,  # 5.0e-11 m²/s = 5.0e7 nm²/s
            'barrier': 1e5,     #1.0e-2,      # 1.0e-16 m²/s = 1.0e-2 nm²/s (effectively impermeable)
            'air_crack': 1e5    # 2.4e-5 m²/s = 2.4e13 nm²/s
        }
        
        # Solubility coefficients (mol/nm³/Pa) - converted from m³
        self.solubilities = {
            'PET': 0.263e-27,      # 1.0e-6 mol/m³/Pa = 1.0e-33 mol/nm³/Pa
            'interlayer': 0.263e-27, #2.0e-33,
            'barrier': 0.263e-27, #1.0e-36,
            'air_crack': 0.263e-27, #4.0e-29  # 4.0e-2 mol/m³/Pa = 4.0e-29 mol/nm³/Pa
        }
        
        # Layer thickness defaults (nanometers)
        self.thicknesses = {
            'h0': 500,    # PET substrate (0.5 μm = 500 nm)
            'h1': 50,      # Adhesion promoter (50 nm)
            'h2': 50,      # Barrier 1 (50 nm)
            'h3': 50,      # Interlayer (50 nm)
            'h4': 50,      # Barrier 2 (50 nm)
            'h5': 50       # Top coat (50 nm)
        }
    
    def get_diffusivity(self, material):
        """Get diffusivity for specified material"""
        return self.diffusivities.get(material, 1.0e6)
    
    def get_solubility(self, material):
        """Get solubility for specified material"""
        return self.solubilities.get(material, 1.0e-33)
    
    def get_thickness(self, layer):
        """Get default thickness for specified layer"""
        return self.thicknesses.get(layer, 500)

# Create global instance
materials = MaterialProperties()