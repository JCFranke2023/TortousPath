"""
Material properties for PECVD barrier coating permeation simulation
Water vapor diffusion at 37°C, 100% RH
Units: nanometers, seconds
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

class MaterialProperties:
    """Material properties manager with multiple material sets"""
    
    def __init__(self, material_set: str = 'pure_PET'):
        """
        Initialize material properties
        
        Args:
            material_set: Name of material set to use
        """
        self.material_set_name = material_set
        
        # Define material sets
        self.material_sets = {
            'pure_PET': {
                'description': 'Pure PET - all materials have same properties as PET',
                'diffusivities': {  # nm²/s
                    'PET': 1.0e5,
                    'dense_PET': 1.0e5 / 23,  # 1/23 of PET
                    'interlayer': 1.0e5,
                    'barrier': 1.0e5,
                    'air_crack': 1.0e5
                },
                'solubilities': {  # mol/nm³/Pa
                    'PET': 0.263e-27,
                    'dense_PET': 0.263e-27 * 23,  # 23x PET
                    'interlayer': 0.263e-27,
                    'barrier': 0.263e-27,
                    'air_crack': 0.263e-27
                }
            },
            'realistic': {
                'description': 'Realistic material properties for water vapor at 37°C',
                'diffusivities': {  # nm²/s
                    'PET': 1.44e11,
                    'dense_PET': 1.44e11 / 23,
                    'interlayer': 5.0e7,
                    'barrier': 1.0e-2,
                    'air_crack': 2.4e13
                },
                'solubilities': {  # mol/nm³/Pa
                    'PET': 1.0e-33,
                    'dense_PET': 1.0e-33 * 23,
                    'interlayer': 2.0e-33,
                    'barrier': 1.0e-36,
                    'air_crack': 4.0e-29
                }
            },
            'high_barrier': {
                'description': 'High performance barrier coating',
                'diffusivities': {  # nm²/s
                    'PET': 1.44e11,
                    'dense_PET': 1.44e11 / 23,
                    'interlayer': 2.0e7,
                    'barrier': 1.0e-4,
                    'air_crack': 2.4e13
                },
                'solubilities': {  # mol/nm³/Pa
                    'PET': 1.0e-33,
                    'dense_PET': 1.0e-33 * 23,
                    'interlayer': 1.0e-33,
                    'barrier': 1.0e-37,
                    'air_crack': 4.0e-29
                }
            },
            'validation': {
                'description': 'Simplified properties for validation - all same as PET',
                'diffusivities': {  # nm²/s
                    'PET': 1.0e6,
                    'dense_PET': 1.0e6 / 23,
                    'interlayer': 1.0e6,
                    'barrier': 1.0e6,
                    'air_crack': 1.0e6
                },
                'solubilities': {  # mol/nm³/Pa
                    'PET': 2.63e-28,
                    'dense_PET': 2.63e-28 * 23,
                    'interlayer': 2.63e-28,
                    'barrier': 2.63e-28,
                    'air_crack': 2.63e-28
                }
            }
        }
        
        # Load selected material set
        self.load_material_set(material_set)
        
        # Layer thicknesses (nanometers)
        # All layers are 50 nm except PET (500 nm) and dense_PET (500 nm)
        self.thicknesses = {
            'h0': 500,      # dense_PET substrate - 500 nm
            'h1': 500,      # PET layer - 500 nm
            'h2': 50,       # Adhesion promoter - 50 nm
            'h3': 50,       # Barrier 1 - 50 nm
            'h4': 50,       # Interlayer - 50 nm
            'h5': 50,       # Barrier 2 - 50 nm
            'h6': 50        # Top coat - 50 nm
        }
        
        # Material assignment for each layer
        self.layer_materials = {
            'h0': 'dense_PET',   # Dense PET substrate
            'h1': 'PET',         # Regular PET layer
            'h2': 'interlayer',  # Adhesion promoter
            'h3': 'barrier',     # Barrier 1
            'h4': 'interlayer',  # Interlayer
            'h5': 'barrier',     # Barrier 2
            'h6': 'interlayer'   # Top coat
        }

    def load_material_set(self, set_name: str):
        """Load a specific material set"""
        if set_name not in self.material_sets:
            available = list(self.material_sets.keys())
            raise ValueError(f"Unknown material set: {set_name}. Available: {available}")
        
        selected = self.material_sets[set_name]
        self.diffusivities = selected['diffusivities'].copy()
        self.solubilities = selected['solubilities'].copy()
        self.material_set_name = set_name
        
        print(f"Loaded material set: {set_name}")
        print(f"  Description: {selected['description']}")
    
    def get_diffusivity(self, material: str) -> float:
        """
        Get diffusivity for specified material
        
        Args:
            material: Material name
        
        Returns:
            Diffusivity in nm²/s
        """
        if material not in self.diffusivities:
            print(f"WARNING: Unknown material '{material}', using PET properties")
            material = 'PET'
        
        return self.diffusivities[material]
    
    def get_solubility(self, material: str) -> float:
        """
        Get solubility for specified material
        
        Args:
            material: Material name
        
        Returns:
            Solubility in mol/nm³/Pa
        """
        if material not in self.solubilities:
            print(f"WARNING: Unknown material '{material}', using PET properties")
            material = 'PET'
        
        return self.solubilities[material]
    
    def get_thickness(self, layer: str) -> float:
        """
        Get thickness for specified layer
        
        Args:
            layer: Layer identifier (h0-h5)
        
        Returns:
            Thickness in nm
        """
        return self.thicknesses.get(layer, 50)
    
    def set_thickness(self, layer: str, thickness: float):
        """Set thickness for a specific layer"""
        if layer in self.thicknesses:
            old_thickness = self.thicknesses[layer]
            self.thicknesses[layer] = thickness
            print(f"Layer {layer} thickness changed: {old_thickness} nm → {thickness} nm")
        else:
            print(f"WARNING: Unknown layer '{layer}'")
    
    def get_layer_material(self, layer: str) -> str:
        """Get the material assigned to a specific layer"""
        return self.layer_materials.get(layer, 'PET')
    
    def calculate_permeability(self, material: str, thickness: float) -> float:
        """
        Calculate permeability coefficient
        P = D * S / L
        
        Args:
            material: Material name
            thickness: Layer thickness in nm
        
        Returns:
            Permeability in mol/(nm·Pa·s)
        """
        D = self.get_diffusivity(material)
        S = self.get_solubility(material)
        
        if thickness > 0:
            return D * S / thickness
        else:
            return 0.0
    
    def calculate_resistance(self, material: str, thickness: float) -> float:
        """
        Calculate diffusion resistance
        R = L / (D * S)
        
        Returns:
            Resistance in (Pa·s·nm)/mol
        """
        P = self.calculate_permeability(material, thickness)
        if P > 0:
            return 1 / P
        else:
            return float('inf')
    
    def get_total_resistance(self) -> float:
        """Calculate total resistance of the multilayer stack"""
        total_R = 0
        
        for layer, thickness in self.thicknesses.items():
            material = self.layer_materials[layer]
            # Skip crack materials for this calculation
            if material != 'air_crack':
                R = self.calculate_resistance(material, thickness)
                if R != float('inf'):
                    total_R += R
        
        return total_R
    
    def get_material_summary(self) -> Dict:
        """Get summary of all material properties"""
        summary = {
            'material_set': self.material_set_name,
            'n_layers': 7,
            'materials': {},
            'layer_stack': []
        }
        
        # Material properties
        for mat in self.diffusivities.keys():
            summary['materials'][mat] = {
                'diffusivity_nm2_s': self.diffusivities[mat],
                'diffusivity_m2_s': self.diffusivities[mat] * 1e-18,
                'solubility_mol_nm3_Pa': self.solubilities[mat],
                'solubility_mol_m3_Pa': self.solubilities[mat] * 1e27
            }
        
        # Layer stack - now 7 layers
        for layer in ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if layer in self.thicknesses:
                summary['layer_stack'].append({
                    'layer': layer,
                    'material': self.layer_materials.get(layer, 'unknown'),
                    'thickness_nm': self.thicknesses[layer]
                })
        
        return summary    

    def export_for_abaqus(self) -> Dict:
        """
        Export material properties in format ready for ABAQUS
        
        Returns:
            Dictionary with material definitions
        """
        abaqus_materials = {}
        
        # Only export materials that are actually used
        used_materials = set(self.layer_materials.values())
        used_materials.add('air_crack')  # Always include for cracks
        
        for material in used_materials:
            abaqus_materials[material] = {
                'Diffusivity': {
                    'table': [(self.get_diffusivity(material),)]
                },
                'Solubility': {
                    'table': [(self.get_solubility(material),)]
                }
            }
        
        return abaqus_materials
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save current material properties to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'material_set': self.material_set_name,
            'diffusivities': self.diffusivities,
            'solubilities': self.solubilities,
            'thicknesses': self.thicknesses,
            'layer_materials': self.layer_materials
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Material properties saved to: {filepath}")
    
    def load_from_file(self, filepath: Union[str, Path]):
        """Load material properties from JSON file"""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.material_set_name = data.get('material_set', 'custom')
        self.diffusivities = data['diffusivities']
        self.solubilities = data['solubilities']
        self.thicknesses = data.get('thicknesses', self.thicknesses)
        self.layer_materials = data.get('layer_materials', self.layer_materials)
        
        print(f"Material properties loaded from: {filepath}")
    
    def validate_properties(self) -> tuple[bool, list[str]]:
        """
        Validate material properties for physical consistency
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True
        
        # Check diffusivity values
        for mat, D in self.diffusivities.items():
            if D <= 0:
                warnings.append(f"ERROR: Negative diffusivity for {mat}")
                is_valid = False
            
            # Check dense_PET relationship
            if mat == 'dense_PET':
                expected_D = self.diffusivities.get('PET', 0) / 23
                if abs(D - expected_D) > 1e-10:
                    warnings.append(f"WARNING: dense_PET diffusivity not 1/23 of PET")
        
        # Check solubility values
        for mat, S in self.solubilities.items():
            if S <= 0:
                warnings.append(f"ERROR: Negative solubility for {mat}")
                is_valid = False
            
            # Check dense_PET relationship
            if mat == 'dense_PET':
                expected_S = self.solubilities.get('PET', 0) * 23
                if abs(S - expected_S) > 1e-40:
                    warnings.append(f"WARNING: dense_PET solubility not 23x PET")
        
        # Check layer thicknesses
        for layer, thickness in self.thicknesses.items():
            if thickness <= 0:
                warnings.append(f"ERROR: Invalid thickness for {layer}: {thickness} nm")
                is_valid = False
        
        # Check expected thicknesses for 7-layer structure
        if self.thicknesses.get('h0') != 500:
            warnings.append(f"INFO: dense_PET thickness is {self.thicknesses['h0']} nm (expected 500 nm)")
        if self.thicknesses.get('h1') != 500:
            warnings.append(f"INFO: PET thickness is {self.thicknesses['h1']} nm (expected 500 nm)")
        
        for layer in ['h2', 'h3', 'h4', 'h5', 'h6']:
            if layer in self.thicknesses and self.thicknesses[layer] != 50:
                warnings.append(f"INFO: Layer {layer} thickness is {self.thicknesses[layer]} nm (expected 50 nm)")
        
        return is_valid, warnings

    def print_summary(self):
        """Print formatted summary of material properties"""
        print("\n" + "="*70)
        print(f"MATERIAL PROPERTIES SUMMARY")
        print(f"Material Set: {self.material_set_name}")
        print("="*70)
        
        print("\nDiffusivities (nm²/s):")
        print(f"  {'Material':<12} {'D (nm²/s)':<15} {'D (m²/s)':<15} Note")
        print("  " + "-"*60)
        for mat, D in self.diffusivities.items():
            note = ""
            if mat == 'dense_PET':
                note = "(1/23 × PET)"
            print(f"  {mat:<12} {D:<15.2e} {D*1e-18:<15.2e} {note}")
        
        print("\nSolubilities (mol/nm³/Pa):")
        print(f"  {'Material':<12} {'S (mol/nm³/Pa)':<18} Note")
        print("  " + "-"*60)
        for mat, S in self.solubilities.items():
            note = ""
            if mat == 'dense_PET':
                note = "(23 × PET)"
            print(f"  {mat:<12} {S:<18.3e} {note}")
        
        print("\nLayer Stack (7 layers):")
        print(f"  {'Layer':<6} {'Material':<12} {'Thickness (nm)':<15} Description")
        print("  " + "-"*60)
        
        layer_descriptions = {
            'h0': 'Dense substrate',
            'h1': 'PET layer',
            'h2': 'Adhesion promoter',
            'h3': 'Barrier 1',
            'h4': 'Interlayer',
            'h5': 'Barrier 2',
            'h6': 'Top coat'
        }
        
        total_thickness = 0
        for layer in ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if layer in self.thicknesses:
                mat = self.layer_materials.get(layer, 'unknown')
                thick = self.thicknesses[layer]
                desc = layer_descriptions.get(layer, '')
                print(f"  {layer:<6} {mat:<12} {thick:<15.0f} {desc}")
                total_thickness += thick
        
        print("  " + "-"*60)
        print(f"  {'Total':<6} {'':<12} {total_thickness:<15.0f}")
        
        # Calculate permeation for both substrate layers
        P_dense = self.calculate_permeability('dense_PET', self.thicknesses['h0'])
        P_pet = self.calculate_permeability('PET', self.thicknesses['h1'])
        
        print(f"\nPermeabilities:")
        print(f"  dense_PET layer: {P_dense:.3e} mol/(nm·Pa·s)")
        print(f"  PET layer: {P_pet:.3e} mol/(nm·Pa·s)")
        
        print("="*70 + "\n")

# Create global instance with pure_PET as default
materials = MaterialProperties(material_set='pure_PET')

# Utility functions for backward compatibility
def get_material_sets() -> list[str]:
    """Get list of available material sets"""
    return list(materials.material_sets.keys())

def set_material_set(set_name: str):
    """Switch to a different material set"""
    global materials
    materials.load_material_set(set_name)

def use_dense_substrate():
    """Switch substrate to dense_PET"""
    materials.set_substrate_type('dense_PET')

def use_normal_substrate():
    """Switch substrate to normal PET"""
    materials.set_substrate_type('PET')


# Command line interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Material properties manager')
    parser.add_argument('--set', choices=get_material_sets(), 
                       default='pure_PET', help='Material set to use')
    parser.add_argument('--substrate', choices=['PET', 'dense_PET'],
                       default='PET', help='Substrate type')
    parser.add_argument('--save', help='Save properties to file')
    parser.add_argument('--load', help='Load properties from file')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate material properties')
    
    args = parser.parse_args()
    
    # Create material properties
    materials = MaterialProperties(material_set=args.set)
    
    # Set substrate type
    materials.set_substrate_type(args.substrate)
    
    # Load from file if specified
    if args.load:
        materials.load_from_file(args.load)
    
    # Print summary
    materials.print_summary()
    
    # Validate if requested
    if args.validate:
        is_valid, warnings = materials.validate_properties()
        print("\nValidation Results:")
        if is_valid:
            print("  ✓ All properties are valid")
        else:
            print("  ✗ Validation failed")
        
        for warning in warnings:
            print(f"  {warning}")
    
    # Save if requested
    if args.save:
        materials.save_to_file(args.save)
