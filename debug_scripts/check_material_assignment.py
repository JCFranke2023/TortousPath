"""
Debug script to check material assignments and diffusivity values
"""

from odbAccess import openOdb
from abaqusConstants import *
import sys

def check_materials(odb_path):
    """Check material properties and flux behavior"""
    
    print("="*60)
    print("MATERIAL AND FLUX DIAGNOSTIC")
    print("="*60)
    
    odb = openOdb(odb_path, readOnly=True)
    
    try:
        step = odb.steps['Permeation']
        assembly = odb.rootAssembly
        instance_name = assembly.instances.keys()[0]
        instance = assembly.instances[instance_name]
        
        # Get frames
        frames = step.frames
        print("\nNumber of frames: {}".format(len(frames)))
        
        # Check first few frames and last frame
        check_frames = [0, 1, 2, min(5, len(frames)-1), len(frames)-1]
        
        for frame_idx in check_frames:
            if frame_idx >= len(frames):
                continue
                
            frame = frames[frame_idx]
            print("\n--- FRAME {} (t = {:.3e} s) ---".format(frame_idx, frame.frameValue))
            
            # Check NNC (concentration)
            if 'NNC' in frame.fieldOutputs:
                nnc = frame.fieldOutputs['NNC']
                conc_values = [v.data for v in nnc.values if hasattr(v, 'data')]
                if conc_values:
                    c_min = min(conc_values)
                    c_max = max(conc_values)
                    c_avg = sum(conc_values) / len(conc_values)
                    print("  NNC: min={:.3f}, max={:.3f}, avg={:.3f}".format(c_min, c_max, c_avg))
                    
                    # Check gradient exists
                    if c_max - c_min > 0.01:
                        print("  -> Concentration gradient EXISTS: ΔC = {:.3f}".format(c_max - c_min))
                    else:
                        print("  -> WARNING: No significant gradient!")
            
            # Check MFL (flux)
            if 'MFL' in frame.fieldOutputs:
                mfl = frame.fieldOutputs['MFL']
                flux_magnitudes = []
                flux_y_components = []
                
                for value in mfl.values:
                    if hasattr(value, 'data') and len(value.data) >= 2:
                        # Calculate magnitude
                        mag = (value.data[0]**2 + value.data[1]**2)**0.5
                        flux_magnitudes.append(mag)
                        flux_y_components.append(abs(value.data[1]))  # y-component
                
                if flux_magnitudes:
                    max_flux = max(flux_magnitudes)
                    avg_flux = sum(flux_magnitudes) / len(flux_magnitudes)
                    max_y_flux = max(flux_y_components) if flux_y_components else 0
                    
                    print("  MFL Magnitude: max={:.3e}, avg={:.3e}".format(max_flux, avg_flux))
                    print("  MFL Y-component: max={:.3e}".format(max_y_flux))
                    
                    if max_flux < 1e-20:
                        print("  -> WARNING: Flux is essentially ZERO!")
                    
                    # Check a few individual values
                    print("  Sample MFL vectors (first 3):")
                    count = 0
                    for value in mfl.values:
                        if count >= 3:
                            break
                        if hasattr(value, 'data'):
                            print("    {}: [{:.3e}, {:.3e}]".format(count, value.data[0], value.data[1]))
                        count += 1
        
        # Check section assignments
        print("\n--- SECTION ASSIGNMENTS ---")
        if hasattr(instance, 'sectionAssignments'):
            for sa in instance.sectionAssignments:
                print("  Section: {}".format(sa.sectionName))
                if hasattr(sa.region, 'elements'):
                    print("    Elements: {}".format(len(sa.region.elements)))
        
        # Try to check material properties from input file
        print("\n--- CHECKING FOR MATERIAL ISSUES ---")
        print("Common issues:")
        print("1. Diffusivity too HIGH -> instant equilibrium (no flux)")
        print("2. Diffusivity too LOW -> no visible flux")
        print("3. Wrong units (m²/s vs nm²/s)")
        print("4. Materials not assigned to correct regions")
        
        # Estimate what's happening
        if len(frames) > 1:
            t1 = frames[1].frameValue
            print("\nFirst time increment: {:.3e} s".format(t1))
            
            # Get instance dimensions
            nodes = instance.nodes
            y_coords = [n.coordinates[1] for n in nodes]
            thickness = max(y_coords) - min(y_coords)  # in nm
            thickness_m = thickness * 1e-9  # in m
            
            print("Model thickness: {:.3e} nm = {:.3e} m".format(thickness, thickness_m))
            
            # Estimate diffusivity from time scale
            # If diffusion is instant, D * t / L² ~ 1
            D_instant = thickness_m**2 / t1
            print("\nDiffusivity for instant equilibrium: D > {:.3e} m²/s".format(D_instant))
            print("If your material D is near or above this, flux will appear and disappear immediately")
            
            # Expected diffusivity for barriers
            print("\nTypical diffusivities:")
            print("  Air/vacuum: ~1e-5 m²/s")
            print("  Water in polymer: ~1e-12 to 1e-13 m²/s")  
            print("  Barrier material: ~1e-16 to 1e-20 m²/s")
        
        odb.close()
        return True
        
    except Exception as e:
        print("ERROR: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        odb.close()
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: abaqus python check_material_assignment.py <odb_file>")
        sys.exit(1)
    
    odb_file = sys.argv[1]
    check_materials(odb_file)
    