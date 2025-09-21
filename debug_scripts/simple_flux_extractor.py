"""
Flux extraction for nm-based ABAQUS models
Handles unit conversion properly
"""

from odbAccess import openOdb
from abaqusConstants import *
import sys
import os

def extract_flux_nm_units(odb_path):
    """
    Extract flux from ODB with nm-based geometry
    MFL units: kg/(nm²·s) 
    Output units: g/(m²·day)
    """
    
    job_name = os.path.basename(odb_path).replace('.odb', '')
    print("\nProcessing: {}".format(job_name))
    print("Geometry units: nanometers")
    print("MFL units: kg/(nm²·s)")
    
    odb = openOdb(odb_path, readOnly=True)
    
    output_file = job_name + '_flux.csv'
    f = open(output_file, 'w')
    f.write('time_s,flux_kg_nm2_s,flux_g_m2_day\n')
    
    try:
        step = odb.steps['Permeation']
        instance = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]
        
        # Get y-range (in nm)
        y_coords = [n.coordinates[1] for n in instance.nodes]
        y_min = min(y_coords)
        y_max = max(y_coords)
        print("Y range: {:.1f} to {:.1f} nm".format(y_min, y_max))
        
        print("Processing {} frames...".format(len(step.frames)))
        
        for i, frame in enumerate(step.frames):
            time = frame.frameValue
            
            # Try MFL first (if it exists and is non-zero)
            flux_nm = 0  # kg/(nm²·s)
            
            if 'MFL' in frame.fieldOutputs:
                mfl = frame.fieldOutputs['MFL']
                # Get flux values near bottom
                bottom_flux = []
                for value in mfl.values:
                    if hasattr(value, 'data') and len(value.data) >= 2:
                        # MFL2 is y-component
                        flux_y = abs(value.data[1])  # kg/(nm²·s)
                        if flux_y > 0:
                            bottom_flux.append(flux_y)
                
                if bottom_flux:
                    flux_nm = sum(bottom_flux) / len(bottom_flux)

            
            # Convert units: kg/(nm²·s) to g/(m²·day)
            # 1 kg/(nm²·s) = 10^18 kg/(m²·s) = 10^21 g/(m²·s) = 8.64×10^25 g/(m²·day)
            flux_g_m2_day = flux_nm * 1e18 * 1000 * 86400
            
            f.write('{:.6e},{:.6e},{:.6e}\n'.format(time, flux_nm, flux_g_m2_day))
            
            # Progress report
            if i == 0 or i == len(step.frames)-1 or i % 10 == 0:
                print("  Frame {}: t={:.1e}s, MFL={:.2e} kg/(nm²·s) = {:.2e} g/(m²·day)".format(
                    i, time, flux_nm, flux_g_m2_day))
        
        f.close()
        odb.close()
        
        print("\nSaved: {}".format(output_file))
        print("\nUnit conversion applied:")
        print("  1 kg/(nm²·s) = 8.64×10^25 g/(m²·day)")
        
        return True
        
    except Exception as e:
        print("ERROR: {}".format(str(e)))
        f.close()
        odb.close()
        return False


def extract_flux_from_gradient_nm(odb_path, crack_width=100, crack_spacing=10000):
    """
    Alternative: Calculate flux from concentration gradient
    For nm-based models with proper unit handling
    """
    
    job_name = os.path.basename(odb_path).replace('.odb', '')
    print("\nProcessing: {}".format(job_name))
    print("Using concentration gradient method")
    
    # Diffusivities in nm²/s
    D_barrier = 1e-2      # nm²/s (= 1e-20 m²/s)
    D_crack = 2.4e13       # nm²/s (= 1e-8 m²/s)
    
    # Effective diffusivity
    crack_fraction = crack_width / crack_spacing
    D_eff = crack_fraction * D_crack + (1 - crack_fraction) * D_barrier
    
    print("Crack fraction: {:.1%}".format(crack_fraction))
    print("D_effective: {:.2e} nm²/s".format(D_eff))
    
    odb = openOdb(odb_path, readOnly=True)
    
    output_file = job_name + '_flux_gradient.csv'
    f = open(output_file, 'w')
    f.write('time_s,flux_g_m2_day,c_bottom,c_top\n')
    
    try:
        step = odb.steps['Permeation']
        instance = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]
        
        # Get y-range (in nm)
        y_coords = [n.coordinates[1] for n in instance.nodes]
        y_min = min(y_coords)
        y_max = max(y_coords)
        print('Y range: {:.1f} to {:.1f} nm'.format(y_min, y_max))
        print("Processing {} frames...".format(len(step.frames)))
        for i, frame in enumerate(step.frames):
            time = frame.frameValue

            if 'NNC11' not in frame.fieldOutputs:
                print(f'frame {i} has these field outputs: {list(frame.fieldOutputs.keys())}')
                continue
            
            nnc = frame.fieldOutputs['NNC11']
            
            # Collect concentrations
            c_bottom = []
            c_near_bottom = []
            c_top = []
            print(f"  Frame {i}: processing {len(nnc.values)} NNC values...")
            for value in nnc.values:
                if not hasattr(value, 'data'):
                    continue
                    
                if hasattr(value, 'nodeLabel') and value.nodeLabel:
                    for node in instance.nodes:
                        if node.label == value.nodeLabel:
                            y = node.coordinates[1]
                            
                            if y < y_min + 50:  # Bottom layer
                                c_bottom.append(value.data)
                            elif y_min + 50 < y < y_min + 150:  # Near bottom
                                c_near_bottom.append(value.data)
                            elif y > y_max - 50:  # Top
                                c_top.append(value.data)
                            break
            print(f'Frame {i}: c_bottom={len(c_bottom)}, c_near_bottom={len(c_near_bottom)}, c_top={len(c_top)}')
            if c_bottom and c_near_bottom:
                avg_c_bottom = sum(c_bottom) / len(c_bottom)
                avg_c_near = sum(c_near_bottom) / len(c_near_bottom)
                avg_c_top = sum(c_top) / len(c_top) if c_top else 1.0
                
                # Gradient in 1/nm
                dy = 100  # nm distance
                gradient = (avg_c_near - avg_c_bottom) / dy
                
                # Flux: J = D * dC/dy in kg/(nm²·s)
                flux_nm = abs(D_eff * gradient)
                
                # Convert to g/(m²·day)
                flux_g_m2_day = flux_nm * 1e18 * 1000 * 86400
                
                f.write('{:.3e},{:.6e},{:.4f},{:.4f}\n'.format(
                    time, flux_g_m2_day, avg_c_bottom, avg_c_top))
                
                if i == 0 or i == len(step.frames)-1 or i % 10 == 0:
                    print("  Frame {}: flux = {:.2e} g/(m²·day)".format(i, flux_g_m2_day))
        
        f.close()
        odb.close()
        print("\nSaved: {}".format(output_file))
        return True
        
    except Exception as e:
        print("ERROR: {}".format(str(e)))
        f.close()
        odb.close()
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: abaqus python nm_flux_extractor.py <odb_file> [method]")
        print("\nMethods:")
        print("  mfl     - Extract from MFL field (default)")
        print("  gradient - Calculate from concentration gradient")
        print("\nExample:")
        print("  abaqus python nm_flux_extractor.py Job.odb")
        print("  abaqus python nm_flux_extractor.py Job.odb gradient")
        sys.exit(1)
    
    odb_file = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'mfl'
    
    if method == 'gradient':
        # Parse crack parameters from filename if possible
        if 'Job_c' in odb_file:
            try:
                parts = os.path.basename(odb_file).split('_')
                crack_width = float(parts[1][1:])
                crack_spacing = float(parts[2][1:])
                print("Detected parameters: c={} nm, s={} nm".format(crack_width, crack_spacing))
                extract_flux_from_gradient_nm(odb_file, crack_width, crack_spacing)
            except:
                extract_flux_from_gradient_nm(odb_file)
        else:
            extract_flux_from_gradient_nm(odb_file)
    else:
        extract_flux_nm_units(odb_file)
