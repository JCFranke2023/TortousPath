"""
Verify the inconsistency between concentration change and flux
"""

from odbAccess import openOdb
from abaqusConstants import *
import sys

def verify_mass_balance(odb_path):
    """Check if concentration change matches flux (mass balance)"""
    
    print("="*60)
    print("MASS BALANCE VERIFICATION")
    print("="*60)
    
    odb = openOdb(odb_path, readOnly=True)
    
    try:
        step = odb.steps['Permeation']
        assembly = odb.rootAssembly
        instance_name = assembly.instances.keys()[0]
        instance = assembly.instances[instance_name]
        
        frames = step.frames
        print("Total frames: {}".format(len(frames)))
        
        # Track concentration change over time
        concentration_history = []
        flux_history = []
        
        # Analyze frames
        for i, frame in enumerate(frames):
            time = frame.frameValue
            
            # Get total concentration in domain
            if 'NNC' in frame.fieldOutputs:
                nnc = frame.fieldOutputs['NNC']
                conc_values = []
                y_positions = []
                
                for value in nnc.values:
                    if hasattr(value, 'data'):
                        conc_values.append(value.data)
                        # Get position
                        if hasattr(value, 'nodeLabel') and value.nodeLabel:
                            try:
                                node = instance.getNodeFromLabel(value.nodeLabel)
                                y_positions.append(node.coordinates[1])
                            except:
                                pass
                
                if conc_values:
                    avg_conc = sum(conc_values) / len(conc_values)
                    min_conc = min(conc_values)
                    max_conc = max(conc_values)
                    
                    concentration_history.append({
                        'time': time,
                        'avg': avg_conc,
                        'min': min_conc,
                        'max': max_conc
                    })
            
            # Get flux
            if 'MFL' in frame.fieldOutputs:
                mfl = frame.fieldOutputs['MFL']
                
                # Get flux at bottom (y minimum)
                y_coords = [n.coordinates[1] for n in instance.nodes]
                y_min = min(y_coords)
                
                bottom_flux = []
                all_flux = []
                
                for value in mfl.values:
                    if hasattr(value, 'data') and len(value.data) >= 2:
                        mag = (value.data[0]**2 + value.data[1]**2)**0.5
                        all_flux.append(mag)
                        
                        # Try to get position - ABAQUS doesn't always provide this
                        # Skip position-based filtering for now
                        bottom_flux.append(abs(value.data[1]))  # Just collect all y-components
                
                flux_history.append({
                    'time': time,
                    'max_flux': max(all_flux) if all_flux else 0,
                    'avg_flux': sum(all_flux)/len(all_flux) if all_flux else 0,
                    'bottom_flux': sum(bottom_flux)/len(bottom_flux) if bottom_flux else 0
                })
            
            # Print every 10th frame or first/last
            if i == 0 or i == len(frames)-1 or i % max(1, len(frames)//10) == 0:
                print("\nFrame {} (t={:.2e}s):".format(i, time))
                if concentration_history:
                    c = concentration_history[-1]
                    print("  Conc: min={:.3f}, max={:.3f}, avg={:.3f}".format(
                        c['min'], c['max'], c['avg']))
                if flux_history:
                    f = flux_history[-1]
                    print("  Flux: max={:.2e}, avg={:.2e}, bottom={:.2e}".format(
                        f['max_flux'], f['avg_flux'], f['bottom_flux']))
        
        # Calculate rate of concentration change
        print("\n" + "="*60)
        print("CONCENTRATION CHANGE RATE vs FLUX")
        print("="*60)
        
        if len(concentration_history) > 1:
            for i in range(1, min(5, len(concentration_history))):
                dt = concentration_history[i]['time'] - concentration_history[i-1]['time']
                dc_avg = concentration_history[i]['avg'] - concentration_history[i-1]['avg']
                dc_max = concentration_history[i]['max'] - concentration_history[i-1]['max']
                
                rate_avg = dc_avg / dt if dt > 0 else 0
                rate_max = dc_max / dt if dt > 0 else 0
                
                print("\nTime {:.2e} to {:.2e}s:".format(
                    concentration_history[i-1]['time'],
                    concentration_history[i]['time']))
                print("  dC_avg/dt = {:.2e}".format(rate_avg))
                print("  dC_max/dt = {:.2e}".format(rate_max))
                if i-1 < len(flux_history):
                    print("  Flux at t: {:.2e}".format(flux_history[i-1]['max_flux']))
                
                if abs(rate_avg) > 1e-10 and flux_history[i-1]['max_flux'] < 1e-20:
                    print("  *** INCONSISTENCY: Concentration changing but flux ~0!")
        
        # Check if this is a numerical issue
        print("\n" + "="*60)
        print("DIAGNOSIS")
        print("="*60)
        
        if concentration_history and flux_history:
            # Is concentration building up?
            c_change = concentration_history[-1]['avg'] - concentration_history[0]['avg']
            print("Total concentration change: {:.3f}".format(c_change))
            
            # Is there a gradient?
            gradient = concentration_history[-1]['max'] - concentration_history[-1]['min']
            print("Final concentration gradient: {:.3f}".format(gradient))
            
            # What's the flux?
            final_flux = flux_history[-1]['max_flux']
            print("Final max flux: {:.2e}".format(final_flux))
            
            if abs(c_change) > 0.1 and gradient > 0.1 and final_flux < 1e-15:
                print("\n*** PROBLEM CONFIRMED ***")
                print("Concentration is changing and gradient exists,")
                print("but MFL is not being calculated correctly!")
                print("\nPOSSIBLE CAUSES:")
                print("1. Element type doesn't support MFL properly")
                print("2. MFL is calculated at integration points with no gradient")
                print("3. Material properties not properly linked to elements")
                print("\nRECOMMENDATION:")
                print("Use concentration gradient method to calculate flux manually")
        
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
        print("Usage: abaqus python verify_flux_concentration.py <odb_file>")
        sys.exit(1)
    
    odb_file = sys.argv[1]
    verify_mass_balance(odb_file)
    