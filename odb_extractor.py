"""
ODB data extractor - runs in ABAQUS Python environment
Extracts flux data to CSV without requiring pandas
"""

import os
import sys
from pathlib import Path

# ABAQUS imports
try:
    from odbAccess import openOdb
    from abaqusConstants import *
    ABAQUS_ENV = True
except ImportError:
    print("ERROR: This script must be run with ABAQUS Python")
    sys.exit(1)

class ODBExtractor:
    def __init__(self, output_dir='extracted_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_flux_data(self, odb_path, job_name):
        """Extract flux vs time from ODB and save to CSV"""
        print("Extracting from: {}".format(odb_path))
        
        try:
            odb = openOdb(odb_path)
            
            # Try history output first
            data = self._try_history_extraction(odb)
            
            # Fallback to field output
            if data is None:
                data = self._try_field_extraction(odb)
            
            odb.close()
            
            if data:
                # Save to CSV without pandas
                csv_path = self.output_dir / '{}_flux.csv'.format(job_name)
                self._save_csv(data, csv_path)
                print("  Saved: {}".format(csv_path))
                return True
            else:
                print("  ERROR: No flux data extracted")
                return False
                
        except Exception as e:
            print("  ERROR: {}".format(str(e)))
            return False
    
    def _try_history_extraction(self, odb):
        """Try to extract from history output"""
        try:
            step = odb.steps['Permeation']
            
            for region_name, region in step.historyRegions.items():
                if 'Bottom' in region_name or 'Outlet' in region_name or 'H-Output' in region_name:
                    if 'FMFL' in region.historyOutputs:
                        fmfl = region.historyOutputs['FMFL']
                        
                        data = {
                            'time_s': [],
                            'flux_kg_m2_s': []
                        }
                        
                        for time, value in fmfl.data:
                            data['time_s'].append(time)
                            # Assuming unit area for now - adjust if needed
                            data['flux_kg_m2_s'].append(abs(value))
                        
                        print("  Extracted {} points from history".format(len(data['time_s'])))
                        return data
            
            return None
            
        except:
            return None
    
    def _try_field_extraction(self, odb):
        """Try to extract from field output"""
        try:
            assembly = odb.rootAssembly
            instance_name = assembly.instances.keys()[0]
            instance = assembly.instances[instance_name]
            
            step = odb.steps['Permeation']
            
            data = {
                'time_s': [],
                'flux_kg_m2_s': []
            }
            
            # Get mesh information
            nodes = instance.nodes
            print("  Instance has {} nodes".format(len(nodes)))
            
            # Find bottom nodes (y ~ 0)
            bottom_node_labels = []
            tolerance = 1.0  # nm
            for node in nodes:
                if abs(node.coordinates[1]) < tolerance:
                    bottom_node_labels.append(node.label)
            
            print("  Found {} bottom nodes".format(len(bottom_node_labels)))
            
            if not bottom_node_labels:
                print("  WARNING: No bottom nodes found, trying alternative method")
                # Find minimum y-coordinate
                y_coords = [node.coordinates[1] for node in nodes]
                y_min = min(y_coords)
                print("  Minimum y-coordinate: {}".format(y_min))
                for node in nodes:
                    if abs(node.coordinates[1] - y_min) < tolerance:
                        bottom_node_labels.append(node.label)
                print("  Found {} nodes at y_min".format(len(bottom_node_labels)))
            
            # Extract flux at each frame
            frame_count = 0
            for frame in step.frames:
                time = frame.frameValue
                frame_count += 1
                
                if 'MFL' in frame.fieldOutputs:
                    mfl = frame.fieldOutputs['MFL']
                    
                    # Get bottom surface flux
                    bottom_flux = []
                    
                    # Method 1: Direct value access
                    for value in mfl.values:
                        # Check different possible attributes
                        node_label = None
                        if hasattr(value, 'nodeLabel') and value.nodeLabel is not None:
                            node_label = value.nodeLabel
                        elif hasattr(value, 'node') and value.node is not None:
                            node_label = value.node.label
                        
                        if node_label and node_label in bottom_node_labels:
                            # MFL is a vector (MFL1, MFL2, MFL3) for 3D or (MFL1, MFL2) for 2D
                            # For vertical flux, we want the y-component
                            if len(value.data) >= 2:
                                flux_y = value.data[1]  # y-component
                                bottom_flux.append(abs(flux_y))
                    
                    # Method 2: If method 1 fails, try getting subset
                    if not bottom_flux and hasattr(mfl, 'getSubset'):
                        try:
                            # Create node set for bottom nodes
                            from odbAccess import NodeSet
                            bottom_set = instance.NodeSet(name='temp_bottom', nodes=bottom_node_labels)
                            subset = mfl.getSubset(region=bottom_set)
                            for value in subset.values:
                                if len(value.data) >= 2:
                                    bottom_flux.append(abs(value.data[1]))
                        except:
                            pass
                    
                    if bottom_flux:
                        # Average flux
                        avg_flux = sum(bottom_flux) / len(bottom_flux)
                        data['time_s'].append(time)
                        data['flux_kg_m2_s'].append(avg_flux)
                        
                        if frame_count <= 3:  # Debug first few frames
                            print("    Frame {}: t={:.1f}s, {} flux values, avg={:.3e}".format(
                                frame_count, time, len(bottom_flux), avg_flux))
                    else:
                        print("    Frame {}: No flux values extracted".format(frame_count))
                        
                elif 'NNC' in frame.fieldOutputs:
                    print("  Frame has NNC (concentration) but not MFL (flux)")
                else:
                    print("  Frame has neither MFL nor NNC")
            
            if data['time_s']:
                print("  Extracted {} time points from {} frames".format(
                    len(data['time_s']), frame_count))
                return data
            else:
                print("  No data extracted from {} frames".format(frame_count))
            
            return None
            
        except Exception as e:
            import traceback
            print("  Field extraction error: {}".format(str(e)))
            print("  Traceback:")
            traceback.print_exc()
            return None
    
    def _save_csv(self, data, csv_path):
        """Save data to CSV without pandas"""
        with open(str(csv_path), 'w') as f:
            # Header
            f.write('time_s,flux_kg_m2_s,flux_g_m2_day\n')
            
            # Data rows
            for i in range(len(data['time_s'])):
                time = data['time_s'][i]
                flux_kg = data['flux_kg_m2_s'][i]
                flux_g = flux_kg * 1000 * 86400  # Convert to g/m²/day
                f.write('{},{},{}\n'.format(time, flux_kg, flux_g))
    
    def process_batch(self, job_list_file):
        """Process multiple ODB files from a list"""
        with open(job_list_file, 'r') as f:
            job_list = f.readlines()
        
        success_count = 0
        for line in job_list:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    job_name = parts[0]
                    odb_path = parts[1]
                    
                    if self.extract_flux_data(odb_path, job_name):
                        success_count += 1
        
        print("\nExtracted {}/{} ODB files successfully".format(
            success_count, len(job_list)))


# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract flux data from ODB files')
    parser.add_argument('--odb', help='Single ODB file path')
    parser.add_argument('--job', help='Job name for single ODB')
    parser.add_argument('--batch', help='Batch file with job list')
    parser.add_argument('--output', default='extracted_data', help='Output directory')
    
    args = parser.parse_args()
    
    extractor = ODBExtractor(args.output)
    
    if args.odb and args.job:
        # Single ODB
        extractor.extract_flux_data(args.odb, args.job)
    elif args.batch:
        # Batch processing
        extractor.process_batch(args.batch)
    else:
        print("Usage:")
        print("  Single: abaqus python odb_extractor.py --odb file.odb --job JobName")
        print("  Batch:  abaqus python odb_extractor.py --batch job_list.txt")