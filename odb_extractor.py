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
    
    def _try_field_extraction(self, odb):
        """Extract NNC values for manual flux calculation"""
        try:
            assembly = odb.rootAssembly
            instance_name = assembly.instances.keys()[0]
            instance = assembly.instances[instance_name]
            
            step = odb.steps['Permeation']
            
            data = {
                'time_s': [],
                'nnc_bottom': [],
                'nnc_second': [],
                'gradient': []
            }
            
            # Get mesh information and find second-lowest row
            nodes = instance.nodes
            y_coords = [node.coordinates[1] for node in nodes]
            y_min = min(y_coords)
            
            # Find unique y-coordinates and sort them
            unique_y = sorted(list(set(y_coords)))
            if len(unique_y) < 2:
                print("  ERROR: Not enough y-levels for gradient calculation")
                return None
            
            y_bottom = unique_y[0]  # Lowest row
            y_second = unique_y[1]  # Second-lowest row
            tolerance = 0.1  # nm
            
            print("  Bottom row: y = {:.3f} nm".format(y_bottom))
            print("  Second row: y = {:.3f} nm".format(y_second))
            print("  Distance: {:.3f} nm".format(y_second - y_bottom))
            
            # Create node label maps for fast lookup
            bottom_nodes = set()
            second_nodes = set()
            
            for node in nodes:
                y = node.coordinates[1]
                if abs(y - y_bottom) < tolerance:
                    bottom_nodes.add(node.label)
                elif abs(y - y_second) < tolerance:
                    second_nodes.add(node.label)
            
            print("  Found {} bottom nodes, {} second-row nodes".format(
                len(bottom_nodes), len(second_nodes)))
            
            # Extract NNC at each frame
            frame_count = 0
            for frame in step.frames:
                time = frame.frameValue
                frame_count += 1
                
                if 'NNC11' not in frame.fieldOutputs:
                    print("  Frame {} has no NNC field".format(frame_count))
                    continue
                
                nnc = frame.fieldOutputs['NNC11']
                
                # Collect concentrations from both rows
                bottom_concentrations = []
                second_concentrations = []
                
                for value in nnc.values:
                    if hasattr(value, 'nodeLabel') and value.nodeLabel:
                        if value.nodeLabel in bottom_nodes:
                            bottom_concentrations.append(value.data)
                        elif value.nodeLabel in second_nodes:
                            second_concentrations.append(value.data)
                
                if bottom_concentrations and second_concentrations:
                    avg_bottom = sum(bottom_concentrations) / len(bottom_concentrations)
                    avg_second = sum(second_concentrations) / len(second_concentrations)
                    gradient = (avg_second - avg_bottom) / (y_second - y_bottom)
                    
                    data['time_s'].append(time)
                    data['nnc_bottom'].append(avg_bottom)
                    data['nnc_second'].append(avg_second)
                    data['gradient'].append(gradient)
                    
                    if frame_count <= 3:  # Debug first few frames
                        print("    Frame {}: bottom={:.3e}, second={:.3e}, grad={:.3e}".format(
                            frame_count, avg_bottom, avg_second, gradient))
            
            if data['time_s']:
                print("  Extracted {} time points for gradient calculation".format(len(data['time_s'])))
                return data
            else:
                print("  No NNC data extracted")
                return None
                
        except Exception as e:
            print("  NNC extraction error: {}".format(str(e)))
            return None

    def _save_csv(self, data, csv_path):
        """Save NNC data for manual flux calculation"""
        with open(str(csv_path), 'w') as f:
            # Header for NNC data
            f.write('time_s,nnc_bottom,nnc_second,gradient_per_nm\n')
            
            # Data rows
            for i in range(len(data['time_s'])):
                time = data['time_s'][i]
                nnc_bottom = data['nnc_bottom'][i]
                nnc_second = data['nnc_second'][i]
                gradient = data['gradient'][i]
                f.write('{},{},{},{}\n'.format(time, nnc_bottom, nnc_second, gradient))

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