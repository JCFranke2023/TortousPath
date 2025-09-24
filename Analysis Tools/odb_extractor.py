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
    def __init__(self, output_dir=None, save_raw_json=True):
        """
        Initialize ODB extractor
        
        Args:
            output_dir: Base directory for output (e.g., simulations/sim_name/extracted_data)
            save_raw_json: Whether to save raw extraction data as JSON
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path('extracted_data')
        
        self.save_raw_json = save_raw_json
        
        # Create subdirectories
        self.nnc_dir = self.output_dir / 'nnc_gradients'
        self.json_dir = self.output_dir / 'raw_json'
        
        self.nnc_dir.mkdir(parents=True, exist_ok=True)
        if self.save_raw_json:
            self.json_dir.mkdir(parents=True, exist_ok=True)    

    def extract_flux_data(self, odb_path, job_name=None, verbose=True):
        """
        Extract flux vs time from ODB and save to CSV
        
        Args:
            odb_path: Path to ODB file
            job_name: Job name (extracted from ODB path if not provided)
            verbose: Print progress messages
        
        Returns:
            Dict with extraction status and file paths
        """
        result = {
            'success': False,
            'job_name': job_name,
            'odb_path': str(odb_path),
            'output_files': {},
            'errors': []
        }
        
        # Extract job name from path if not provided
        if not job_name:
            job_name = Path(odb_path).stem
            result['job_name'] = job_name
        
        if verbose:
            print("Extracting from: {}".format(odb_path))
        
        try:
            odb = openOdb(str(odb_path))
            
            # Try field extraction
            data = self._try_field_extraction(odb, verbose)
            
            odb.close()
            
            if data:
                # Save NNC gradient data
                csv_path = self.nnc_dir / '{}_flux.csv'.format(job_name)
                self._save_csv(data, csv_path)
                result['output_files']['nnc_csv'] = str(csv_path)
                
                if verbose:
                    print("  Saved NNC data: {}".format(csv_path))
                
                # Save raw JSON if requested
                if self.save_raw_json:
                    json_path = self.json_dir / '{}_raw.json'.format(job_name)
                    self._save_raw_json(data, json_path, result)
                    result['output_files']['raw_json'] = str(json_path)
                    
                    if verbose:
                        print("  Saved raw JSON: {}".format(json_path))
                
                result['success'] = True
                result['data_points'] = len(data['time_s'])
                
                # Add extraction summary
                result['summary'] = {
                    'frames_extracted': len(data['time_s']),
                    'time_range': [data['time_s'][0], data['time_s'][-1]],
                    'gradient_range': [min(data['gradient']), max(data['gradient'])]
                }
                
                return result
            else:
                result['errors'].append("No flux data extracted from ODB")
                if verbose:
                    print("  ERROR: No flux data extracted")
                return result
                
        except Exception as e:
            error_msg = "Extraction failed: {}".format(str(e))
            result['errors'].append(error_msg)
            if verbose:
                print("  ERROR: {}".format(str(e)))
            return result
    
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
            total_frames = len(step.frames)
            
            print("  Processing {} frames...".format(total_frames))
            
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
                            if frame_count == 30:
                                print(f'{value.nodeLabel=}, {value.data=}')
                            second_concentrations.append(value.data)
                
                if bottom_concentrations and second_concentrations:
                    avg_bottom = sum(bottom_concentrations) / len(bottom_concentrations)
                    avg_second = sum(second_concentrations) / len(second_concentrations)
                    gradient = (avg_second - avg_bottom) / (y_second - y_bottom)
                    
                    data['time_s'].append(time)
                    data['nnc_bottom'].append(avg_bottom)
                    data['nnc_second'].append(avg_second)
                    data['gradient'].append(gradient)
                    
                    # Progress indicator
                    if frame_count % 10 == 0 or frame_count == total_frames:
                        print("    Processed frame {}/{}".format(frame_count, total_frames))
                    
                    if frame_count <= 3:  # Debug first few frames
                        print("      Frame {}: bottom={:.3e}, second={:.3e}, grad={:.3e}".format(
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

    def _save_raw_json(self, data, json_path, result_dict):
        """Save raw extraction data as JSON for debugging"""
        import json
        
        raw_data = {
            'extraction_info': result_dict,
            'time_data': data['time_s'],
            'nnc_bottom': data['nnc_bottom'],
            'nnc_second': data['nnc_second'],
            'gradient': data['gradient']
        }
        
        with open(str(json_path), 'w') as f:
            json.dump(raw_data, f, indent=2)

    def process_batch(self, job_list_file, base_output_dir=None):
        """
        Process multiple ODB files from a list
        
        Args:
            job_list_file: File containing job list
            base_output_dir: Base directory for all outputs
        
        Returns:
            Summary dict of all extractions
        """
        print("\n=== BATCH ODB EXTRACTION ===")
        
        with open(job_list_file, 'r') as f:
            job_list = f.readlines()
        
        results = []
        success_count = 0
        
        for line in job_list:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    job_name = parts[0].strip()
                    odb_path = parts[1].strip()
                    
                    # Set output directory if specified
                    if len(parts) >= 3 and base_output_dir:
                        sim_name = parts[2].strip()
                        self.output_dir = Path(base_output_dir) / sim_name / 'extracted_data'
                        self.nnc_dir = self.output_dir / 'nnc_gradients'
                        self.json_dir = self.output_dir / 'raw_json'
                        self.nnc_dir.mkdir(parents=True, exist_ok=True)
                        if self.save_raw_json:
                            self.json_dir.mkdir(parents=True, exist_ok=True)
                    
                    print("\nProcessing: {}".format(job_name))
                    result = self.extract_flux_data(odb_path, job_name, verbose=True)
                    results.append(result)
                    
                    if result['success']:
                        success_count += 1
        
        # Print summary
        print("\n=== EXTRACTION SUMMARY ===")
        print("Processed: {}/{}".format(success_count, len(results)))
        
        for result in results:
            status = "SUCCESS" if result['success'] else "FAILED"
            print("  {}: {}".format(result['job_name'], status))
            if not result['success'] and result['errors']:
                print("    Error: {}".format(result['errors'][0]))
        
        return {
            'total': len(results),
            'successful': success_count,
            'failed': len(results) - success_count,
            'results': results
        }

# Main execution
if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Extract flux data from ODB files')
    parser.add_argument('--odb', help='Single ODB file path')
    parser.add_argument('--job', help='Job name for single ODB')
    parser.add_argument('--batch', help='Batch file with job list')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--simulation', help='Simulation name (for organized output)')
    parser.add_argument('--no-json', action='store_true', help='Skip saving raw JSON')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.simulation:
        output_dir = Path('simulations') / args.simulation / 'extracted_data'
    elif args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('extracted_data')
    
    # Create extractor
    extractor = ODBExtractor(
        output_dir=output_dir,
        save_raw_json=not args.no_json
    )
    
    if args.odb:
        # Single ODB processing
        if not Path(args.odb).exists():
            print("ERROR: ODB file not found: {}".format(args.odb))
            sys.exit(1)
        
        job_name = args.job if args.job else Path(args.odb).stem
        result = extractor.extract_flux_data(
            args.odb, 
            job_name,
            verbose=not args.quiet
        )
        
        # Save extraction result
        result_file = output_dir / 'extraction_result.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        if result['success']:
            print("\nExtraction successful")
            print("Output files:")
            for key, path in result['output_files'].items():
                print("  {}: {}".format(key, path))
            sys.exit(0)
        else:
            print("\nExtraction failed")
            for error in result['errors']:
                print("  ERROR: {}".format(error))
            sys.exit(1)
            
    elif args.batch:
        # Batch processing
        if not Path(args.batch).exists():
            print("ERROR: Batch file not found: {}".format(args.batch))
            sys.exit(1)
        
        summary = extractor.process_batch(args.batch, base_output_dir=args.output)
        
        # Save batch summary
        summary_file = output_dir / 'batch_extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if summary['successful'] > 0:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("Usage:")
        print("  Single: abaqus python odb_extractor.py --odb file.odb [--job JobName] [--simulation sim_name]")
        print("  Batch:  abaqus python odb_extractor.py --batch job_list.txt [--output base_dir]")
        sys.exit(1)
        