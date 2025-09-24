"""
ODB data extractor - runs in ABAQUS Python environment
Extracts flux data to CSV with proper file logging
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# ABAQUS imports
try:
    from odbAccess import openOdb
    from abaqusConstants import *
    ABAQUS_ENV = True
except ImportError:
    print("ERROR: This script must be run with ABAQUS Python")
    sys.exit(1)

class ODBExtractor:
    def __init__(self, output_dir=None, save_raw_json=True, log_dir=None, verbose=True):
        """
        Initialize ODB extractor with logging
        
        Args:
            output_dir: Base directory for output (e.g., simulations/sim_name/extracted_data)
            save_raw_json: Whether to save raw extraction data as JSON
            log_dir: Directory for log files (default: simulation_results/logs)
            verbose: Whether to print messages to console in addition to log file
        """
        self.verbose = verbose
        
        # Setup output directory
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
        
        # Setup logging
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path('simulation_results') / 'logs'
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f'odb_extractor_{timestamp}.log'
        
        # Clear/create log file
        with open(self.log_file, 'w') as f:
            f.write(f"ODB Extractor Log - {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n")
        
        self.log("ODB Extractor initialized")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Log file: {self.log_file}")
    
    def log(self, message, level="INFO"):
        """Log message to file and optionally to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(formatted_message + "\n")
        except Exception as e:
            # If we can't write to log, at least print
            print(f"WARNING: Could not write to log: {e}")
        
        # Print to console if verbose
        if self.verbose:
            print(message)

    def extract_flux_data(self, odb_path, job_name=None):
        """
        Extract flux vs time from ODB and save to CSV
        
        Args:
            odb_path: Path to ODB file
            job_name: Job name (extracted from ODB path if not provided)
        
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
        
        self.log("=" * 60)
        self.log(f"Starting extraction for job: {job_name}")
        self.log(f"ODB path: {odb_path}")
        
        try:
            # Open ODB
            self.log("Opening ODB file...")
            odb = openOdb(str(odb_path))
            self.log(f"  ODB opened successfully")
            
            # Try field extraction
            self.log("Extracting NNC field data...")
            data = self._try_field_extraction(odb)
            
            # Close ODB
            odb.close()
            self.log("  ODB closed")
            
            if data:
                # Save NNC gradient data
                csv_path = self.nnc_dir / f'{job_name}_flux.csv'
                self._save_csv(data, csv_path)
                result['output_files']['nnc_csv'] = str(csv_path)
                
                self.log(f"  Saved NNC data: {csv_path}")
                
                # Save raw JSON if requested
                if self.save_raw_json:
                    json_path = self.json_dir / f'{job_name}_raw.json'
                    self._save_raw_json(data, json_path, result)
                    result['output_files']['raw_json'] = str(json_path)
                    self.log(f"  Saved raw JSON: {json_path}")
                
                result['success'] = True
                result['data_points'] = len(data['time_s'])
                
                # Add extraction summary
                result['summary'] = {
                    'frames_extracted': len(data['time_s']),
                    'time_range': [data['time_s'][0], data['time_s'][-1]],
                    'gradient_range': [min(data['gradient']), max(data['gradient'])]
                }
                
                self.log(f"Extraction completed successfully")
                self.log(f"  Frames extracted: {len(data['time_s'])}")
                self.log(f"  Time range: {data['time_s'][0]:.1f} to {data['time_s'][-1]:.1f} s")
                
                return result
            else:
                error_msg = "No flux data extracted from ODB"
                result['errors'].append(error_msg)
                self.log(error_msg, "ERROR")
                return result
                
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            result['errors'].append(error_msg)
            self.log(error_msg, "ERROR")
            
            # Log traceback for debugging
            import traceback
            self.log("Traceback:", "ERROR")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}", "ERROR")
            
            return result
    
    def _try_field_extraction(self, odb):
        """Extract NNC values for manual flux calculation"""
        try:
            self.log("  Getting assembly and instance information...")
            assembly = odb.rootAssembly
            instance_names = assembly.instances.keys()
            
            if not instance_names:
                self.log("    ERROR: No instances found in assembly", "ERROR")
                return None
            
            instance_name = instance_names[0]
            instance = assembly.instances[instance_name]
            self.log(f"    Using instance: {instance_name}")
            
            # Check for Permeation step
            if 'Permeation' not in odb.steps:
                self.log("    ERROR: 'Permeation' step not found", "ERROR")
                self.log(f"    Available steps: {odb.steps.keys()}")
                return None
            
            step = odb.steps['Permeation']
            self.log(f"    Found Permeation step with {len(step.frames)} frames")
            
            # Initialize data structure
            data = {
                'time_s': [],
                'nnc_bottom': [],
                'nnc_second': [],
                'gradient': []
            }
            
            # Get mesh information and find second-lowest row
            self.log("  Analyzing mesh structure...")
            nodes = instance.nodes
            y_coords = [node.coordinates[1] for node in nodes]
            y_min = min(y_coords)
            
            # Find unique y-coordinates and sort them
            unique_y = sorted(list(set(y_coords)))
            if len(unique_y) < 2:
                self.log("    ERROR: Not enough y-levels for gradient calculation", "ERROR")
                return None
            
            y_bottom = unique_y[0]  # Lowest row
            y_second = unique_y[1]  # Second-lowest row
            tolerance = 0.1  # nm
            
            self.log(f"    Bottom row: y = {y_bottom:.3f} nm")
            self.log(f"    Second row: y = {y_second:.3f} nm")
            self.log(f"    Distance: {y_second - y_bottom:.3f} nm")
            
            # Create node label maps for fast lookup
            bottom_nodes = set()
            second_nodes = set()
            
            for node in nodes:
                y = node.coordinates[1]
                if abs(y - y_bottom) < tolerance:
                    bottom_nodes.add(node.label)
                elif abs(y - y_second) < tolerance:
                    second_nodes.add(node.label)
            
            self.log(f"    Found {len(bottom_nodes)} bottom nodes, {len(second_nodes)} second-row nodes")
            
            # Extract NNC at each frame
            frame_count = 0
            total_frames = len(step.frames)
            
            self.log(f"  Processing {total_frames} frames...")
            
            for frame in step.frames:
                time = frame.frameValue
                frame_count += 1
                
                if 'NNC11' not in frame.fieldOutputs:
                    self.log(f"    Frame {frame_count} has no NNC field", "WARNING")
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
                    
                    # Progress indicator
                    if frame_count % 10 == 0 or frame_count == total_frames:
                        self.log(f"    Processed frame {frame_count}/{total_frames}")
                    
                    # Debug first few frames
                    if frame_count <= 3:
                        self.log(f"      Frame {frame_count}: bottom={avg_bottom:.3e}, second={avg_second:.3e}, grad={gradient:.3e}", "DEBUG")
            
            if data['time_s']:
                self.log(f"  Extracted {len(data['time_s'])} time points for gradient calculation")
                return data
            else:
                self.log("  No NNC data extracted", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"  NNC extraction error: {str(e)}", "ERROR")
            import traceback
            for line in traceback.format_exc().splitlines():
                self.log(f"    {line}", "ERROR")
            return None

    def _save_csv(self, data, csv_path):
        """Save NNC data for manual flux calculation"""
        self.log(f"  Saving CSV data to {csv_path}...")
        
        try:
            with open(str(csv_path), 'w') as f:
                # Header for NNC data
                f.write('time_s,nnc_bottom,nnc_second,gradient_per_nm\n')
                
                # Data rows
                for i in range(len(data['time_s'])):
                    time = data['time_s'][i]
                    nnc_bottom = data['nnc_bottom'][i]
                    nnc_second = data['nnc_second'][i]
                    gradient = data['gradient'][i]
                    f.write(f'{time},{nnc_bottom},{nnc_second},{gradient}\n')
            
            self.log(f"    CSV saved successfully ({len(data['time_s'])} rows)")
            
        except Exception as e:
            self.log(f"    ERROR saving CSV: {str(e)}", "ERROR")
            raise

    def _save_raw_json(self, data, json_path, result_dict):
        """Save raw extraction data as JSON for debugging"""
        self.log(f"  Saving raw JSON data to {json_path}...")
        
        try:
            raw_data = {
                'extraction_info': {
                    'job_name': result_dict.get('job_name'),
                    'odb_path': result_dict.get('odb_path'),
                    'extraction_time': datetime.now().isoformat(),
                    'data_points': len(data['time_s'])
                },
                'time_data': data['time_s'],
                'nnc_bottom': data['nnc_bottom'],
                'nnc_second': data['nnc_second'],
                'gradient': data['gradient']
            }
            
            with open(str(json_path), 'w') as f:
                json.dump(raw_data, f, indent=2)
            
            self.log(f"    JSON saved successfully")
            
        except Exception as e:
            self.log(f"    ERROR saving JSON: {str(e)}", "ERROR")

    def process_batch(self, job_list_file, base_output_dir=None):
        """
        Process multiple ODB files from a list
        
        Args:
            job_list_file: File containing job list
            base_output_dir: Base directory for all outputs
        
        Returns:
            Summary dict of all extractions
        """
        self.log("=" * 60)
        self.log("BATCH ODB EXTRACTION")
        self.log("=" * 60)
        
        try:
            with open(job_list_file, 'r') as f:
                job_list = f.readlines()
            
            self.log(f"Loaded job list from: {job_list_file}")
            
        except Exception as e:
            self.log(f"ERROR loading job list: {str(e)}", "ERROR")
            return None
        
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
                    
                    self.log(f"\nProcessing: {job_name}")
                    result = self.extract_flux_data(odb_path, job_name)
                    results.append(result)
                    
                    if result['success']:
                        success_count += 1
        
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("EXTRACTION SUMMARY")
        self.log("=" * 60)
        self.log(f"Processed: {success_count}/{len(results)}")
        
        for result in results:
            status = "SUCCESS" if result['success'] else "FAILED"
            self.log(f"  {result['job_name']}: {status}")
            if not result['success'] and result['errors']:
                self.log(f"    Error: {result['errors'][0]}")
        
        # Save batch summary
        summary = {
            'total': len(results),
            'successful': success_count,
            'failed': len(results) - success_count,
            'results': results,
            'log_file': str(self.log_file)
        }
        
        return summary
    
    def get_log_file_path(self):
        """Return the path to the current log file"""
        return str(self.log_file)

# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract flux data from ODB files')
    parser.add_argument('--odb', help='Single ODB file path')
    parser.add_argument('--job', help='Job name for single ODB')
    parser.add_argument('--batch', help='Batch file with job list')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--simulation', help='Simulation name (for organized output)')
    parser.add_argument('--log_dir', help='Directory for log files')
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
    
    # Determine log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    elif args.simulation:
        log_dir = Path('simulations') / args.simulation / 'logs'
    else:
        log_dir = Path('simulation_results') / 'logs'
    
    # Create extractor
    extractor = ODBExtractor(
        output_dir=output_dir,
        save_raw_json=not args.no_json,
        log_dir=log_dir,
        verbose=not args.quiet
    )
    
    # Log command line arguments
    extractor.log("Command line arguments:")
    for arg, value in vars(args).items():
        extractor.log(f"  {arg}: {value}")
    
    if args.odb:
        # Single ODB processing
        if not Path(args.odb).exists():
            extractor.log(f"ERROR: ODB file not found: {args.odb}", "ERROR")
            sys.exit(1)
        
        job_name = args.job if args.job else Path(args.odb).stem
        result = extractor.extract_flux_data(args.odb, job_name)
        
        # Save extraction result
        result_file = output_dir / 'extraction_result.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        extractor.log(f"Extraction result saved to: {result_file}")
        
        if result['success']:
            extractor.log("\nExtraction successful")
            extractor.log("Output files:")
            for key, path in result['output_files'].items():
                extractor.log(f"  {key}: {path}")
            extractor.log(f"Log file: {extractor.get_log_file_path()}")
            sys.exit(0)
        else:
            extractor.log("\nExtraction failed", "ERROR")
            for error in result['errors']:
                extractor.log(f"  ERROR: {error}", "ERROR")
            extractor.log(f"Log file: {extractor.get_log_file_path()}")
            sys.exit(1)
            
    elif args.batch:
        # Batch processing
        if not Path(args.batch).exists():
            extractor.log(f"ERROR: Batch file not found: {args.batch}", "ERROR")
            sys.exit(1)
        
        summary = extractor.process_batch(args.batch, base_output_dir=args.output)
        
        if summary:
            # Save batch summary
            summary_file = output_dir / 'batch_extraction_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            extractor.log(f"\nBatch summary saved to: {summary_file}")
            extractor.log(f"Log file: {extractor.get_log_file_path()}")
            
            if summary['successful'] > 0:
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            extractor.log("Batch processing failed", "ERROR")
            sys.exit(1)
    else:
        extractor.log("Usage:")
        extractor.log("  Single: abaqus python odb_extractor.py --odb file.odb [--job JobName] [--simulation sim_name]")
        extractor.log("  Batch:  abaqus python odb_extractor.py --batch job_list.txt [--output base_dir]")
        sys.exit(1)
        