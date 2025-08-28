#!/usr/bin/env python
"""
Minimal ABAQUS data extractor - runs in ABAQUS environment
Only extracts raw data and saves to JSON, no analysis
"""

import os
import sys
import json
from pathlib import Path

# Create output directory
output_dir = Path('simulation_results')
output_dir.mkdir(exist_ok=True)
raw_data_dir = output_dir / 'raw_data'
raw_data_dir.mkdir(exist_ok=True)
log_dir = output_dir / 'logs'
log_dir.mkdir(exist_ok=True)

def log_message(message):
    """Simple logging to file"""
    with open(log_dir / 'extractor.log', 'a') as f:
        f.write("{}\n".format(message))

# Clear previous log
log_file = log_dir / 'extractor.log'
if log_file.exists():
    log_file.unlink()

log_message("=== ABAQUS DATA EXTRACTOR ===")

try:
    from abaqus import *
    from abaqusConstants import *
    import odbAccess
    log_message("ABAQUS imports successful")
    ABAQUS_ENV = True
except ImportError as e:
    log_message("ABAQUS import failed: {}".format(str(e)))
    ABAQUS_ENV = False

def extract_raw_data(job_name, odb_path):
    """Extract raw time and flux data from ODB file"""
    log_message("Extracting data from: {}".format(odb_path))
    
    if not ABAQUS_ENV:
        log_message("No ABAQUS environment - cannot extract data")
        return None
    
    if not os.path.exists(odb_path):
        log_message("ODB file not found: {}".format(odb_path))
        return None
    
    try:
        # Open ODB
        odb = odbAccess.openOdb(path=odb_path)
        log_message("ODB opened successfully")
        
        # Get step and frames
        step = odb.steps['Permeation']
        frames = step.frames
        log_message("Found {} frames".format(len(frames)))
        
        # Extract data from each frame
        extraction_data = {
            'job_name': job_name,
            'odb_path': odb_path,
            'total_frames': len(frames),
            'times': [],
            'flux_data': [],
            'extraction_info': {
                'mfl_values_per_frame': 0,
                'successful_frames': 0,
                'failed_frames': 0
            }
        }
        
        for i, frame in enumerate(frames):
            try:
                time = frame.frameValue
                extraction_data['times'].append(time)
                
                if 'MFL' in frame.fieldOutputs:
                    mfl_field = frame.fieldOutputs['MFL']
                    
                    if i == 0:  # Record info from first frame
                        extraction_data['extraction_info']['mfl_values_per_frame'] = len(mfl_field.values)
                    
                    # Extract flux values - keep it simple
                    flux_values = []
                    valid_count = 0
                    
                    for value in mfl_field.values:
                        if hasattr(value, 'data'):
                            try:
                                # Handle both scalar and vector data
                                if hasattr(value.data, '__len__') and len(value.data) > 0:
                                    # Vector data - compute magnitude
                                    magnitude_sq = 0.0
                                    for component in value.data:
                                        magnitude_sq += component * component
                                    data_val = magnitude_sq ** 0.5
                                else:
                                    # Scalar data
                                    data_val = float(value.data)
                                
                                flux_values.append(abs(data_val))
                                valid_count += 1
                                
                            except Exception:
                                flux_values.append(0.0)
                    
                    # Store raw flux values for this frame
                    extraction_data['flux_data'].append({
                        'frame': i,
                        'time': time,
                        'flux_values': flux_values,
                        'valid_count': valid_count,
                        'total_count': len(mfl_field.values)
                    })
                    
                    extraction_data['extraction_info']['successful_frames'] += 1
                    
                    # Log progress every 10 frames
                    if i % 10 == 0:
                        log_message("Frame {}: time={}, valid_values={}".format(i, time, valid_count))
                
                else:
                    log_message("Frame {}: No MFL field found".format(i))
                    extraction_data['flux_data'].append({
                        'frame': i,
                        'time': time,
                        'flux_values': [],
                        'valid_count': 0,
                        'total_count': 0
                    })
                    extraction_data['extraction_info']['failed_frames'] += 1
            
            except Exception as e:
                log_message("Error processing frame {}: {}".format(i, str(e)))
                extraction_data['extraction_info']['failed_frames'] += 1
        
        # Close ODB
        odb.close()
        log_message("ODB closed successfully")
        
        # Save raw data
        output_file = raw_data_dir / "{}_raw_data.json".format(job_name)
        with open(output_file, 'w') as f:
            json.dump(extraction_data, f, indent=2)
        
        log_message("Raw data saved to: {}".format(output_file))
        log_message("Extraction summary:")
        log_message("  Total frames: {}".format(extraction_data['total_frames']))
        log_message("  Successful: {}".format(extraction_data['extraction_info']['successful_frames']))
        log_message("  Failed: {}".format(extraction_data['extraction_info']['failed_frames']))
        
        return output_file
        
    except Exception as e:
        log_message("Error during extraction: {}".format(str(e)))
        import traceback
        log_message("Full traceback: {}".format(traceback.format_exc()))
        return None

if __name__ == '__main__':
    # Simple argument parsing
    job_name = None
    odb_path = None
    
    for i, arg in enumerate(sys.argv):
        if arg == '--job' and i + 1 < len(sys.argv):
            job_name = sys.argv[i + 1]
        elif arg == '--odb' and i + 1 < len(sys.argv):
            odb_path = sys.argv[i + 1]
    
    log_message("Job: {}".format(job_name))
    log_message("ODB: {}".format(odb_path))
    
    if job_name and odb_path:
        result_file = extract_raw_data(job_name, odb_path)
        
        if result_file:
            log_message("=== EXTRACTION COMPLETE ===")
            log_message("Next step: Run python analyzer on {}".format(result_file))
            
            # Create a status file for the Python script
            with open(log_dir / 'extraction_status.json', 'w') as f:
                json.dump({
                    'status': 'success',
                    'job_name': job_name,
                    'raw_data_file': str(result_file),
                    'extraction_complete': True
                }, f)
        else:
            log_message("=== EXTRACTION FAILED ===")
            with open(log_dir / 'extraction_status.json', 'w') as f:
                json.dump({
                    'status': 'failed',
                    'job_name': job_name,
                    'extraction_complete': False
                }, f)
    else:
        log_message("ERROR: Missing job name or ODB path")
        log_message("Usage: python abaqus_extractor.py --job <n> --odb <path>")
