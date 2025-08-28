#!/usr/bin/env python
"""
Simplified post-processor that focuses on basic functionality
Creates output files to confirm it's working
"""

import os
import sys
import json
from pathlib import Path

# Create output directory first thing
try:
    output_dir = Path('simulation_results')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'json').mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    
    # Create a status file to confirm the script ran
    with open(output_dir / 'status.txt', 'w') as f:
        f.write("Post processor started successfully\n")
        f.write("Working directory: {}\n".format(os.getcwd()))
        f.write("Python path: {}\n".format(sys.executable))
        f.write("Arguments: {}\n".format(sys.argv))
except Exception as e:
    # If we can't even create directories, write to current directory
    with open('postprocessor_error.txt', 'w') as f:
        f.write("Error creating output directories: {}\n".format(str(e)))

try:
    # Try ABAQUS imports
    from abaqus import *
    from abaqusConstants import *
    import odbAccess
    ABAQUS_ENV = True
    
    with open(output_dir / 'status.txt', 'a') as f:
        f.write("ABAQUS imports successful\n")
        
except ImportError as e:
    ABAQUS_ENV = False
    with open(output_dir / 'status.txt', 'a') as f:
        f.write("ABAQUS imports failed: {}\n".format(str(e)))

try:
    import numpy as np
    with open(output_dir / 'status.txt', 'a') as f:
        f.write("NumPy import successful\n")
except ImportError as e:
    with open(output_dir / 'status.txt', 'a') as f:
        f.write("NumPy import failed: {}\n".format(str(e)))

def simple_analysis(job_name, odb_path):
    """Simple analysis function with extensive logging"""
    log_file = output_dir / 'analysis.log'
    
    with open(log_file, 'w') as f:
        f.write("=== SIMPLE ANALYSIS LOG ===\n")
        f.write("Job: {}\n".format(job_name))
        f.write("ODB path: {}\n".format(odb_path))
        f.write("ODB exists: {}\n".format(os.path.exists(odb_path)))
        f.write("ABAQUS env: {}\n".format(ABAQUS_ENV))
        
        if not os.path.exists(odb_path):
            f.write("ERROR: ODB file not found\n")
            f.write("Current directory contents:\n")
            for file in os.listdir('.'):
                if job_name in file:
                    f.write("  {}\n".format(file))
            return None
        
        if not ABAQUS_ENV:
            f.write("Creating mock results (no ABAQUS)\n")
            result = {
                'job_name': job_name,
                'breakthrough_time': 1800.0,
                'steady_state_flux': 1e-12,
                'lag_time': 900.0,
                'status': 'mock_data'
            }
        else:
            f.write("Attempting to open ODB file\n")
            try:
                odb = odbAccess.openOdb(path=odb_path)
                f.write("ODB opened successfully\n")
                
                f.write("Available steps: {}\n".format(list(odb.steps.keys())))
                
                if 'Permeation' in odb.steps:
                    step = odb.steps['Permeation']
                    f.write("Found Permeation step\n")
                    f.write("History regions: {}\n".format(list(step.historyRegions.keys())))
                    
                    # Try to get some basic info
                    frames = step.frames
                    f.write("Number of frames: {}\n".format(len(frames)))
                    
                    if len(frames) > 0:
                        last_frame = frames[-1]
                        f.write("Final time: {}\n".format(last_frame.frameValue))
                        if hasattr(last_frame, 'fieldOutputs'):
                            f.write("Field outputs: {}\n".format(list(last_frame.fieldOutputs.keys())))
                
                odb.close()
                f.write("ODB closed successfully\n")
                
                # Create basic result
                result = {
                    'job_name': job_name,
                    'breakthrough_time': None,
                    'steady_state_flux': None,
                    'lag_time': None,
                    'status': 'odb_opened_successfully'
                }
                
            except Exception as e:
                f.write("Error opening ODB: {}\n".format(str(e)))
                import traceback
                f.write("Traceback:\n{}\n".format(traceback.format_exc()))
                return None
    
    # Save result to JSON
    try:
        result_file = output_dir / 'json' / 'simple_results.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        with open(log_file, 'a') as f:
            f.write("Results saved to: {}\n".format(result_file))
            
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write("Error saving results: {}\n".format(str(e)))
    
    return result

if __name__ == '__main__':
    try:
        with open(output_dir / 'status.txt', 'a') as f:
            f.write("Main execution started\n")
            f.write("Command line args: {}\n".format(sys.argv))
        
        # Simple argument parsing
        job_name = None
        odb_path = None
        
        for i, arg in enumerate(sys.argv):
            if arg == '--job' and i + 1 < len(sys.argv):
                job_name = sys.argv[i + 1]
            elif arg == '--odb' and i + 1 < len(sys.argv):
                odb_path = sys.argv[i + 1]
        
        with open(output_dir / 'status.txt', 'a') as f:
            f.write("Parsed job: {}\n".format(job_name))
            f.write("Parsed odb: {}\n".format(odb_path))
        
        if job_name and odb_path:
            result = simple_analysis(job_name, odb_path)
            with open(output_dir / 'status.txt', 'a') as f:
                f.write("Analysis completed: {}\n".format(result is not None))
        else:
            with open(output_dir / 'status.txt', 'a') as f:
                f.write("Missing job name or ODB path\n")
                
        with open(output_dir / 'status.txt', 'a') as f:
            f.write("=== SCRIPT COMPLETED ===\n")
            
    except Exception as e:
        try:
            with open(output_dir / 'status.txt', 'a') as f:
                f.write("FATAL ERROR: {}\n".format(str(e)))
                import traceback
                f.write("Traceback:\n{}\n".format(traceback.format_exc()))
        except:
            with open('fatal_error.txt', 'w') as f:
                f.write("FATAL ERROR: {}\n".format(str(e)))
                