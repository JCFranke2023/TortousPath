#!/usr/bin/env python
"""
Batch execution script for PECVD barrier coating simulation pipeline
Manages the complete workflow from simulation to analysis with stage control
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import time

class SimulationBatch:
    def __init__(self, simulation_name=None, stage=4, parameters=None):
        """
        Initialize batch runner
        
        Args:
            simulation_name: Descriptive name for this simulation
            stage: Execution stage (1-4)
            parameters: Dict with crack_width, crack_spacing, crack_offset
        """
        self.simulation_name = simulation_name
        self.stage = stage
        self.parameters = parameters or {}
        
        # Generate job name from parameters
        self.job_name = self._generate_job_name()
        
        # Setup directory structure
        self.sim_dir = Path('simulations') / self.simulation_name
        self.setup_directories()
        
        # Setup logging
        self.log_file = self.sim_dir / 'summary' / 'simulation_log.txt'
        self.start_time = datetime.now()
        
        # Stage definitions
        self.stages = {
            1: "Run simulation only",
            2: "Run simulation + organize files",
            3: "Run simulation + organize + extract ODB",
            4: "Run simulation + organize + extract + calculate flux (complete)"
        }
    
    def _generate_job_name(self):
        """Generate job name from parameters"""
        c = self.parameters.get('crack_width', 100)
        s = self.parameters.get('crack_spacing', 10000)
        o = self.parameters.get('crack_offset', 0.25)
        return f"Job_c{c:.0f}_s{s:.0f}_o{int(o*100)}"
    
    def setup_directories(self):
        """Create directory structure for this simulation"""
        directories = [
            self.sim_dir / 'models',
            self.sim_dir / 'abaqus_files' / 'inputs',
            self.sim_dir / 'abaqus_files' / 'logs',
            self.sim_dir / 'abaqus_files' / 'temp',
            self.sim_dir / 'extracted_data' / 'nnc_gradients',
            self.sim_dir / 'extracted_data' / 'raw_json',
            self.sim_dir / 'analysis' / 'flux_data',
            self.sim_dir / 'analysis' / 'metrics',
            self.sim_dir / 'analysis' / 'plots',
            self.sim_dir / 'summary'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def save_config(self):
        """Save simulation configuration"""
        config = {
            'simulation_name': self.simulation_name,
            'job_name': self.job_name,
            'parameters': self.parameters,
            'stage': self.stage,
            'start_time': self.start_time.isoformat(),
            'stages_to_run': [self.stages[i] for i in range(1, self.stage + 1)]
        }
        
        config_file = self.sim_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.log(f"Configuration saved to {config_file}")
    
    def run_stage_1_simulation(self):
        """Stage 1: Run ABAQUS simulation"""
        self.log("=" * 60)
        self.log("STAGE 1: Running ABAQUS simulation")
        self.log(f"Job name: {self.job_name}")
        self.log(f"Parameters: {self.parameters}")
        
        try:
            # Prepare parameter string for simulation_runner
            param_str = (f"--crack_width {self.parameters.get('crack_width', 100)} "
                        f"--crack_spacing {self.parameters.get('crack_spacing', 10000)} "
                        f"--crack_offset {self.parameters.get('crack_offset', 0.25)}")
            
            # Run simulation
            cmd = f"abaqus cae noGUI=simulation_runner.py -- {param_str}"
            self.log(f"Executing: {cmd}")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("Simulation completed successfully")
                
                # Check for ODB file
                odb_pattern = f"{self.job_name}.odb"
                if Path(odb_pattern).exists() or (Path('abaqus_files') / 'jobs' / self.job_name / odb_pattern).exists():
                    self.log(f"ODB file created: {odb_pattern}")
                    return True
                else:
                    self.log("WARNING: ODB file not found", "WARNING")
                    return False
            else:
                self.log(f"Simulation failed with return code {result.returncode}", "ERROR")
                if result.stderr:
                    self.log(f"Error output: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Exception in Stage 1: {str(e)}", "ERROR")
            return False
    
    def run_stage_2_organize(self):
        """Stage 2: Organize ABAQUS files"""
        self.log("=" * 60)
        self.log("STAGE 2: Organizing ABAQUS files")
        
        try:
            # Find all ABAQUS files for this job
            file_patterns = {
                'models': ['.cae', '.odb'],
                'abaqus_files/inputs': ['.inp'],
                'abaqus_files/logs': ['.msg', '.sta', '.dat', '.prt'],
                'abaqus_files/temp': ['.lck', '.023', '.ipm', '.rpy', '.rec']
            }
            
            files_moved = 0
            
            # Search in current directory and abaqus_files
            search_paths = [Path('.'), Path('abaqus_files') / 'jobs' / self.job_name]
            
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                    
                for subdir, extensions in file_patterns.items():
                    target_dir = self.sim_dir / subdir
                    
                    for ext in extensions:
                        pattern = f"{self.job_name}*{ext}"
                        for file_path in search_path.glob(pattern):
                            if file_path.is_file():
                                target_path = target_dir / file_path.name
                                
                                # Special handling for CAE and ODB - keep accessible
                                if ext in ['.cae', '.odb']:
                                    # Copy instead of move for primary files
                                    shutil.copy2(file_path, target_path)
                                    self.log(f"Copied {file_path.name} to {target_dir}")
                                else:
                                    shutil.move(str(file_path), str(target_path))
                                    self.log(f"Moved {file_path.name} to {target_dir}")
                                
                                files_moved += 1
            
            self.log(f"Organized {files_moved} files")
            return True
            
        except Exception as e:
            self.log(f"Exception in Stage 2: {str(e)}", "ERROR")
            return False
    
    def run_stage_3_extract(self):
        """Stage 3: Extract data from ODB"""
        self.log("=" * 60)
        self.log("STAGE 3: Extracting data from ODB")
        
        try:
            # Find ODB file
            odb_path = self.sim_dir / 'models' / f"{self.job_name}.odb"
            
            if not odb_path.exists():
                # Try alternate location
                odb_path = Path('abaqus_files') / 'jobs' / self.job_name / f"{self.job_name}.odb"
            
            if not odb_path.exists():
                self.log(f"ODB file not found: {odb_path}", "ERROR")
                return False
            
            self.log(f"Found ODB: {odb_path}")
            
            # Run ODB extractor
            output_dir = self.sim_dir / 'extracted_data' / 'nnc_gradients'
            cmd = f'abaqus python odb_extractor.py --odb "{odb_path}" --job {self.job_name} --output "{output_dir}"'
            
            self.log(f"Executing: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check for output file
                csv_file = output_dir / f"{self.job_name}_flux.csv"
                if csv_file.exists():
                    self.log(f"Data extracted to: {csv_file}")
                    return True
                else:
                    self.log("Extraction completed but no CSV found", "WARNING")
                    return False
            else:
                self.log(f"Extraction failed with return code {result.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Exception in Stage 3: {str(e)}", "ERROR")
            return False
    
    def run_stage_4_flux(self):
        """Stage 4: Calculate flux from extracted data"""
        self.log("=" * 60)
        self.log("STAGE 4: Calculating flux and metrics")
        
        try:
            # Find extracted CSV file
            csv_file = self.sim_dir / 'extracted_data' / 'nnc_gradients' / f"{self.job_name}_flux.csv"
            
            if not csv_file.exists():
                self.log(f"CSV file not found: {csv_file}", "ERROR")
                return False
            
            self.log(f"Processing: {csv_file}")
            
            # Run flux post-processor
            output_dir = self.sim_dir / 'analysis'
            cmd = f'python flux_postprocessor.py "{csv_file}" --output "{output_dir}"'
            
            self.log(f"Executing: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("Flux calculation completed")
                
                # Check for output files
                metrics_file = output_dir / f"{self.job_name}_metrics.json"
                plot_file = output_dir / 'plots' / f"{self.job_name}_analysis.png"
                
                if metrics_file.exists():
                    # Load and log key metrics
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    self.log("Key metrics:")
                    if 'metrics' in metrics:
                        m = metrics['metrics']
                        self.log(f"  Steady-state flux: {m.get('steady_state_flux', 'N/A')} g/(m²·day)")
                        self.log(f"  Breakthrough time: {m.get('breakthrough_time_h', 'N/A')} hours")
                        self.log(f"  Lag time: {m.get('lag_time_h', 'N/A')} hours")
                    
                    # Save summary metrics
                    self._save_summary_metrics(metrics)
                    
                return True
            else:
                self.log(f"Flux calculation failed with return code {result.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Exception in Stage 4: {str(e)}", "ERROR")
            return False
    
    def _save_summary_metrics(self, metrics):
        """Save summary metrics and generate report"""
        try:
            # Save metrics summary
            summary_file = self.sim_dir / 'summary' / 'metrics_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Generate markdown report
            report_file = self.sim_dir / 'summary' / 'report.md'
            self._generate_report(metrics, report_file)
            
        except Exception as e:
            self.log(f"Error saving summary: {str(e)}", "WARNING")
    
    def _generate_report(self, metrics, report_file):
        """Generate markdown report"""
        report = f"""# Simulation Report: {self.simulation_name}

## Configuration
- **Job Name**: {self.job_name}
- **Start Time**: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}
- **Execution Stage**: {self.stage} - {self.stages[self.stage]}

## Parameters
- **Crack Width**: {self.parameters.get('crack_width', 'N/A')} nm
- **Crack Spacing**: {self.parameters.get('crack_spacing', 'N/A')} nm  
- **Crack Offset**: {self.parameters.get('crack_offset', 'N/A')}

## Results
"""
        
        if 'metrics' in metrics:
            m = metrics['metrics']
            report += f"""
### Permeation Metrics
- **Steady-State Flux**: {m.get('steady_state_flux', 'N/A')} g/(m²·day)
- **Breakthrough Time**: {m.get('breakthrough_time_h', 'N/A')} hours
- **Lag Time**: {m.get('lag_time_h', 'N/A')} hours
- **Time to 90% SS**: {m.get('time_to_90_percent_h', 'N/A')} hours

### Crack Properties
- **Crack Fraction**: {m.get('crack_fraction', 'N/A')}
"""
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.log(f"Report generated: {report_file}")
    
    def run(self):
        """Execute the batch process up to specified stage"""
        self.log("=" * 60)
        self.log(f"PECVD BARRIER COATING SIMULATION BATCH")
        self.log(f"Simulation: {self.simulation_name}")
        self.log(f"Target stage: {self.stage} - {self.stages[self.stage]}")
        self.log("=" * 60)
        
        # Save configuration
        self.save_config()
        
        # Track success through stages
        success = True
        
        # Stage 1: Run simulation
        if self.stage >= 1 and success:
            success = self.run_stage_1_simulation()
            if not success:
                self.log("Stage 1 failed. Stopping execution.", "ERROR")
        
        # Stage 2: Organize files
        if self.stage >= 2 and success:
            time.sleep(1)  # Brief pause to ensure files are written
            success = self.run_stage_2_organize()
            if not success:
                self.log("Stage 2 failed. Stopping execution.", "ERROR")
        
        # Stage 3: Extract ODB
        if self.stage >= 3 and success:
            success = self.run_stage_3_extract()
            if not success:
                self.log("Stage 3 failed. Stopping execution.", "ERROR")
        
        # Stage 4: Calculate flux
        if self.stage >= 4 and success:
            success = self.run_stage_4_flux()
            if not success:
                self.log("Stage 4 failed. Stopping execution.", "ERROR")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.log("=" * 60)
        if success:
            self.log("BATCH EXECUTION COMPLETED SUCCESSFULLY")
        else:
            self.log("BATCH EXECUTION FAILED", "ERROR")
        
        self.log(f"Total duration: {duration}")
        self.log(f"Results saved in: {self.sim_dir}")
        
        return success


def get_user_input():
    """Interactive prompt for simulation parameters"""
    print("\n" + "=" * 60)
    print("PECVD BARRIER COATING SIMULATION SETUP")
    print("=" * 60)
    
    # Get simulation name
    default_name = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sim_name = input(f"Enter simulation name [{default_name}]: ").strip()
    if not sim_name:
        sim_name = default_name
    
    # Get execution stage
    print("\nExecution stages:")
    print("1 - Run simulation only")
    print("2 - Run + organize files")
    print("3 - Run + organize + extract ODB")
    print("4 - Run + organize + extract + calculate flux (complete)")
    
    stage = input("Select execution stage [4]: ").strip()
    stage = int(stage) if stage and stage.isdigit() else 4
    stage = max(1, min(4, stage))
    
    # Get crack parameters
    print("\nCrack parameters (press Enter for defaults):")
    
    crack_width = input("Crack width in nm [100]: ").strip()
    crack_width = float(crack_width) if crack_width else 100.0
    
    crack_spacing = input("Crack spacing in nm [10000]: ").strip()
    crack_spacing = float(crack_spacing) if crack_spacing else 10000.0
    
    crack_offset = input("Crack offset fraction [0.25]: ").strip()
    crack_offset = float(crack_offset) if crack_offset else 0.25
    
    parameters = {
        'crack_width': crack_width,
        'crack_spacing': crack_spacing,
        'crack_offset': crack_offset
    }
    
    # Confirm settings
    print("\n" + "=" * 60)
    print("SIMULATION CONFIGURATION")
    print(f"Name: {sim_name}")
    print(f"Stage: {stage}")
    print(f"Parameters: {parameters}")
    print("=" * 60)
    
    confirm = input("\nProceed with these settings? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Simulation cancelled.")
        return None, None, None
    
    return sim_name, stage, parameters


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run PECVD barrier coating simulation batch')
    
    # Command line arguments
    parser.add_argument('--name', help='Simulation name')
    parser.add_argument('--stage', type=int, choices=[1,2,3,4], default=4,
                       help='Execution stage (1-4)')
    parser.add_argument('--crack_width', type=float, help='Crack width in nm')
    parser.add_argument('--crack_spacing', type=float, help='Crack spacing in nm')
    parser.add_argument('--crack_offset', type=float, help='Crack offset fraction')
    parser.add_argument('--config', help='Load parameters from JSON config file')
    parser.add_argument('--interactive', action='store_true', 
                       help='Use interactive mode (default if no args)')
    
    args = parser.parse_args()
    
    # Load from config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        sim_name = config.get('simulation_name', f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        stage = config.get('stage', 4)
        parameters = config.get('parameters', {})
    
    # Use command line arguments if provided
    elif args.name:
        sim_name = args.name
        stage = args.stage
        parameters = {
            'crack_width': args.crack_width or 100.0,
            'crack_spacing': args.crack_spacing or 10000.0,
            'crack_offset': args.crack_offset or 0.25
        }
    
    # Otherwise use interactive mode
    else:
        sim_name, stage, parameters = get_user_input()
        if sim_name is None:
            return
    
    # Create and run batch
    batch = SimulationBatch(sim_name, stage, parameters)
    success = batch.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
    