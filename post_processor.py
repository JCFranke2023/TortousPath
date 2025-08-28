"""
Post-processor for PECVD barrier coating permeation simulation results
Extracts breakthrough time, steady-state flux, and lag time from ABAQUS ODB files
"""

import os
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

try:
    # ABAQUS imports
    from abaqus import *
    from abaqusConstants import *
    import odbAccess
    import visualization
    ABAQUS_ENV = True
except ImportError:
    ABAQUS_ENV = False
    print("Running in development mode - ABAQUS imports not available")

class PostProcessor:
    def __init__(self):
        self.results_cache = {}
    
        # Create results output directory in root
        self.output_dir = Path('.') / 'simulation_results'
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'json').mkdir(exist_ok=True)

        # Setup logging to file
        self.log_file = self.output_dir / 'postprocessor.log'

    def log(self, message):
        """Write log message to file and try to print"""
        with open(self.log_file, 'a') as f:
            f.write("{}\n".format(message))
        try:
            print(message)
        except:
            pass  # Print might not work in ABAQUS noGUI

    def extract_flux_history(self, odb_path):
        """Extract mass flux history from ODB file"""
        if not ABAQUS_ENV:
            self.log("Running in development mode - using mock data")
            times = np.linspace(0, 86400, 100)
            flux = 1e-12 * (1 - np.exp(-times/10000))
            return times, flux
            
        self.log("Opening ODB file: {}".format(odb_path))
        if not os.path.exists(odb_path):
            raise FileNotFoundError("ODB file not found: {}".format(odb_path))
            
        # Open ODB file
        odb = odbAccess.openOdb(path=odb_path)
        
        try:
            self.log("ODB opened successfully")
            
            # Check available steps
            self.log("Available steps: {}".format(list(odb.steps.keys())))
            
            if 'Permeation' not in odb.steps:
                raise ValueError("Permeation step not found in ODB")
                
            step = odb.steps['Permeation']
            self.log("Found Permeation step")
            
            # Get field outputs to see what's available
            frames = step.frames
            self.log("Number of frames: {}".format(len(frames)))
            
            if len(frames) > 0:
                last_frame = frames[-1]
                self.log("Available field outputs in last frame: {}".format(list(last_frame.fieldOutputs.keys())))
            
            # Get history output for mass flux
            history_regions = step.historyRegions
            self.log("Available history regions: {}".format(list(history_regions.keys())))
            
            # Find flux history output
            flux_history = None
            for region_name, region in history_regions.items():
                outputs = list(region.historyOutputs.keys())
                self.log("Region '{}' outputs: {}".format(region_name, outputs))
                
                # Look for mass flux outputs
                for output_name in outputs:
                    if any(keyword in output_name.upper() for keyword in ['MFL', 'FLUX', 'FL']):
                        flux_history = region.historyOutputs[output_name]
                        self.log("Found flux history '{}' in region: {}".format(output_name, region_name))
                        break
                if flux_history:
                    break
            
            if flux_history is None:
                self.log("No flux history found. Available outputs:")
                for region_name, region in history_regions.items():
                    for output_name in region.historyOutputs.keys():
                        self.log("  Region '{}': '{}'".format(region_name, output_name))
                
                # Try using concentration field data as fallback
                self.log("Attempting to use concentration field data...")
                return self._extract_from_field_data(odb, step)
                
            # Extract time and flux data
            self.log("Extracting data from flux history...")
            data_points = flux_history.data
            self.log("Number of data points: {}".format(len(data_points)))
            
            times = np.array([point[0] for point in data_points])
            flux_values = np.array([point[1] for point in data_points])
            
            self.log("Time range: {:.1f} to {:.1f} seconds".format(times[0], times[-1]))
            self.log("Flux range: {:.2e} to {:.2e}".format(flux_values.min(), flux_values.max()))
            
            return times, flux_values
            
        finally:
            odb.close()
            self.log("ODB file closed")

    def calculate_breakthrough_time(self, times, flux, threshold_fraction=0.01):
        """Calculate breakthrough time (when flux reaches threshold of steady-state)"""
        # Estimate steady-state flux (average of last 10% of data)
        n_steady = max(1, len(flux) // 10)
        steady_flux = np.mean(flux[-n_steady:])
        
        # Find breakthrough time
        threshold_flux = threshold_fraction * steady_flux
        breakthrough_idx = np.where(flux >= threshold_flux)[0]
        
        if len(breakthrough_idx) == 0:
            return None, steady_flux
            
        breakthrough_time = times[breakthrough_idx[0]]
        return breakthrough_time, steady_flux

    def calculate_lag_time(self, times, flux):
        """Calculate lag time using steady-state flux extrapolation method"""
        if len(times) < 10:
            return None
            
        # Use last 20% of data to fit steady-state slope
        n_fit = max(10, len(times) // 5)
        t_fit = times[-n_fit:]
        f_fit = flux[-n_fit:]
        
        # Linear fit: flux = slope * (t - lag_time)
        # Extrapolate back to find x-intercept
        coeffs = np.polyfit(t_fit, f_fit, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        if abs(slope) < 1e-20:
            return None
            
        lag_time = -intercept / slope
        return max(0, lag_time)  # Lag time should be positive

    def analyze_single_result(self, job_name, odb_path=None):
        """Analyze single simulation result"""
        if odb_path is None:
            odb_path = job_name + '.odb'
        
        self.log("=== STARTING ANALYSIS ===")
        self.log("Analyzing: {}".format(job_name))
        self.log("ODB path: {}".format(odb_path))
        self.log("ODB exists: {}".format(os.path.exists(odb_path)))
        self.log("Current working directory: {}".format(os.getcwd()))
        self.log("ABAQUS environment: {}".format(ABAQUS_ENV))
        
        if not os.path.exists(odb_path):
            self.log("ERROR: ODB file not found at: {}".format(odb_path))
            self.log("Files in current directory:")
            for f in os.listdir('.'):
                if job_name in f or f.endswith('.odb'):
                    self.log("  {}".format(f))
            return None
        
        try:
            # Extract flux history
            self.log("Extracting flux history...")
            times, flux = self.extract_flux_history(odb_path)
            self.log("Extracted {} time points".format(len(times)))
            
            # Calculate key metrics
            self.log("Calculating metrics...")
            breakthrough_time, steady_flux = self.calculate_breakthrough_time(times, flux)
            lag_time = self.calculate_lag_time(times, flux)
            
            self.log("Breakthrough time: {}".format(breakthrough_time))
            self.log("Steady flux: {}".format(steady_flux))
            self.log("Lag time: {}".format(lag_time))
            
            # Store results
            results = {
                'job_name': job_name,
                'odb_path': odb_path,
                'breakthrough_time': breakthrough_time,
                'steady_state_flux': steady_flux,
                'lag_time': lag_time,
                'time_data': times.tolist() if hasattr(times, 'tolist') else times,
                'flux_data': flux.tolist() if hasattr(flux, 'tolist') else flux
            }
            
            self.results_cache[job_name] = results
            self.log("Analysis completed successfully")
            return results
            
        except Exception as e:
            self.log("Error analyzing {}: {}".format(job_name, str(e)))
            import traceback
            self.log("Full traceback:")
            self.log(traceback.format_exc())
            return None
        
    def analyze_parameter_sweep(self, job_results):
        """Analyze multiple simulation results from parameter sweep"""
        all_results = []
        
        for job_info in job_results:
            job_name = job_info['job_name']
            parameters = job_info['parameters']
            
            # Analyze individual result
            result = self.analyze_single_result(job_name)
            
            if result is not None:
                result['parameters'] = parameters
                all_results.append(result)
        
        return all_results

    def extract_concentration_field(self, odb_path, time_point=-1):
        """Extract concentration field at specific time"""
        if not ABAQUS_ENV:
            print("Would extract concentration field from: {}".format(odb_path))
            return None, None, None  # x, y, concentration
            
        if not os.path.exists(odb_path):
            raise FileNotFoundError("ODB file not found: {}".format(odb_path))
            
        odb = odbAccess.openOdb(path=odb_path)
        
        try:
            # Get field output
            step = odb.steps['Permeation']
            frames = step.frames
            
            if time_point < 0:
                frame = frames[-1]  # Last frame
            else:
                # Find frame closest to specified time
                times = [f.frameValue for f in frames]
                idx = np.argmin(np.abs(np.array(times) - time_point))
                frame = frames[idx]
            
            # Get concentration field
            conc_field = frame.fieldOutputs['CONC']
            
            # Extract coordinates and values
            coordinates = []
            concentrations = []
            
            for value in conc_field.values:
                coordinates.append(value.nodeLabel)  # Node coordinates would need to be extracted separately
                concentrations.append(value.data)
                
            return frame.frameValue, coordinates, concentrations
            
        finally:
            odb.close()

    def create_summary_report(self, results_list, output_file='results_summary.json'):
        """Create summary report of all results"""
        # Ensure output file is in the results directory
        if not os.path.isabs(output_file):
            output_file = self.output_dir / 'json' / output_file
        
        summary = {
            'analysis_info': {
                'total_simulations': len(results_list),
                'successful_analyses': len([r for r in results_list if r is not None]),
                'analysis_date': None  # Could add timestamp
            },
            'results': results_list
        }
        
        # Calculate statistics if multiple results
        if len(results_list) > 1:
            valid_results = [r for r in results_list if r is not None]
            
            if valid_results:
                breakthrough_times = [r['breakthrough_time'] for r in valid_results if r['breakthrough_time'] is not None]
                steady_fluxes = [r['steady_state_flux'] for r in valid_results if r['steady_state_flux'] is not None]
                lag_times = [r['lag_time'] for r in valid_results if r['lag_time'] is not None]
                
                summary['statistics'] = {
                    'breakthrough_time': {
                        'mean': np.mean(breakthrough_times) if breakthrough_times else None,
                        'std': np.std(breakthrough_times) if breakthrough_times else None,
                        'min': np.min(breakthrough_times) if breakthrough_times else None,
                        'max': np.max(breakthrough_times) if breakthrough_times else None
                    },
                    'steady_state_flux': {
                        'mean': np.mean(steady_fluxes) if steady_fluxes else None,
                        'std': np.std(steady_fluxes) if steady_fluxes else None,
                        'min': np.min(steady_fluxes) if steady_fluxes else None,
                        'max': np.max(steady_fluxes) if steady_fluxes else None
                    },
                    'lag_time': {
                        'mean': np.mean(lag_times) if lag_times else None,
                        'std': np.std(lag_times) if lag_times else None,
                        'min': np.min(lag_times) if lag_times else None,
                        'max': np.max(lag_times) if lag_times else None
                    }
                }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("Summary report saved to: {}".format(output_file))
        return summary

    def plot_results(self, results_list, output_dir=None):
        """Create plots of results (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available - skipping plots")
            return
            
        # Use default plots directory if not specified
        if output_dir is None:
            output_dir = self.output_dir / 'plots'
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        valid_results = [r for r in results_list if r is not None]
        
        # Plot flux vs time for each simulation
        plt.figure(figsize=(10, 6))
        for result in valid_results:
            times = np.array(result['time_data'])
            flux = np.array(result['flux_data'])
            label = result['job_name']
            plt.plot(times/3600, flux, label=label)  # Convert to hours
            
        plt.xlabel('Time (hours)')
        plt.ylabel('Mass Flux')
        plt.title('Water Vapor Flux vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'flux_vs_time.png'))
        plt.close()
        
        print("Plot saved to: {}".format(os.path.join(output_dir, 'flux_vs_time.png')))
        
        # Parameter sensitivity plots (if parameter sweep data available)
        if len(valid_results) > 1 and 'parameters' in valid_results[0]:
            self._plot_parameter_sensitivity(valid_results, output_dir)

    def _plot_parameter_sensitivity(self, results, output_dir):
        """Create parameter sensitivity plots"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
            
        # Extract parameter values and metrics
        data = defaultdict(list)
        
        for result in results:
            params = result['parameters']
            for param_name, param_value in params.items():
                data[param_name].append(param_value)
                data[param_name + '_breakthrough'].append(result['breakthrough_time'])
                data[param_name + '_flux'].append(result['steady_state_flux'])
        
        # Create plots for each parameter
        for param in ['crack_width', 'crack_spacing', 'crack_offset']:
            if param in data:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                
                x_vals = data[param]
                y1_vals = data[param + '_breakthrough']
                y2_vals = data[param + '_flux']
                
                # Breakthrough time vs parameter
                ax1.scatter(x_vals, y1_vals)
                ax1.set_xlabel(param.replace('_', ' ').title())
                ax1.set_ylabel('Breakthrough Time (s)')
                ax1.set_title('Breakthrough Time vs {}'.format(param.replace('_', ' ').title()))
                ax1.grid(True)
                
                # Steady flux vs parameter
                ax2.scatter(x_vals, y2_vals)
                ax2.set_xlabel(param.replace('_', ' ').title())
                ax2.set_ylabel('Steady State Flux')
                ax2.set_title('Steady State Flux vs {}'.format(param.replace('_', ' ').title()))
                ax2.grid(True)
                
                plt.tight_layout()
                plot_path = os.path.join(output_dir, 'sensitivity_{}.png'.format(param))
                plt.savefig(plot_path)
                plt.close()
                print("Sensitivity plot saved to: {}".format(plot_path))

    def _extract_from_field_data(self, odb, step):
        """Extract concentration field data as fallback when flux history not available"""
        self.log("Using concentration field data as fallback")
        
        frames = step.frames
        times = []
        concentrations = []
        
        for frame in frames:
            if 'CONC' in frame.fieldOutputs:
                times.append(frame.frameValue)
                conc_field = frame.fieldOutputs['CONC']
                
                # Get concentration at outlet (bottom surface)
                # This is a simplified approach - you might need to modify based on your geometry
                outlet_conc = 0.0
                node_count = 0
                for value in conc_field.values:
                    # Approximate outlet as nodes with minimum y-coordinate
                    if hasattr(value, 'nodeLabel') and hasattr(value, 'data'):
                        outlet_conc += value.data
                        node_count += 1
                
                if node_count > 0:
                    concentrations.append(outlet_conc / node_count)
                else:
                    concentrations.append(0.0)
        
        self.log("Extracted {} concentration data points".format(len(times)))
        
        # Convert concentration gradient to flux (simplified)
        times = np.array(times)
        flux = np.gradient(concentrations, times)  # Approximate flux from concentration gradient
        
        return times, flux

# Create global instance
post_processor = PostProcessor()

# Command line interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Post-process PECVD barrier simulation results')
    parser.add_argument('--job', help='Single job name to analyze')
    parser.add_argument('--sweep', help='JSON file with sweep results to analyze')
    parser.add_argument('--odb', help='ODB file path (for single job)')
    parser.add_argument('--output', default='results_summary.json', help='Output summary file')
    
    args = parser.parse_args()
    
    # Clear previous log
    post_processor.log_file.unlink() if post_processor.log_file.exists() else None
    
    post_processor.log("=== POST PROCESSOR STARTED ===")
    post_processor.log("Current working directory: {}".format(os.getcwd()))
    post_processor.log("ABAQUS environment: {}".format(ABAQUS_ENV))
    post_processor.log("Arguments: {}".format(vars(args)))
    
    if args.job:
        # Convert relative path to absolute path if needed
        odb_path = args.odb
        if odb_path and not os.path.isabs(odb_path):
            odb_path = os.path.abspath(odb_path)
            post_processor.log("Converted ODB path to absolute: {}".format(odb_path))
        
        # Analyze single job
        result = post_processor.analyze_single_result(args.job, odb_path)
        if result:
            post_processor.log("Creating summary report...")
            summary = post_processor.create_summary_report([result], args.output)
            post_processor.log("Creating plots...")
            post_processor.plot_results([result])
            post_processor.log("=== PROCESSING COMPLETE ===")
        else:
            post_processor.log("Analysis failed - no results to save")
            post_processor.log("=== PROCESSING FAILED ===")
            
    elif args.sweep:
        # Analyze parameter sweep
        with open(args.sweep, 'r') as f:
            job_results = json.load(f)
        
        results = post_processor.analyze_parameter_sweep(job_results)
        summary = post_processor.create_summary_report(results, args.output)
        post_processor.plot_results(results)
        
    else:
        post_processor.log("Please specify --job or --sweep option")
        