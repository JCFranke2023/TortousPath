"""
Post-processor for PECVD barrier coating permeation simulation results
Extracts breakthrough time, steady-state flux, and lag time from ABAQUS ODB files
"""

import os
import sys
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
        
    def extract_flux_history(self, odb_path):
        """Extract mass flux history from ODB file"""
        if not ABAQUS_ENV:
            print("Would extract flux history from: {}".format(odb_path))
            # Return mock data for development
            times = np.linspace(0, 86400, 100)
            flux = 1e-12 * (1 - np.exp(-times/10000))  # Exponential approach to steady state
            return times, flux
            
        if not os.path.exists(odb_path):
            raise FileNotFoundError("ODB file not found: {}".format(odb_path))
            
        # Open ODB file
        odb = odbAccess.openOdb(path=odb_path)
        
        try:
            # Get history output for mass flux
            history_region = odb.steps['Permeation'].historyRegions
            
            # Find flux history output (should be at outlet surface)
            flux_history = None
            for region_name, region in history_region.items():
                if 'MFL' in region.historyOutputs:
                    flux_history = region.historyOutputs['MFL']
                    break
            
            if flux_history is None:
                raise ValueError("No mass flux history found in ODB")
                
            # Extract time and flux data
            times = np.array([point[0] for point in flux_history.data])
            flux_values = np.array([point[1] for point in flux_history.data])
            
            return times, flux_values
            
        finally:
            odb.close()

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
            
        print("Analyzing: {}".format(job_name))
        
        try:
            # Extract flux history
            times, flux = self.extract_flux_history(odb_path)
            
            # Calculate key metrics
            breakthrough_time, steady_flux = self.calculate_breakthrough_time(times, flux)
            lag_time = self.calculate_lag_time(times, flux)
            
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
            return results
            
        except Exception as e:
            print("Error analyzing {}: {}".format(job_name, str(e)))
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

    def plot_results(self, results_list, output_dir='plots'):
        """Create plots of results (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available - skipping plots")
            return
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
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
                plt.savefig(os.path.join(output_dir, 'sensitivity_{}.png'.format(param)))
                plt.close()

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
    
    if args.job:
        # Analyze single job
        result = post_processor.analyze_single_result(args.job, args.odb)
        if result:
            summary = post_processor.create_summary_report([result], args.output)
            post_processor.plot_results([result])
            
    elif args.sweep:
        # Analyze parameter sweep
        with open(args.sweep, 'r') as f:
            job_results = json.load(f)
        
        results = post_processor.analyze_parameter_sweep(job_results)
        summary = post_processor.create_summary_report(results, args.output)
        post_processor.plot_results(results)
        
    else:
        print("Please specify --job or --sweep option")
        