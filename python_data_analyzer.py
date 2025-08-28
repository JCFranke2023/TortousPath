#!/usr/bin/env python
"""
Python data analyzer - runs in normal Python environment
Analyzes raw data extracted from ABAQUS and creates reports/plots
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

class DataAnalyzer:
    def __init__(self, output_base_dir='simulation_results'):
        """Initialize analyzer with output directory structure"""
        self.output_dir = Path(output_base_dir)
        self.raw_data_dir = self.output_dir / 'raw_data'
        self.json_dir = self.output_dir / 'json'
        self.plots_dir = self.output_dir / 'plots'
        self.logs_dir = self.output_dir / 'logs'
        
        # Ensure directories exist
        for dir_path in [self.json_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self, raw_data_file):
        """Load raw data extracted from ABAQUS"""
        try:
            with open(raw_data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading raw data: {e}")
            return None
    
    def process_flux_data(self, raw_data):
        """Process raw flux data into time series"""
        times = raw_data['times']
        processed_flux = []
        
        for frame_data in raw_data['flux_data']:
            flux_values = frame_data['flux_values']
            
            if flux_values:
                # Calculate statistics for this frame
                avg_flux = np.mean(flux_values)
                max_flux = np.max(flux_values)
                
                # Use max flux if average is very small (might be more meaningful)
                final_flux = max_flux if avg_flux < 1e-15 else avg_flux
                processed_flux.append(final_flux)
            else:
                processed_flux.append(0.0)
        
        return np.array(times), np.array(processed_flux)
    
    def calculate_breakthrough_time(self, times, flux, threshold_fraction=0.01):
        """Calculate breakthrough time when flux reaches threshold of steady-state"""
        if len(flux) < 2:
            return None, None
        
        # Check for meaningful flux
        max_flux = np.max(flux)
        if max_flux == 0.0:
            return None, 0.0
        
        # Steady state flux (average of last 20% of data)
        n_steady = max(1, len(flux) // 5)
        steady_flux = np.mean(flux[-n_steady:])
        
        # Find breakthrough time
        threshold_flux = threshold_fraction * steady_flux
        breakthrough_indices = np.where(flux >= threshold_flux)[0]
        
        if len(breakthrough_indices) == 0:
            return None, steady_flux
        
        breakthrough_time = times[breakthrough_indices[0]]
        return breakthrough_time, steady_flux
    
    def calculate_lag_time(self, times, flux):
        """Calculate lag time using steady-state extrapolation method"""
        if len(times) < 10:
            return None
        
        # Check for meaningful flux
        if np.max(flux) == 0.0:
            return None
        
        # Use last 20% of data to fit steady-state slope
        n_fit = max(10, len(times) // 5)
        t_fit = times[-n_fit:]
        f_fit = flux[-n_fit:]
        
        # Linear fit: flux = slope * (t - lag_time)
        coeffs = np.polyfit(t_fit, f_fit, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        if abs(slope) < 1e-20:
            return None
        
        lag_time = -intercept / slope
        return max(0, lag_time)
    
    def analyze_single_job(self, raw_data_file):
        """Analyze single job from raw data file"""
        print(f"Analyzing: {raw_data_file}")
        
        # Load raw data
        raw_data = self.load_raw_data(raw_data_file)
        if not raw_data:
            return None
        
        job_name = raw_data['job_name']
        
        # Process flux data
        times, flux = self.process_flux_data(raw_data)
        
        print(f"  Processed {len(times)} time points")
        print(f"  Time range: {times[0]:.1f} to {times[-1]:.1f} seconds")
        print(f"  Flux range: {flux.min():.2e} to {flux.max():.2e}")
        
        # Calculate metrics
        breakthrough_time, steady_flux = self.calculate_breakthrough_time(times, flux)
        lag_time = self.calculate_lag_time(times, flux)
        
        # Create results
        results = {
            'job_name': job_name,
            'raw_data_file': str(raw_data_file),
            'analysis_results': {
                'breakthrough_time': float(breakthrough_time) if breakthrough_time else None,
                'breakthrough_time_hours': float(breakthrough_time/3600) if breakthrough_time else None,
                'steady_state_flux': float(steady_flux) if steady_flux else None,
                'lag_time': float(lag_time) if lag_time else None,
                'lag_time_hours': float(lag_time/3600) if lag_time else None,
                'total_time': float(times[-1]),
                'total_time_hours': float(times[-1]/3600),
                'data_points': len(times),
                'max_flux': float(flux.max()),
                'min_flux': float(flux.min()),
                'has_meaningful_flux': float(flux.max()) > 1e-15
            },
            'time_data': times.tolist(),
            'flux_data': flux.tolist(),
            'extraction_info': raw_data.get('extraction_info', {})
        }
        
        print(f"  Analysis complete:")
        if breakthrough_time:
            print(f"    Breakthrough time: {breakthrough_time:.1f} s ({breakthrough_time/3600:.2f} h)")
        else:
            print(f"    Breakthrough time: Not detected")
        
        if steady_flux:
            print(f"    Steady-state flux: {steady_flux:.2e}")
        else:
            print(f"    Steady-state flux: Not calculated")
            
        if lag_time:
            print(f"    Lag time: {lag_time:.1f} s ({lag_time/3600:.2f} h)")
        else:
            print(f"    Lag time: Not calculated")
        
        return results
    
    def create_flux_plot(self, results, show_plot=False):
        """Create flux vs time plot"""
        times = np.array(results['time_data'])
        flux = np.array(results['flux_data'])
        
        # Convert to hours for plotting
        times_hours = times / 3600
        
        plt.figure(figsize=(12, 8))
        plt.plot(times_hours, flux, 'b-', linewidth=2, label='Mass Flux')
        plt.xlabel('Time (hours)')
        plt.ylabel('Mass Flux')
        plt.title(f'Water Vapor Flux vs Time - {results["job_name"]}')
        plt.grid(True, alpha=0.3)
        
        # Add breakthrough time marker
        analysis = results['analysis_results']
        if analysis['breakthrough_time']:
            bt_hours = analysis['breakthrough_time_hours']
            plt.axvline(x=bt_hours, color='r', linestyle='--', 
                       label=f'Breakthrough time ({bt_hours:.2f} h)')
        
        # Add lag time marker
        if analysis['lag_time']:
            lag_hours = analysis['lag_time_hours']
            plt.axvline(x=lag_hours, color='g', linestyle=':', 
                       label=f'Lag time ({lag_hours:.2f} h)')
        
        # Add steady-state line
        if analysis['steady_state_flux']:
            plt.axhline(y=analysis['steady_state_flux'], color='orange', 
                       linestyle='-', alpha=0.5, 
                       label=f'Steady-state flux ({analysis["steady_state_flux"]:.2e})')
        
        plt.legend()
        
        # Set appropriate y-scale
        if flux.max() > 0:
            if flux.max() / flux.min() > 100:  # Large range - use log scale
                plt.yscale('log')
                plt.ylabel('Mass Flux (log scale)')
        
        plot_file = self.plots_dir / f'{results["job_name"]}_flux_vs_time.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"  Plot saved: {plot_file}")
        return plot_file
    
    def save_results(self, results, output_file=None):
        """Save analysis results to JSON"""
        if output_file is None:
            output_file = f"{results['job_name']}_analysis.json"
        
        output_path = self.json_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Results saved: {output_path}")
        return output_path
    
    def create_summary_report(self, results_list):
        """Create summary report for multiple analyses"""
        if not results_list:
            return None
        
        summary = {
            'analysis_info': {
                'total_analyses': len(results_list),
                'successful_analyses': len([r for r in results_list if r]),
                'meaningful_flux_count': len([r for r in results_list 
                                            if r and r['analysis_results']['has_meaningful_flux']])
            },
            'individual_results': results_list
        }
        
        # Calculate statistics if multiple results
        if len(results_list) > 1:
            valid_results = [r for r in results_list if r]
            
            breakthrough_times = [r['analysis_results']['breakthrough_time'] 
                                for r in valid_results 
                                if r['analysis_results']['breakthrough_time']]
            
            steady_fluxes = [r['analysis_results']['steady_state_flux'] 
                           for r in valid_results 
                           if r['analysis_results']['steady_state_flux']]
            
            lag_times = [r['analysis_results']['lag_time'] 
                        for r in valid_results 
                        if r['analysis_results']['lag_time']]
            
            def calc_stats(values):
                if not values:
                    return None
                arr = np.array(values)
                return {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'count': len(values)
                }
            
            summary['statistics'] = {
                'breakthrough_time': calc_stats(breakthrough_times),
                'steady_state_flux': calc_stats(steady_fluxes),
                'lag_time': calc_stats(lag_times)
            }
        
        # Save summary
        summary_file = self.json_dir / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved: {summary_file}")
        return summary

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ABAQUS simulation data')
    parser.add_argument('--job', help='Job name to analyze')
    parser.add_argument('--raw-data', help='Path to raw data file')
    parser.add_argument('--all', action='store_true', help='Analyze all raw data files')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    
    args = parser.parse_args()
    
    analyzer = DataAnalyzer()
    
    if args.raw_data:
        # Analyze specific raw data file
        results = analyzer.analyze_single_job(args.raw_data)
        if results:
            analyzer.save_results(results)
            analyzer.create_flux_plot(results, args.plot)
    
    elif args.job:
        # Find raw data file for job
        raw_data_file = analyzer.raw_data_dir / f"{args.job}_raw_data.json"
        if raw_data_file.exists():
            results = analyzer.analyze_single_job(raw_data_file)
            if results:
                analyzer.save_results(results)
                analyzer.create_flux_plot(results, args.plot)
        else:
            print(f"Raw data file not found: {raw_data_file}")
    
    elif args.all:
        # Analyze all raw data files
        raw_files = list(analyzer.raw_data_dir.glob("*_raw_data.json"))
        if raw_files:
            all_results = []
            for raw_file in raw_files:
                print(f"\nProcessing: {raw_file.name}")
                result = analyzer.analyze_single_job(raw_file)
                if result:
                    analyzer.save_results(result)
                    analyzer.create_flux_plot(result, args.plot)
                    all_results.append(result)
            
            if all_results:
                analyzer.create_summary_report(all_results)
                print(f"\nProcessed {len(all_results)} jobs successfully")
        else:
            print("No raw data files found")
    
    else:
        # Check for extraction status and analyze if available
        status_file = analyzer.output_dir / 'extraction_status.json'
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            if status['status'] == 'success':
                raw_data_file = Path(status['raw_data_file'])
                if raw_data_file.exists():
                    results = analyzer.analyze_single_job(raw_data_file)
                    if results:
                        analyzer.save_results(results)
                        analyzer.create_flux_plot(results, args.plot)
                else:
                    print(f"Raw data file not found: {raw_data_file}")
            else:
                print("Extraction failed - no data to analyze")
        else:
            print("No extraction status found. Run ABAQUS extractor first.")

if __name__ == '__main__':
    main()
    