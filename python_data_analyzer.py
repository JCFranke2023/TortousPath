#!/usr/bin/env python
"""
Python data analyzer - runs in normal Python environment
Analyzes raw data extracted from ABAQUS and creates reports/plots
Now includes data cleanup and validation functionality
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
    
    def clean_and_validate_data(self, raw_data):
        """Clean and validate raw flux data"""
        times = np.array(raw_data['times'])
        cleaned_data = {
            'original_frames': len(raw_data['flux_data']),
            'valid_frames': 0,
            'cleaned_times': [],
            'cleaned_flux': [],
            'issues_found': [],
            'statistics': {}
        }
        
        for i, frame_data in enumerate(raw_data['flux_data']):
            time = frame_data['time']
            flux_values = frame_data['flux_values']
            valid_count = frame_data['valid_count']
            
            # Skip frames with no valid data
            if valid_count == 0 or not flux_values:
                cleaned_data['issues_found'].append(f"Frame {i}: No valid flux data")
                continue
            
            # Clean flux values - remove invalid entries
            valid_flux = []
            for val in flux_values:
                if isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                    valid_flux.append(abs(val))  # Take absolute value
            
            if not valid_flux:
                cleaned_data['issues_found'].append(f"Frame {i}: All flux values invalid")
                continue
            
            # Calculate representative flux for this frame
            flux_array = np.array(valid_flux)
            
            # Filter out extreme outliers (more than 3 standard deviations from mean)
            if len(flux_array) > 1:
                mean_flux = np.mean(flux_array)
                std_flux = np.std(flux_array)
                if std_flux > 0:
                    outlier_mask = np.abs(flux_array - mean_flux) < 3 * std_flux
                    flux_array = flux_array[outlier_mask]
            
            if len(flux_array) == 0:
                cleaned_data['issues_found'].append(f"Frame {i}: All values were outliers")
                continue
            
            # Choose representative value
            # Use maximum flux if the range is very large (might indicate breakthrough)
            # Use mean flux for steady conditions
            flux_range = np.max(flux_array) - np.min(flux_array)
            mean_val = np.mean(flux_array)
            max_val = np.max(flux_array)
            
            if mean_val > 0 and flux_range / mean_val > 10:
                # Large relative range - use max (potential breakthrough signature)
                representative_flux = max_val
            else:
                # Use mean for stable conditions
                representative_flux = mean_val
            
            cleaned_data['cleaned_times'].append(time)
            cleaned_data['cleaned_flux'].append(representative_flux)
            cleaned_data['valid_frames'] += 1
        
        # Convert to numpy arrays
        cleaned_data['cleaned_times'] = np.array(cleaned_data['cleaned_times'])
        cleaned_data['cleaned_flux'] = np.array(cleaned_data['cleaned_flux'])
        
        # Calculate statistics
        if len(cleaned_data['cleaned_flux']) > 0:
            flux_array = cleaned_data['cleaned_flux']
            cleaned_data['statistics'] = {
                'total_frames_processed': cleaned_data['valid_frames'],
                'data_quality_ratio': cleaned_data['valid_frames'] / cleaned_data['original_frames'],
                'flux_min': float(np.min(flux_array)),
                'flux_max': float(np.max(flux_array)),
                'flux_mean': float(np.mean(flux_array)),
                'flux_std': float(np.std(flux_array)),
                'zero_flux_frames': int(np.sum(flux_array == 0.0)),
                'nonzero_flux_frames': int(np.sum(flux_array > 0.0)),
                'time_span_hours': float((cleaned_data['cleaned_times'][-1] - cleaned_data['cleaned_times'][0]) / 3600)
            }
        
        return cleaned_data
    
    def process_flux_data(self, raw_data):
        """Process and clean raw flux data into time series"""
        # First clean the data
        cleaned_data = self.clean_and_validate_data(raw_data)
        
        # Log data quality issues
        if cleaned_data['issues_found']:
            print(f"  Data quality issues found:")
            for issue in cleaned_data['issues_found'][:5]:  # Show first 5
                print(f"    {issue}")
            if len(cleaned_data['issues_found']) > 5:
                print(f"    ... and {len(cleaned_data['issues_found']) - 5} more issues")
        
        # Report statistics
        stats = cleaned_data['statistics']
        if stats:
            print(f"  Data quality: {stats['data_quality_ratio']:.1%} frames valid")
            print(f"  Flux range: {stats['flux_min']:.2e} to {stats['flux_max']:.2e}")
            if stats['nonzero_flux_frames'] > 0:
                print(f"  Non-zero flux detected in {stats['nonzero_flux_frames']} frames")
            else:
                print(f"  WARNING: All flux values are zero - check boundary conditions")
        
        return cleaned_data['cleaned_times'], cleaned_data['cleaned_flux'], cleaned_data
    
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
    
    def diagnose_zero_flux(self, raw_data, cleaned_data):
        """Diagnose why flux might be zero and suggest fixes"""
        diagnosis = {
            'likely_causes': [],
            'suggestions': []
        }
        
        # Check extraction statistics
        extraction_info = raw_data.get('extraction_info', {})
        if extraction_info.get('failed_frames', 0) > 0:
            diagnosis['likely_causes'].append("Some frames failed during extraction")
            diagnosis['suggestions'].append("Check ABAQUS ODB file integrity")
        
        # Check if MFL field was found
        total_frames = extraction_info.get('successful_frames', 0)
        if total_frames == 0:
            diagnosis['likely_causes'].append("No MFL field data extracted")
            diagnosis['suggestions'].append("Verify mass diffusion analysis was run")
            diagnosis['suggestions'].append("Check if boundary conditions were applied")
        
        # Check data statistics
        stats = cleaned_data.get('statistics', {})
        if stats.get('zero_flux_frames', 0) == stats.get('total_frames_processed', 0):
            diagnosis['likely_causes'].append("All flux values are exactly zero")
            diagnosis['suggestions'].append("Check boundary conditions (inlet/outlet concentrations)")
            diagnosis['suggestions'].append("Verify material diffusivities are not too small")
            diagnosis['suggestions'].append("Check if simulation time is sufficient for diffusion")
            diagnosis['suggestions'].append("Verify mesh quality and element types (DC2D4)")
        
        return diagnosis
    
    def analyze_single_job(self, raw_data_file):
        """Analyze single job from raw data file"""
        print(f"Analyzing: {raw_data_file}")
        
        # Load raw data
        raw_data = self.load_raw_data(raw_data_file)
        if not raw_data:
            return None
        
        job_name = raw_data['job_name']
        
        # Process and clean flux data
        times, flux, cleaned_data = self.process_flux_data(raw_data)
        
        print(f"  Processed {len(times)} time points")
        if len(times) > 0:
            print(f"  Time range: {times[0]:.1f} to {times[-1]:.1f} seconds")
            print(f"  Flux range: {flux.min():.2e} to {flux.max():.2e}")
        
        # Diagnose issues if flux is all zero
        diagnosis = None
        if len(flux) == 0 or np.max(flux) == 0.0:
            print(f"  WARNING: Zero flux detected - running diagnosis...")
            diagnosis = self.diagnose_zero_flux(raw_data, cleaned_data)
            
            print(f"  Likely causes:")
            for cause in diagnosis['likely_causes']:
                print(f"    - {cause}")
            print(f"  Suggestions:")
            for suggestion in diagnosis['suggestions']:
                print(f"    - {suggestion}")
        
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
                'total_time': float(times[-1]) if len(times) > 0 else None,
                'total_time_hours': float(times[-1]/3600) if len(times) > 0 else None,
                'data_points': len(times),
                'max_flux': float(flux.max()) if len(flux) > 0 else 0.0,
                'min_flux': float(flux.min()) if len(flux) > 0 else 0.0,
                'has_meaningful_flux': float(flux.max()) > 1e-15 if len(flux) > 0 else False
            },
            'time_data': times.tolist() if len(times) > 0 else [],
            'flux_data': flux.tolist() if len(flux) > 0 else [],
            'data_cleaning': {
                'original_frames': cleaned_data.get('original_frames', 0),
                'valid_frames': cleaned_data.get('valid_frames', 0),
                'issues_found': cleaned_data.get('issues_found', []),
                'statistics': cleaned_data.get('statistics', {})
            },
            'diagnosis': diagnosis,
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
        """Create flux vs time plot with diagnostics"""
        times = np.array(results['time_data'])
        flux = np.array(results['flux_data'])
        
        if len(times) == 0:
            print("  No data to plot")
            return None
        
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
        
        # Add diagnostic info if flux is zero
        if not analysis['has_meaningful_flux']:
            plt.text(0.5, 0.8, 'WARNING: Zero flux detected\nCheck boundary conditions', 
                    transform=plt.gca().transAxes, fontsize=12, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
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