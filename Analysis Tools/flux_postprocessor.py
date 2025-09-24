"""
Pure Python post-processor for PECVD barrier coating flux data
Processes NNC gradient data to calculate mass flux using Fick's law within PET
"""

import os
import sys
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit

# Import material properties for consistency
try:
    from material_properties import materials
except ImportError:
    print("WARNING: Could not import material_properties. Using default values.")
    
    class DefaultMaterials:
        def get_diffusivity(self, material):
            defaults = {
                'PET': 1.0e6,           # nm²/s
                'interlayer': 5.0e7,    
                'barrier': 1.0e-2,      
                'air_crack': 2.4e13     
            }
            return defaults.get(material, 1.0e6)
    
    materials = DefaultMaterials()

class FluxPostProcessor:
    def __init__(self, output_dir=None, verbose=True):
        """
        Initialize flux post-processor
        
        Args:
            output_dir: Base directory for analysis outputs (e.g., simulations/sim_name/analysis)
            verbose: Print progress messages
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path('flux_analysis')
        
        self.verbose = verbose
        
        # Create subdirectories
        self.flux_dir = self.output_dir / 'flux_data'
        self.metrics_dir = self.output_dir / 'metrics'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.flux_dir, self.metrics_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Physical constants
        self.MW_water = 18.0  # g/mol - molecular weight of water
        
        # Get PET diffusivity from material properties
        self.D_PET = materials.get_diffusivity('PET')  # nm²/s
        
        if self.verbose:
            print(f"Initialized FluxPostProcessor")
            print(f"  Output directory: {self.output_dir}")
            print(f"  D_PET = {self.D_PET:.2e} nm²/s")

    def process_nnc_data(self, csv_file, job_name=None, save_plots=True):
        """
        Process NNC gradient data to calculate mass flux
        
        Args:
            csv_file: Path to CSV file with NNC gradient data
            job_name: Job name (extracted from filename if not provided)
            save_plots: Whether to generate and save plots
        
        Returns:
            Dict with complete analysis results
        """
        if self.verbose:
            print(f"\n=== PROCESSING NNC DATA ===")
            print(f"Input file: {csv_file}")
        
        # Extract job name from filename if not provided
        if not job_name:
            job_name = Path(csv_file).stem.replace('_flux', '')
        
        # Parse job parameters from filename
        params = self._parse_job_parameters(job_name)
        
        # Load NNC gradient data
        data = self._load_nnc_data(csv_file)
        if not data:
            return {
                'success': False,
                'job_name': job_name,
                'error': 'Failed to load NNC data'
            }
        
        # Calculate mass flux using Fick's law
        flux_data = self._calculate_mass_flux(data, params)
        
        # Calculate permeation metrics
        metrics = self._calculate_permeation_metrics(flux_data, params)
        
        # Prepare complete results
        results = {
            'success': True,
            'job_name': job_name,
            'input_file': str(csv_file),
            'parameters': params,
            'metrics': metrics,
            'flux_data': flux_data,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        output_files = self._save_results(results)
        results['output_files'] = output_files
        
        # Create plots if requested
        if save_plots:
            plot_file = self._create_plots(results)
            results['output_files']['plot'] = str(plot_file)
        
        if self.verbose:
            print(f"\n✓ Analysis completed for {job_name}")
            print(f"  Output files saved to: {self.output_dir}")
        
        return results

    def _parse_job_parameters(self, job_name):
        """Parse crack parameters from job name"""
        params = {
            'crack_width': 100.0,      # nm - default
            'crack_spacing': 10000.0,  # nm - default  
            'crack_offset': 0.25       # fraction - default
        }
        
        try:
            if 'Job_c' in job_name:
                parts = job_name.split('_')
                if len(parts) >= 4:
                    params['crack_width'] = float(parts[1][1:])       # c100 -> 100.0
                    params['crack_spacing'] = float(parts[2][1:])     # s10000 -> 10000.0
                    params['crack_offset'] = float(parts[3][1:]) / 100 # o25 -> 0.25
                    
                    print(f"Parsed parameters: width={params['crack_width']} nm, "
                          f"spacing={params['crack_spacing']} nm, offset={params['crack_offset']}")
        except Exception as e:
            print(f"Could not parse parameters from {job_name}: {e}")
            print("Using default parameters")
        
        return params
    
    def _load_nnc_data(self, csv_file):
        """Load NNC gradient data from CSV"""
        data = {
            'time_s': [],
            'nnc_bottom': [],
            'nnc_second': [], 
            'gradient_per_nm': []
        }
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data['time_s'].append(float(row['time_s']))
                    data['nnc_bottom'].append(float(row['nnc_bottom']))
                    data['nnc_second'].append(float(row['nnc_second']))
                    data['gradient_per_nm'].append(float(row['gradient_per_nm']))
            
            print(f"Loaded {len(data['time_s'])} data points")
            print(f"Time range: {data['time_s'][0]:.1f} to {data['time_s'][-1]:.1f} seconds")
            print(f"Gradient range: {min(data['gradient_per_nm']):.2e} to {max(data['gradient_per_nm']):.2e} mol/nm⁴")
            
            return data
            
        except Exception as e:
            print(f"ERROR loading data: {e}")
            return None
    
    def _calculate_mass_flux(self, data, params):
        """Calculate mass flux using Fick's law within PET layer"""
        print(f"\n=== CALCULATING MASS FLUX ===")
        print(f"Using Fick's law: J = D_PET × |dC/dy|")
        print(f"D_PET = {self.D_PET:.2e} nm²/s")
        
        flux_data = {
            'time_s': [],
            'flux_mol_nm2_s': [],
            'flux_kg_nm2_s': [],
            'flux_g_m2_day': [],
            'cumulative_g_m2': []
        }
        
        cumulative = 0.0
        
        for i in range(len(data['time_s'])):
            time = data['time_s'][i]
            gradient = abs(data['gradient_per_nm'][i])  # mol/nm⁴
            
            # Apply Fick's law within PET
            flux_mol_nm2_s = self.D_PET * gradient  # mol/(nm²·s)
            
            # Convert to mass flux
            flux_kg_nm2_s = flux_mol_nm2_s * self.MW_water * 1e-3  # kg/(nm²·s)
            
            # Convert to standard units: g/(m²·day)
            # nm² to m²: × 1e18, s to day: × 86400, kg to g: × 1000
            flux_g_m2_day = flux_kg_nm2_s * 1e18 * 86400 * 1000
            
            # Calculate cumulative mass (for lag time calculation)
            if i > 0:
                dt = time - data['time_s'][i-1]  # seconds
                cumulative += flux_g_m2_day * dt / 86400  # g/m²
            
            flux_data['time_s'].append(time)
            flux_data['flux_mol_nm2_s'].append(flux_mol_nm2_s)
            flux_data['flux_kg_nm2_s'].append(flux_kg_nm2_s)
            flux_data['flux_g_m2_day'].append(flux_g_m2_day)
            flux_data['cumulative_g_m2'].append(cumulative)
        
        # Print summary
        if flux_data['flux_g_m2_day']:
            max_flux = max(flux_data['flux_g_m2_day'])
            final_flux = flux_data['flux_g_m2_day'][-1]
            print(f"Max flux: {max_flux:.2e} g/(m²·day)")
            print(f"Final flux: {final_flux:.2e} g/(m²·day)")
            print(f"Total cumulative: {cumulative:.2e} g/m²")
        
        return flux_data
    
    def _calculate_permeation_metrics(self, flux_data, params):
        """Calculate key permeation metrics"""
        print(f"\n=== CALCULATING PERMEATION METRICS ===")
        
        if not flux_data['flux_g_m2_day']:
            return {'error': 'No flux data available'}
        
        metrics = {}
        times = np.array(flux_data['time_s'])
        fluxes = np.array(flux_data['flux_g_m2_day'])
        cumulative = np.array(flux_data['cumulative_g_m2'])
        
        # Steady-state flux (average of last 10% of data)
        n_points = len(fluxes)
        steady_points = max(1, n_points // 10)
        metrics['steady_state_flux'] = float(np.mean(fluxes[-steady_points:]))
        metrics['steady_state_flux_units'] = 'g/(m²·day)'
        
        # Breakthrough time (10% of steady-state)
        threshold = 0.1 * metrics['steady_state_flux']
        breakthrough_idx = np.where(fluxes > threshold)[0]
        if len(breakthrough_idx) > 0:
            metrics['breakthrough_time_s'] = float(times[breakthrough_idx[0]])
            metrics['breakthrough_time_h'] = metrics['breakthrough_time_s'] / 3600
        else:
            metrics['breakthrough_time_s'] = None
            metrics['breakthrough_time_h'] = None
        
        # Lag time calculation (from cumulative mass vs time)
        if len(times) > 5 and metrics['steady_state_flux'] > 0:
            # Find linear region (20% to 80% of steady state)
            ss_flux = metrics['steady_state_flux']
            mask = (fluxes > 0.2 * ss_flux) & (fluxes < 0.8 * ss_flux)
            
            if np.sum(mask) > 2:
                try:
                    # Linear fit to cumulative mass
                    x_fit = times[mask]
                    y_fit = cumulative[mask]
                    coeffs = np.polyfit(x_fit, y_fit, 1)
                    
                    if coeffs[0] > 0:  # Positive slope
                        metrics['lag_time_s'] = float(-coeffs[1] / coeffs[0])
                        metrics['lag_time_h'] = metrics['lag_time_s'] / 3600
                    else:
                        metrics['lag_time_s'] = None
                        metrics['lag_time_h'] = None
                except:
                    metrics['lag_time_s'] = None
                    metrics['lag_time_h'] = None
            else:
                metrics['lag_time_s'] = None
                metrics['lag_time_h'] = None
        else:
            metrics['lag_time_s'] = None
            metrics['lag_time_h'] = None
        
        # Time to 90% steady-state
        if metrics['steady_state_flux'] > 0:
            threshold_90 = 0.9 * metrics['steady_state_flux']
            ss_90_idx = np.where(fluxes > threshold_90)[0]
            if len(ss_90_idx) > 0:
                metrics['time_to_90_percent_s'] = float(times[ss_90_idx[0]])
                metrics['time_to_90_percent_h'] = metrics['time_to_90_percent_s'] / 3600
            else:
                metrics['time_to_90_percent_s'] = None
                metrics['time_to_90_percent_h'] = None
        
        # Total simulation time
        metrics['total_time_s'] = float(times[-1])
        metrics['total_time_h'] = metrics['total_time_s'] / 3600
        
        # Crack parameters for reference
        metrics['crack_width_nm'] = params['crack_width']
        metrics['crack_spacing_nm'] = params['crack_spacing']
        metrics['crack_offset'] = params['crack_offset']
        metrics['crack_fraction'] = params['crack_width'] / params['crack_spacing']
        
        # Print summary
        print(f"Steady-state flux: {metrics['steady_state_flux']:.2e} g/(m²·day)")
        if metrics['breakthrough_time_h']:
            print(f"Breakthrough time: {metrics['breakthrough_time_h']:.2f} hours")
        if metrics['lag_time_h']:
            print(f"Lag time: {metrics['lag_time_h']:.2f} hours")
        if metrics['time_to_90_percent_h']:
            print(f"Time to 90% steady-state: {metrics['time_to_90_percent_h']:.2f} hours")
        
        return metrics
    
    def _save_results(self, results):
        """
        Save processed results to appropriate directories
        
        Returns:
            Dict of output file paths
        """
        job_name = results['job_name']
        output_files = {}
        
        try:
            # Save processed flux data as CSV
            csv_file = self.flux_dir / f"{job_name}_processed_flux.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_s', 'time_h', 'flux_g_m2_day', 'cumulative_g_m2'])
                
                flux_data = results['flux_data']
                for i in range(len(flux_data['time_s'])):
                    writer.writerow([
                        flux_data['time_s'][i],
                        flux_data['time_s'][i] / 3600,
                        flux_data['flux_g_m2_day'][i],
                        flux_data['cumulative_g_m2'][i]
                    ])
            
            output_files['flux_csv'] = str(csv_file)
            if self.verbose:
                print(f"  Saved flux data: {csv_file.name}")
            
            # Save metrics as JSON
            json_file = self.metrics_dir / f"{job_name}_metrics.json"
            metrics_output = {
                'job_name': results['job_name'],
                'parameters': results['parameters'],
                'metrics': results['metrics'],
                'analysis_timestamp': results['analysis_timestamp']
            }
            
            with open(json_file, 'w') as f:
                json.dump(metrics_output, f, indent=2)
            
            output_files['metrics_json'] = str(json_file)
            if self.verbose:
                print(f"  Saved metrics: {json_file.name}")
            
            # Save complete analysis results (including flux data)
            full_json_file = self.metrics_dir / f"{job_name}_full_analysis.json"
            
            # Create serializable version (convert numpy arrays if present)
            serializable_results = {
                'job_name': results['job_name'],
                'parameters': results['parameters'],
                'metrics': results['metrics'],
                'analysis_timestamp': results['analysis_timestamp'],
                'flux_summary': {
                    'time_points': len(results['flux_data']['time_s']),
                    'max_flux': max(results['flux_data']['flux_g_m2_day']) if results['flux_data']['flux_g_m2_day'] else 0,
                    'final_cumulative': results['flux_data']['cumulative_g_m2'][-1] if results['flux_data']['cumulative_g_m2'] else 0
                }
            }
            
            with open(full_json_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            output_files['full_analysis'] = str(full_json_file)
            
        except Exception as e:
            if self.verbose:
                print(f"  WARNING: Error saving results: {str(e)}")
        
        return output_files

    def _create_plots(self, results):
        """Create diagnostic plots"""
        job_name = results['job_name']
        flux_data = results['flux_data']
        metrics = results['metrics']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        times_h = np.array(flux_data['time_s']) / 3600  # Convert to hours
        fluxes = np.array(flux_data['flux_g_m2_day'])
        
        # Plot 1: Flux vs time (linear scale)
        ax1.plot(times_h, fluxes, 'b-', linewidth=1.5)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Flux (g/m²/day)')
        ax1.set_title('Mass Flux vs Time (Linear)')
        ax1.grid(True, alpha=0.3)
        
        # Add breakthrough and lag time markers
        if metrics.get('breakthrough_time_h'):
            ax1.axvline(metrics['breakthrough_time_h'], color='r', linestyle='--', 
                       label=f"Breakthrough: {metrics['breakthrough_time_h']:.1f}h")
        if metrics.get('lag_time_h') and metrics['lag_time_h'] > 0:
            ax1.axvline(metrics['lag_time_h'], color='g', linestyle='--',
                       label=f"Lag time: {metrics['lag_time_h']:.1f}h")
        ax1.legend()
        
        # Plot 2: Flux vs time (log scale)
        positive_mask = fluxes > 0
        if np.any(positive_mask):
            ax2.semilogy(times_h[positive_mask], fluxes[positive_mask], 'r-', linewidth=1.5)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Flux (g/m²/day)')
        ax2.set_title('Mass Flux vs Time (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative mass
        cumulative = np.array(flux_data['cumulative_g_m2'])
        ax3.plot(times_h, cumulative, 'g-', linewidth=1.5)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Cumulative Mass (g/m²)')
        ax3.set_title('Cumulative Permeated Mass')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary text
        ax4.axis('off')
        summary_text = f"PERMEATION ANALYSIS SUMMARY\n" + "="*35 + "\n"
        summary_text += f"Job: {job_name}\n\n"
        summary_text += f"Parameters:\n"
        summary_text += f"  Crack width: {metrics['crack_width_nm']:.0f} nm\n"
        summary_text += f"  Crack spacing: {metrics['crack_spacing_nm']:.0f} nm\n"
        summary_text += f"  Crack fraction: {metrics['crack_fraction']:.4f}\n\n"
        summary_text += f"Results:\n"
        summary_text += f"  Steady-state flux: {metrics['steady_state_flux']:.2e} g/m²/day\n"
        if metrics.get('breakthrough_time_h'):
            summary_text += f"  Breakthrough time: {metrics['breakthrough_time_h']:.1f} h\n"
        if metrics.get('lag_time_h'):
            summary_text += f"  Lag time: {metrics['lag_time_h']:.1f} h\n"
        summary_text += f"  Total time: {metrics['total_time_h']:.1f} h\n"
        summary_text += f"  D_PET used: {self.D_PET:.1e} nm²/s"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'PECVD Barrier Permeation Analysis: {job_name}', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / f"{job_name}_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"  Saved plot: {plot_file.name}")
        
        return plot_file

    def process_batch(self, csv_files, save_plots=True):
        """
        Process multiple CSV files
        
        Args:
            csv_files: List of CSV file paths
            save_plots: Whether to generate plots
        
        Returns:
            Summary of all analyses
        """
        if self.verbose:
            print(f"\n=== BATCH FLUX PROCESSING ===")
            print(f"Processing {len(csv_files)} files")
        
        results_summary = {
            'total': len(csv_files),
            'successful': 0,
            'failed': 0,
            'results': []
        }
        
        for csv_file in csv_files:
            try:
                result = self.process_nnc_data(csv_file, save_plots=save_plots)
                results_summary['results'].append(result)
                
                if result.get('success'):
                    results_summary['successful'] += 1
                else:
                    results_summary['failed'] += 1
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ERROR processing {csv_file}: {str(e)}")
                results_summary['failed'] += 1
        
        # Save batch summary
        summary_file = self.output_dir / 'batch_analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        if self.verbose:
            print(f"\n=== BATCH PROCESSING COMPLETE ===")
            print(f"  Successful: {results_summary['successful']}")
            print(f"  Failed: {results_summary['failed']}")
            print(f"  Summary saved: {summary_file}")
        
        return results_summary

def main():
    """Main function for command line usage"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Process NNC gradient data to calculate mass flux')
    parser.add_argument('csv_file', nargs='?', help='Input CSV file with NNC gradient data')
    parser.add_argument('--batch', nargs='+', help='Process multiple CSV files')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--simulation', help='Simulation name (for organized output)')
    parser.add_argument('--job', help='Override job name')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.simulation:
        output_dir = Path('simulations') / args.simulation / 'analysis'
    elif args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('flux_analysis')
    
    # Create processor
    processor = FluxPostProcessor(
        output_dir=output_dir,
        verbose=not args.quiet
    )
    
    # Process files
    if args.batch:
        # Batch processing
        csv_files = []
        for pattern in args.batch:
            csv_files.extend(Path('.').glob(pattern))
        
        if not csv_files:
            print(f"ERROR: No CSV files found matching patterns: {args.batch}")
            return 1
        
        summary = processor.process_batch(csv_files, save_plots=not args.no_plots)
        
        if summary['successful'] > 0:
            return 0
        else:
            return 1
            
    elif args.csv_file:
        # Single file processing
        if not os.path.exists(args.csv_file):
            print(f"ERROR: Input file not found: {args.csv_file}")
            return 1
        
        results = processor.process_nnc_data(
            args.csv_file, 
            job_name=args.job,
            save_plots=not args.no_plots
        )
        
        if results.get('success'):
            if not args.quiet:
                print(f"\nKey metrics:")
                metrics = results.get('metrics', {})
                print(f"  Steady-state flux: {metrics.get('steady_state_flux', 'N/A')} g/(m²·day)")
                print(f"  Breakthrough time: {metrics.get('breakthrough_time_h', 'N/A')} hours")
                print(f"  Lag time: {metrics.get('lag_time_h', 'N/A')} hours")
            return 0
        else:
            print(f"Processing failed: {results.get('error', 'Unknown error')}")
            return 1
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
    