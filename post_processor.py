"""
Post-processor for ABAQUS permeation simulation results
Extracts flux data, calculates metrics, and generates plots
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
import json

# Try importing ABAQUS modules
try:
    from odbAccess import openOdb
    from abaqusConstants import *
    ABAQUS_ENV = True
except ImportError:
    ABAQUS_ENV = False
    print("Running without ABAQUS - limited functionality")

class PostProcessor:
    def __init__(self, results_dir='simulation_results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.results_dir / 'plots'
        self.data_dir = self.results_dir / 'processed_data'
        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        

    def extract_flux_data(self, odb_path):
        """Extract flux vs time from ODB file (tries history first, then field)"""
        if not ABAQUS_ENV:
            print("Cannot extract from ODB without ABAQUS")
            return None
            
        print("Extracting flux data from: {}".format(odb_path))
        
        # Try history output first (more efficient)
        df = self._extract_from_history(odb_path)
        if df is not None:
            print("  Successfully extracted from history output")
            return df
        
        # Fallback to field output
        print("  No history data found, extracting from field output...")
        df = self._extract_from_field(odb_path)
        if df is not None:
            print("  Successfully extracted from field output")
            return df
        
        print("  ERROR: Could not extract flux data")
        return None

    def _extract_from_history(self, odb_path):
        """Extract flux from history output (preferred method)"""
        try:
            odb = openOdb(odb_path)
            step = odb.steps['Permeation']
            
            # Look for history regions
            for region_name, region in step.historyRegions.items():
                if 'Bottom' in region_name or 'Outlet' in region_name:
                    if 'FMFL' in region.historyOutputs:
                        fmfl = region.historyOutputs['FMFL']
                        
                        time_data = []
                        flux_data = []
                        
                        for time, value in fmfl.data:
                            time_data.append(time)
                            # Note: FMFL is integrated, need to divide by area
                            area = (self.geometry.crack_spacing * 1e-9)**2  # nm² to m²
                            flux_data.append(abs(value) / area)
                        
                        odb.close()
                        
                        df = pd.DataFrame({
                            'time_s': time_data,
                            'flux_kg_m2_s': flux_data
                        })
                        df['flux_g_m2_day'] = df['flux_kg_m2_s'] * 1000 * 86400
                        return df
            
            odb.close()
            return None
            
        except Exception as e:
            return None

    def _extract_from_field(self, odb_path):
        """Extract flux from field output (fallback method)"""
        try:
            odb = openOdb(odb_path)
            
                        # Get assembly and instance
            assembly = odb.rootAssembly
            instance_name = assembly.instances.keys()[0]
            instance = assembly.instances[instance_name]
            
            # Get step
            step_name = 'Permeation'
            step = odb.steps[step_name]
            
            # Initialize data storage
            time_data = []
            flux_data = []
            
            # Loop through frames
            for frame in step.frames:
                time = frame.frameValue
                
                # Get flux field
                if 'MFL' in frame.fieldOutputs:
                    mfl = frame.fieldOutputs['MFL']
                    
                    # Get bottom surface nodes (y=0)
                    bottom_flux = []
                    for value in mfl.values:
                        if hasattr(value, 'nodeLabel'):
                            node = instance.nodes[value.nodeLabel - 1]
                            if abs(node.coordinates[1]) < 1.0:  # y ~ 0
                                # MFL2 is y-component of flux
                                bottom_flux.append(abs(value.data[1]))
                    
                    if bottom_flux:
                        avg_flux = np.mean(bottom_flux)
                        time_data.append(time)
                        flux_data.append(avg_flux)
            
            # Create dataframe
            df = pd.DataFrame({
                'time_s': time_data,
                'flux_kg_m2_s': flux_data
            })
            
            # Convert to g/m²/day
            df['flux_g_m2_day'] = df['flux_kg_m2_s'] * 1000 * 86400
            
            odb.close()
            return df
            
        except Exception as e:
            return None


    def calculate_metrics(self, flux_df):
        """Calculate permeation metrics from flux data"""
        if flux_df is None or flux_df.empty:
            return None
            
        metrics = {}
        
        # Steady-state flux (average of last 10% of data)
        n_points = len(flux_df)
        steady_points = n_points // 10
        metrics['J_ss'] = flux_df['flux_g_m2_day'].iloc[-steady_points:].mean()
        
        # Breakthrough time (10% of steady-state)
        threshold = 0.1 * metrics['J_ss']
        breakthrough_idx = flux_df[flux_df['flux_g_m2_day'] > threshold].index
        if len(breakthrough_idx) > 0:
            metrics['t_b'] = flux_df.loc[breakthrough_idx[0], 'time_s'] / 3600  # hours
        else:
            metrics['t_b'] = np.nan
        
        # Lag time (from cumulative permeation)
        cumulative = integrate.cumtrapz(flux_df['flux_g_m2_day'], flux_df['time_s'], initial=0)
        
        # Fit linear portion (20% to 80% of steady state)
        steady_range = (flux_df['flux_g_m2_day'] > 0.2 * metrics['J_ss']) & \
                      (flux_df['flux_g_m2_day'] < 0.8 * metrics['J_ss'])
        
        if steady_range.any():
            x_fit = flux_df.loc[steady_range, 'time_s'].values
            y_fit = cumulative[steady_range]
            
            if len(x_fit) > 2:
                # Linear fit: y = mt + b, lag time = -b/m
                coeffs = np.polyfit(x_fit, y_fit, 1)
                if coeffs[0] > 0:
                    metrics['t_lag'] = -coeffs[1] / coeffs[0] / 3600  # hours
                else:
                    metrics['t_lag'] = np.nan
            else:
                metrics['t_lag'] = np.nan
        else:
            metrics['t_lag'] = np.nan
        
        return metrics
    
    def plot_flux_vs_time(self, flux_df, params, save_path=None):
        """Create flux vs time plot"""
        if flux_df is None or flux_df.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        ax1.plot(flux_df['time_s']/3600, flux_df['flux_g_m2_day'], 'b-', linewidth=1.5)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Flux (g/m²/day)')
        ax1.set_title('Flux vs Time - Linear Scale')
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.semilogy(flux_df['time_s']/3600, flux_df['flux_g_m2_day'], 'r-', linewidth=1.5)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Flux (g/m²/day)')
        ax2.set_title('Flux vs Time - Log Scale')
        ax2.grid(True, alpha=0.3)
        
        # Add parameter info
        param_text = 'c={:.0f}nm, d={:.0f}µm, o={:.2f}'.format(
            params['crack_width'],
            params['crack_spacing']/1000,
            params['crack_offset']
        )
        fig.suptitle('Permeation Results: {}'.format(param_text))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_single_job(self, job_name, params):
        """Process a single simulation job"""
        print("\nProcessing job: {}".format(job_name))
        
        # Find ODB file
        odb_path = Path('abaqus_files/jobs') / job_name / '{}.odb'.format(job_name)
        if not odb_path.exists():
            print("  ODB file not found: {}".format(odb_path))
            return None
        
        # Extract flux data
        flux_df = self.extract_flux_data(str(odb_path))
        if flux_df is None:
            return None
        
        # Save flux data
        csv_path = self.data_dir / '{}_flux.csv'.format(job_name)
        flux_df.to_csv(csv_path, index=False)
        print("  Saved flux data: {}".format(csv_path))
        
        # Calculate metrics
        metrics = self.calculate_metrics(flux_df)
        metrics.update(params)
        
        # Create plot
        plot_path = self.plots_dir / '{}_flux.png'.format(job_name)
        self.plot_flux_vs_time(flux_df, params, plot_path)
        print("  Created plot: {}".format(plot_path))
        
        # Print metrics
        if metrics:
            print("  Metrics:")
            print("    J_ss = {:.3e} g/m²/day".format(metrics['J_ss']))
            print("    t_b = {:.2f} hours".format(metrics['t_b']))
            print("    t_lag = {:.2f} hours".format(metrics['t_lag']))
        
        return metrics
    
    def process_parameter_sweep(self, sweep_results):
        """Process all jobs from a parameter sweep"""
        print("\n=== PROCESSING PARAMETER SWEEP ===")
        
        all_metrics = []
        
        for result in sweep_results:
            job_name = result['job_name']
            params = result['parameters']
            
            metrics = self.process_single_job(job_name, params)
            if metrics:
                all_metrics.append(metrics)
        
        # Create summary dataframe
        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)
            
            # Save summary
            summary_path = self.data_dir / 'parameter_sweep_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print("\nSaved summary: {}".format(summary_path))
            
            # Create comparison plots
            self.create_sensitivity_plots(summary_df)
            
            return summary_df
        
        return None
    
    def create_sensitivity_plots(self, summary_df):
        """Create parameter sensitivity plots"""
        print("\nCreating sensitivity plots...")
        
        # Setup figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Crack width sensitivity
        ax = axes[0, 0]
        for offset in summary_df['crack_offset'].unique():
            mask = summary_df['crack_offset'] == offset
            ax.plot(summary_df[mask]['crack_width'], 
                   summary_df[mask]['J_ss'],
                   'o-', label='offset={:.2f}'.format(offset))
        ax.set_xlabel('Crack Width (nm)')
        ax.set_ylabel('Steady-State Flux (g/m²/day)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Width Sensitivity')
        
        # Plot 2: Crack spacing sensitivity
        ax = axes[0, 1]
        for offset in summary_df['crack_offset'].unique():
            mask = summary_df['crack_offset'] == offset
            ax.plot(summary_df[mask]['crack_spacing']/1000,
                   summary_df[mask]['J_ss'],
                   's-', label='offset={:.2f}'.format(offset))
        ax.set_xlabel('Crack Spacing (µm)')
        ax.set_ylabel('Steady-State Flux (g/m²/day)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Spacing Sensitivity')
        
        # Plot 3: Offset sensitivity
        ax = axes[0, 2]
        for width in summary_df['crack_width'].unique():
            mask = summary_df['crack_width'] == width
            ax.plot(summary_df[mask]['crack_offset'],
                   summary_df[mask]['J_ss'],
                   '^-', label='c={:.0f}nm'.format(width))
        ax.set_xlabel('Crack Offset')
        ax.set_ylabel('Steady-State Flux (g/m²/day)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Offset Sensitivity')
        
        # Plot 4: Breakthrough time
        ax = axes[1, 0]
        ax.scatter(summary_df['J_ss'], summary_df['t_b'], 
                  c=summary_df['crack_offset'], cmap='viridis')
        ax.set_xlabel('Steady-State Flux (g/m²/day)')
        ax.set_ylabel('Breakthrough Time (hours)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title('Breakthrough vs Flux')
        
        # Plot 5: Lag time
        ax = axes[1, 1]
        ax.scatter(summary_df['J_ss'], summary_df['t_lag'],
                  c=summary_df['crack_offset'], cmap='viridis')
        ax.set_xlabel('Steady-State Flux (g/m²/day)')
        ax.set_ylabel('Lag Time (hours)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title('Lag Time vs Flux')
        
        # Plot 6: Barrier improvement factor
        ax = axes[1, 2]
        # Assuming uncoated flux ~ 1000 g/m²/day for PET
        BIF = 1000 / summary_df['J_ss']
        ax.scatter(summary_df['crack_spacing']/1000, BIF,
                  c=summary_df['crack_width'], cmap='plasma')
        ax.set_xlabel('Crack Spacing (µm)')
        ax.set_ylabel('Barrier Improvement Factor')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title('Barrier Performance')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Crack Width (nm)')
        
        plt.suptitle('Parameter Sensitivity Analysis')
        plt.tight_layout()
        
        # Save
        plot_path = self.plots_dir / 'sensitivity_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(plot_path))
    
    def generate_report(self, summary_df):
        """Generate analysis report"""
        print("\n=== ANALYSIS REPORT ===")
        
        report_path = self.results_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("PECVD BARRIER COATING PERMEATION ANALYSIS\n")
            f.write("="*50 + "\n\n")
            
            # Parameter ranges
            f.write("PARAMETER RANGES:\n")
            f.write("-"*30 + "\n")
            f.write("Crack width: {:.0f} - {:.0f} nm\n".format(
                summary_df['crack_width'].min(),
                summary_df['crack_width'].max()))
            f.write("Crack spacing: {:.1f} - {:.1f} µm\n".format(
                summary_df['crack_spacing'].min()/1000,
                summary_df['crack_spacing'].max()/1000))
            f.write("Crack offset: {:.2f} - {:.2f}\n\n".format(
                summary_df['crack_offset'].min(),
                summary_df['crack_offset'].max()))
            
            # Performance metrics
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-"*30 + "\n")
            f.write("Best configuration (minimum flux):\n")
            best_idx = summary_df['J_ss'].idxmin()
            best = summary_df.loc[best_idx]
            f.write("  c={:.0f}nm, d={:.1f}µm, o={:.2f}\n".format(
                best['crack_width'],
                best['crack_spacing']/1000,
                best['crack_offset']))
            f.write("  J_ss = {:.3e} g/m²/day\n".format(best['J_ss']))
            f.write("  t_b = {:.2f} hours\n".format(best['t_b']))
            f.write("  t_lag = {:.2f} hours\n\n".format(best['t_lag']))
            
            # Parameter sensitivities
            f.write("PARAMETER SENSITIVITIES:\n")
            f.write("-"*30 + "\n")
            
            # Width sensitivity
            width_range = summary_df['crack_width'].max() / summary_df['crack_width'].min()
            flux_change_width = summary_df.groupby('crack_width')['J_ss'].mean()
            width_sensitivity = flux_change_width.max() / flux_change_width.min()
            f.write("Crack width ({:.0f}x range): {:.1f}x flux change\n".format(
                width_range, width_sensitivity))
            
            # Spacing sensitivity
            spacing_range = summary_df['crack_spacing'].max() / summary_df['crack_spacing'].min()
            flux_change_spacing = summary_df.groupby('crack_spacing')['J_ss'].mean()
            spacing_sensitivity = flux_change_spacing.max() / flux_change_spacing.min()
            f.write("Crack spacing ({:.0f}x range): {:.1f}x flux change\n".format(
                spacing_range, spacing_sensitivity))
            
            # Offset sensitivity
            flux_change_offset = summary_df.groupby('crack_offset')['J_ss'].mean()
            offset_sensitivity = flux_change_offset.max() / flux_change_offset.min()
            f.write("Crack offset: {:.1f}x flux change\n".format(offset_sensitivity))
            
            f.write("\nRANKING: Width > Spacing > Offset\n")
        
        print("Report saved: {}".format(report_path))


# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Post-process permeation results')
    parser.add_argument('--job', help='Single job name to process')
    parser.add_argument('--sweep', help='Parameter sweep results JSON file')
    parser.add_argument('--params', help='Parameters JSON for single job')
    
    args = parser.parse_args()
    
    processor = PostProcessor()
    
    if args.job:
        # Process single job
        params = {'crack_width': 100, 'crack_spacing': 10000, 'crack_offset': 0.25}
        if args.params:
            with open(args.params, 'r') as f:
                params = json.load(f)
        
        metrics = processor.process_single_job(args.job, params)
        
    elif args.sweep:
        # Process parameter sweep
        with open(args.sweep, 'r') as f:
            sweep_results = json.load(f)
        
        summary_df = processor.process_parameter_sweep(sweep_results)
        if summary_df is not None:
            processor.generate_report(summary_df)
    
    else:
        print("Please specify --job or --sweep")
        print("Example: python post_processor.py --job Job_c100_s10000_o25")
        print("Example: python post_processor.py --sweep sweep_results.json")