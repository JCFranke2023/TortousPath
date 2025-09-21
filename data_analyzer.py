"""
Data analyzer - runs in system Python with pandas/matplotlib
Processes extracted CSV files and creates plots
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

class DataAnalyzer:
    def __init__(self, data_dir='extracted_data', results_dir='analysis_results'):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.results_dir / 'plots'
        self.summary_dir = self.results_dir / 'summary'
        self.plots_dir.mkdir(exist_ok=True)
        self.summary_dir.mkdir(exist_ok=True)
    
    def calculate_metrics(self, flux_df):
        """Calculate permeation metrics from flux data"""
        metrics = {}
        
        # Steady-state flux (average of last 10% of data)
        n_points = len(flux_df)
        if n_points > 10:
            steady_points = max(1, n_points // 10)
            metrics['J_ss'] = flux_df['flux_g_m2_day'].iloc[-steady_points:].mean()
        else:
            metrics['J_ss'] = flux_df['flux_g_m2_day'].iloc[-1]
        
        # Breakthrough time (10% of steady-state)
        threshold = 0.1 * metrics['J_ss']
        breakthrough_mask = flux_df['flux_g_m2_day'] > threshold
        if breakthrough_mask.any():
            breakthrough_idx = breakthrough_mask.idxmax()
            metrics['t_b'] = flux_df.loc[breakthrough_idx, 'time_s'] / 3600  # hours
        else:
            metrics['t_b'] = np.nan
        
        # Lag time calculation
        if len(flux_df) > 5:
            cumulative = integrate.cumtrapz(flux_df['flux_g_m2_day'].values, 
                                          flux_df['time_s'].values, initial=0)
            
            # Find linear region (20% to 80% of steady state)
            mask = (flux_df['flux_g_m2_day'] > 0.2 * metrics['J_ss']) & \
                   (flux_df['flux_g_m2_day'] < 0.8 * metrics['J_ss'])
            
            if mask.sum() > 2:
                x_fit = flux_df.loc[mask, 'time_s'].values
                y_fit = cumulative[mask]
                
                # Linear fit
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
    
    def process_single_csv(self, csv_file, params=None):
        """Process a single CSV file"""
        print("Processing: {}".format(csv_file))
        
        # Read CSV
        flux_df = pd.read_csv(csv_file)
        
        # Calculate metrics
        metrics = self.calculate_metrics(flux_df)
        
        # Add parameters if provided
        if params:
            metrics.update(params)
        
        # Extract job name from filename
        job_name = Path(csv_file).stem.replace('_flux', '')
        
        # Create plot
        self.plot_flux_vs_time(flux_df, job_name, metrics)
        
        return metrics
    
    def plot_flux_vs_time(self, flux_df, job_name, metrics):
        """Create flux vs time plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        ax1.plot(flux_df['time_s']/3600, flux_df['flux_g_m2_day'], 'b-', linewidth=1.5)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Flux (g/m²/day)')
        ax1.set_title('Linear Scale')
        ax1.grid(True, alpha=0.3)
        
        # Add metrics text
        text = 'J_ss = {:.2e} g/m²/day\nt_b = {:.2f} h\nt_lag = {:.2f} h'.format(
            metrics['J_ss'], metrics.get('t_b', 0), metrics.get('t_lag', 0))
        ax1.text(0.95, 0.95, text, transform=ax1.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Log scale
        if (flux_df['flux_g_m2_day'] > 0).any():
            positive_mask = flux_df['flux_g_m2_day'] > 0
            ax2.semilogy(flux_df.loc[positive_mask, 'time_s']/3600, 
                        flux_df.loc[positive_mask, 'flux_g_m2_day'], 
                        'r-', linewidth=1.5)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Flux (g/m²/day)')
        ax2.set_title('Log Scale')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle('Flux Evolution: {}'.format(job_name))
        plt.tight_layout()
        
        # Save
        plot_path = self.plots_dir / '{}_flux.png'.format(job_name)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved plot: {}".format(plot_path))
    
    def process_parameter_sweep(self, param_file=None):
        """Process all CSV files in data directory"""
        print("\n=== PROCESSING PARAMETER SWEEP ===")
        
        # Find all CSV files
        csv_files = list(self.data_dir.glob('*_flux.csv'))
        print("Found {} CSV files".format(len(csv_files)))
        
        # Load parameters if provided
        params_dict = {}
        if param_file and Path(param_file).exists():
            with open(param_file, 'r') as f:
                params_dict = json.load(f)
        
        # Process each file
        all_metrics = []
        for csv_file in csv_files:
            job_name = csv_file.stem.replace('_flux', '')
            params = params_dict.get(job_name, {})
            
            # Try to parse parameters from filename if not in dict
            if not params:
                try:
                    # Parse Job_c100_s10000_o25 format
                    parts = job_name.split('_')
                    if len(parts) >= 4:
                        params = {
                            'crack_width': float(parts[1][1:]),
                            'crack_spacing': float(parts[2][1:]),
                            'crack_offset': float(parts[3][1:]) / 100
                        }
                except:
                    pass
            
            metrics = self.process_single_csv(csv_file, params)
            metrics['job_name'] = job_name
            all_metrics.append(metrics)
        
        # Create summary dataframe
        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)
            
            # Save summary
            summary_path = self.summary_dir / 'parameter_sweep_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print("\nSaved summary: {}".format(summary_path))
            
            # Create plots if we have parameter data
            if 'crack_width' in summary_df.columns:
                self.create_sensitivity_plots(summary_df)
            
            return summary_df
        
        return None
    
    def create_sensitivity_plots(self, df):
        """Create parameter sensitivity plots"""
        print("\nCreating sensitivity plots...")
        
        # Setup figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Width sensitivity
        if 'crack_width' in df.columns:
            ax = axes[0]
            unique_offsets = df['crack_offset'].unique() if 'crack_offset' in df.columns else [0]
            for offset in unique_offsets:
                mask = df['crack_offset'] == offset if 'crack_offset' in df.columns else df.index
                data = df[mask].sort_values('crack_width')
                ax.plot(data['crack_width'], data['J_ss'], 
                       'o-', label='offset={:.2f}'.format(offset))
            ax.set_xlabel('Crack Width (nm)')
            ax.set_ylabel('Steady-State Flux (g/m²/day)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Width Sensitivity')
        
        # Plot 2: Spacing sensitivity
        if 'crack_spacing' in df.columns:
            ax = axes[1]
            unique_offsets = df['crack_offset'].unique() if 'crack_offset' in df.columns else [0]
            for offset in unique_offsets:
                mask = df['crack_offset'] == offset if 'crack_offset' in df.columns else df.index
                data = df[mask].sort_values('crack_spacing')
                ax.plot(data['crack_spacing']/1000, data['J_ss'],
                       's-', label='offset={:.2f}'.format(offset))
            ax.set_xlabel('Crack Spacing (µm)')
            ax.set_ylabel('Steady-State Flux (g/m²/day)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Spacing Sensitivity')
        
        # Plot 3: Offset sensitivity
        if 'crack_offset' in df.columns:
            ax = axes[2]
            unique_widths = df['crack_width'].unique()
            for width in unique_widths[:3]:  # Limit to 3 for clarity
                mask = df['crack_width'] == width
                data = df[mask].sort_values('crack_offset')
                ax.plot(data['crack_offset'], data['J_ss'],
                       '^-', label='c={:.0f}nm'.format(width))
            ax.set_xlabel('Crack Offset')
            ax.set_ylabel('Steady-State Flux (g/m²/day)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Offset Sensitivity')
        
        # Plot 4: Breakthrough vs Flux
        ax = axes[3]
        if 't_b' in df.columns:
            valid = df['t_b'].notna()
            ax.scatter(df.loc[valid, 'J_ss'], df.loc[valid, 't_b'])
            ax.set_xlabel('Steady-State Flux (g/m²/day)')
            ax.set_ylabel('Breakthrough Time (hours)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_title('Breakthrough vs Flux')
        
        # Plot 5: Lag time vs Flux
        ax = axes[4]
        if 't_lag' in df.columns:
            valid = df['t_lag'].notna()
            ax.scatter(df.loc[valid, 'J_ss'], df.loc[valid, 't_lag'])
            ax.set_xlabel('Steady-State Flux (g/m²/day)')
            ax.set_ylabel('Lag Time (hours)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_title('Lag Time vs Flux')
        
        # Plot 6: Summary statistics
        ax = axes[5]
        ax.axis('off')
        summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n"
        summary_text += "Total simulations: {}\n".format(len(df))
        summary_text += "J_ss range: {:.2e} - {:.2e}\n".format(df['J_ss'].min(), df['J_ss'].max())
        if 't_b' in df.columns:
            summary_text += "t_b range: {:.1f} - {:.1f} h\n".format(
                df['t_b'].min(), df['t_b'].max())
        if 't_lag' in df.columns:
            summary_text += "t_lag range: {:.1f} - {:.1f} h\n".format(
                df['t_lag'].min(), df['t_lag'].max())
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Parameter Sensitivity Analysis', fontsize=14)
        plt.tight_layout()
        
        # Save
        plot_path = self.plots_dir / 'sensitivity_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(plot_path))


# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze extracted flux data')
    parser.add_argument('--csv', help='Single CSV file to process')
    parser.add_argument('--sweep', action='store_true', help='Process all CSV files')
    parser.add_argument('--params', help='JSON file with parameters')
    parser.add_argument('--data_dir', default='extracted_data', help='Input directory')
    parser.add_argument('--output', default='analysis_results', help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = DataAnalyzer(args.data_dir, args.output)
    
    if args.csv:
        # Single file
        metrics = analyzer.process_single_csv(args.csv)
        print("\nMetrics:")
        for key, value in metrics.items():
            print("  {}: {}".format(key, value))
    
    elif args.sweep:
        # Process all files
        summary_df = analyzer.process_parameter_sweep(args.params)
        if summary_df is not None:
            print("\n=== ANALYSIS COMPLETE ===")
            print("Results saved in: {}".format(analyzer.results_dir))
    
    else:
        print("Usage:")
        print("  Single: python data_analyzer.py --csv file.csv")
        print("  Sweep:  python data_analyzer.py --sweep")