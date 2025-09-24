#!/usr/bin/env python
"""
Campaign runner for PECVD barrier coating simulations
Manages execution of multiple simulations with parameter sweeps
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

class CampaignRunner:
    """Run and manage simulation campaigns"""
    
    def __init__(self, campaign_name: str, campaign_dir: Optional[Path] = None):
        """
        Initialize campaign runner
        
        Args:
            campaign_name: Name of the campaign
            campaign_dir: Directory containing campaign configuration
        """
        self.campaign_name = campaign_name
        
        if campaign_dir:
            self.campaign_dir = Path(campaign_dir)
        else:
            self.campaign_dir = Path('batch_runs') / campaign_name
        
        if not self.campaign_dir.exists():
            raise ValueError(f"Campaign directory not found: {self.campaign_dir}")
        
        # Load campaign configuration
        self.config_file = self.campaign_dir / 'campaign_config.json'
        if not self.config_file.exists():
            raise ValueError(f"Campaign config not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            self.campaign_config = json.load(f)
        
        # Setup directories
        self.results_dir = self.campaign_dir / 'summary_analysis'
        self.results_dir.mkdir(exist_ok=True)
        
        # Campaign log file
        self.log_file = self.campaign_dir / 'campaign_log.txt'
        
        # Simulation status tracking
        self.status_file = self.campaign_dir / 'campaign_status.json'
        self.load_or_init_status()
        
        # Execution settings
        self.max_parallel = 1  # Sequential by default (ABAQUS licensing)
        self.stage = 4  # Full pipeline by default
        self.retry_failed = True
        self.continue_on_error = True
    
    def load_or_init_status(self):
        """Load existing status or initialize new one"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {
                'campaign_name': self.campaign_name,
                'total_simulations': len(self.campaign_config['simulations']),
                'completed': [],
                'failed': [],
                'pending': [s['simulation_name'] for s in self.campaign_config['simulations']],
                'running': [],
                'start_time': None,
                'end_time': None,
                'last_update': datetime.now().isoformat()
            }
            self.save_status()
    
    def save_status(self):
        """Save current status to file"""
        self.status['last_update'] = datetime.now().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_simulation(self, sim_config: Dict) -> Dict:
        """
        Run a single simulation
        
        Args:
            sim_config: Simulation configuration
        
        Returns:
            Result dictionary
        """
        sim_name = sim_config['simulation_name']
        self.log(f"Starting simulation: {sim_name}")
        
        # Update status
        self.status['running'].append(sim_name)
        if sim_name in self.status['pending']:
            self.status['pending'].remove(sim_name)
        self.save_status()
        
        result = {
            'simulation_name': sim_name,
            'job_name': sim_config['job_name'],
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'outputs': {}
        }
        
        try:
            # Load configuration file if needed
            config_file = sim_config.get('config_file')
            if config_file:
                config_path = self.campaign_dir / config_file
                with open(config_path, 'r') as f:
                    full_config = json.load(f)
                    params = full_config['parameters']
            else:
                params = sim_config['parameters']
            
            # Build command for batch runner
            cmd = [
                'python', 'batch_runner.py',
                '--name', sim_name,
                '--stage', str(self.stage),
                '--crack_width', str(params['crack_width']),
                '--crack_spacing', str(params['crack_spacing']),
                '--crack_offset', str(params['crack_offset'])
            ]
            
            self.log(f"  Executing: {' '.join(cmd)}")
            
            # Run the simulation
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if process.returncode == 0:
                result['success'] = True
                self.log(f"  ✓ Simulation completed: {sim_name}")
                
                # Parse outputs
                self._parse_simulation_outputs(sim_name, result)
                
                # Update status
                self.status['completed'].append(sim_name)
            else:
                result['error'] = f"Process failed with code {process.returncode}"
                if process.stderr:
                    result['error'] += f"\n{process.stderr}"
                self.log(f"  ✗ Simulation failed: {sim_name}", "ERROR")
                self.status['failed'].append(sim_name)
            
            # Save stdout/stderr
            if process.stdout:
                log_dir = Path('simulations') / sim_name / 'summary'
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / 'campaign_stdout.txt', 'w') as f:
                    f.write(process.stdout)
            
        except subprocess.TimeoutExpired:
            result['error'] = "Simulation timed out"
            self.log(f"  ✗ Timeout: {sim_name}", "ERROR")
            self.status['failed'].append(sim_name)
            
        except Exception as e:
            result['error'] = str(e)
            self.log(f"  ✗ Exception: {sim_name} - {e}", "ERROR")
            self.status['failed'].append(sim_name)
        
        finally:
            # Update status
            if sim_name in self.status['running']:
                self.status['running'].remove(sim_name)
            result['end_time'] = datetime.now().isoformat()
            self.save_status()
        
        return result
    
    def _parse_simulation_outputs(self, sim_name: str, result: Dict):
        """Parse and extract key outputs from simulation"""
        sim_dir = Path('simulations') / sim_name
        
        # Check for key output files
        metrics_file = sim_dir / 'summary' / 'metrics_summary.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                result['outputs']['metrics'] = metrics.get('metrics', {})
        
        # Check for ODB file
        models_dir = sim_dir / 'models'
        if models_dir.exists():
            odb_files = list(models_dir.glob('*.odb'))
            if odb_files:
                result['outputs']['odb_file'] = str(odb_files[0])
        
        # Check for plots
        plots_dir = sim_dir / 'analysis' / 'plots'
        if plots_dir.exists():
            plot_files = list(plots_dir.glob('*.png'))
            if plot_files:
                result['outputs']['plots'] = [str(p) for p in plot_files]
    
    def run_campaign(self, parallel: bool = False, max_workers: int = 1):
        """
        Run all simulations in the campaign
        
        Args:
            parallel: Whether to run simulations in parallel
            max_workers: Maximum number of parallel simulations
        """
        self.log("=" * 60)
        self.log(f"STARTING CAMPAIGN: {self.campaign_name}")
        self.log(f"Total simulations: {len(self.campaign_config['simulations'])}")
        self.log(f"Execution mode: {'Parallel' if parallel else 'Sequential'}")
        self.log("=" * 60)
        
        self.status['start_time'] = datetime.now().isoformat()
        self.save_status()
        
        results = []
        
        if parallel and max_workers > 1:
            # Parallel execution
            self.log(f"Running with {max_workers} workers")
            results = self._run_parallel(max_workers)
        else:
            # Sequential execution
            results = self._run_sequential()
        
        self.status['end_time'] = datetime.now().isoformat()
        self.save_status()
        
        # Generate summary
        self._generate_campaign_summary(results)
        
        self.log("=" * 60)
        self.log(f"CAMPAIGN COMPLETED: {self.campaign_name}")
        self.log(f"Successful: {len(self.status['completed'])}")
        self.log(f"Failed: {len(self.status['failed'])}")
        self.log("=" * 60)
        
        return results
    
    def _run_sequential(self) -> List[Dict]:
        """Run simulations sequentially"""
        results = []
        
        for sim_config in self.campaign_config['simulations']:
            sim_name = sim_config['simulation_name']
            
            # Skip if already completed
            if sim_name in self.status['completed']:
                self.log(f"Skipping completed: {sim_name}")
                continue
            
            # Skip or retry failed
            if sim_name in self.status['failed']:
                if self.retry_failed:
                    self.log(f"Retrying failed: {sim_name}")
                    self.status['failed'].remove(sim_name)
                    self.status['pending'].append(sim_name)
                else:
                    self.log(f"Skipping failed: {sim_name}")
                    continue
            
            # Run simulation
            result = self.run_simulation(sim_config)
            results.append(result)
            
            # Check if we should continue after failure
            if not result['success'] and not self.continue_on_error:
                self.log("Stopping campaign due to simulation failure", "ERROR")
                break
        
        return results
    
    def _run_parallel(self, max_workers: int) -> List[Dict]:
        """Run simulations in parallel"""
        results = []
        
        # Filter simulations to run
        to_run = []
        for sim_config in self.campaign_config['simulations']:
            sim_name = sim_config['simulation_name']
            
            if sim_name in self.status['completed']:
                continue
            
            if sim_name in self.status['failed'] and not self.retry_failed:
                continue
            
            to_run.append(sim_config)
        
        self.log(f"Simulations to run: {len(to_run)}")
        
        # Run with process pool
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sim = {
                executor.submit(self.run_simulation, sim_config): sim_config
                for sim_config in to_run
            }
            
            # Process completed simulations
            for future in as_completed(future_to_sim):
                sim_config = future_to_sim[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.log(f"Exception in parallel execution: {e}", "ERROR")
                    results.append({
                        'simulation_name': sim_config['simulation_name'],
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def _generate_campaign_summary(self, results: List[Dict]):
        """Generate campaign summary and analysis"""
        summary = {
            'campaign_name': self.campaign_name,
            'timestamp': datetime.now().isoformat(),
            'total_simulations': len(self.campaign_config['simulations']),
            'completed': len(self.status['completed']),
            'failed': len(self.status['failed']),
            'results': []
        }
        
        # Collect all metrics
        all_metrics = []
        
        for result in results:
            sim_summary = {
                'simulation_name': result['simulation_name'],
                'job_name': result['job_name'],
                'success': result['success']
            }
            
            if result['success'] and 'metrics' in result.get('outputs', {}):
                metrics = result['outputs']['metrics']
                sim_summary['metrics'] = metrics
                
                # Add parameters for correlation
                for sim in self.campaign_config['simulations']:
                    if sim['simulation_name'] == result['simulation_name']:
                        sim_summary['parameters'] = sim['parameters']
                        break
                
                all_metrics.append(sim_summary)
            
            summary['results'].append(sim_summary)
        
        # Save summary
        summary_file = self.results_dir / 'campaign_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"Campaign summary saved: {summary_file}")
        
        # Generate aggregate analysis if we have metrics
        if all_metrics:
            self._generate_aggregate_analysis(all_metrics)
    
    def _generate_aggregate_analysis(self, metrics_data: List[Dict]):
        """Generate aggregate analysis across all simulations"""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create dataframe
            rows = []
            for data in metrics_data:
                row = {
                    'simulation': data['simulation_name'],
                    'crack_width': data['parameters']['crack_width'],
                    'crack_spacing': data['parameters']['crack_spacing'],
                    'crack_offset': data['parameters']['crack_offset']
                }
                
                if 'metrics' in data:
                    m = data['metrics']
                    row.update({
                        'steady_state_flux': m.get('steady_state_flux'),
                        'breakthrough_time_h': m.get('breakthrough_time_h'),
                        'lag_time_h': m.get('lag_time_h'),
                        'crack_fraction': m.get('crack_fraction')
                    })
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Save as CSV
            csv_file = self.results_dir / 'campaign_results.csv'
            df.to_csv(csv_file, index=False)
            self.log(f"Results CSV saved: {csv_file}")
            
            # Generate plots if we have enough data
            if len(df) > 3:
                self._create_summary_plots(df)
            
            # Calculate statistics
            stats = {
                'parameter_ranges': {
                    'crack_width': [df['crack_width'].min(), df['crack_width'].max()],
                    'crack_spacing': [df['crack_spacing'].min(), df['crack_spacing'].max()],
                    'crack_offset': [df['crack_offset'].min(), df['crack_offset'].max()]
                },
                'flux_statistics': {
                    'min': df['steady_state_flux'].min(),
                    'max': df['steady_state_flux'].max(),
                    'mean': df['steady_state_flux'].mean(),
                    'std': df['steady_state_flux'].std()
                }
            }
            
            stats_file = self.results_dir / 'campaign_statistics.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.log(f"Statistics saved: {stats_file}")
            
        except ImportError:
            self.log("Pandas/matplotlib not available for aggregate analysis", "WARNING")
        except Exception as e:
            self.log(f"Error in aggregate analysis: {e}", "ERROR")
    
    def _create_summary_plots(self, df):
        """Create summary plots for campaign results"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Flux vs crack width
            if df['crack_width'].nunique() > 1:
                ax = axes[0, 0]
                for offset in df['crack_offset'].unique():
                    mask = df['crack_offset'] == offset
                    ax.loglog(df[mask]['crack_width'], 
                             df[mask]['steady_state_flux'], 
                             'o-', label=f'Offset={offset}')
                ax.set_xlabel('Crack Width (nm)')
                ax.set_ylabel('Steady-State Flux (g/m²/day)')
                ax.set_title('Flux vs Crack Width')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Flux vs crack spacing
            if df['crack_spacing'].nunique() > 1:
                ax = axes[0, 1]
                for offset in df['crack_offset'].unique():
                    mask = df['crack_offset'] == offset
                    ax.loglog(df[mask]['crack_spacing'],
                             df[mask]['steady_state_flux'],
                             's-', label=f'Offset={offset}')
                ax.set_xlabel('Crack Spacing (nm)')
                ax.set_ylabel('Steady-State Flux (g/m²/day)')
                ax.set_title('Flux vs Crack Spacing')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 3: Flux vs crack fraction
            ax = axes[1, 0]
            ax.loglog(df['crack_fraction'], df['steady_state_flux'], 'ko')
            ax.set_xlabel('Crack Fraction')
            ax.set_ylabel('Steady-State Flux (g/m²/day)')
            ax.set_title('Flux vs Crack Fraction')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Parameter correlation heatmap
            ax = axes[1, 1]
            params = ['crack_width', 'crack_spacing', 'crack_offset', 'steady_state_flux']
            corr_data = df[params].corr()
            im = ax.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(params)))
            ax.set_yticks(range(len(params)))
            ax.set_xticklabels(params, rotation=45, ha='right')
            ax.set_yticklabels(params)
            ax.set_title('Parameter Correlations')
            plt.colorbar(im, ax=ax)
            
            # Add correlation values
            for i in range(len(params)):
                for j in range(len(params)):
                    text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                 ha='center', va='center', color='black')
            
            plt.suptitle(f'Campaign Summary: {self.campaign_name}', fontsize=14)
            plt.tight_layout()
            
            plot_file = self.results_dir / 'campaign_summary_plots.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"Summary plots saved: {plot_file}")
            
        except Exception as e:
            self.log(f"Error creating plots: {e}", "WARNING")
    
    def get_status_report(self) -> str:
        """Get current campaign status report"""
        report = f"""
Campaign Status Report: {self.campaign_name}
{'=' * 50}
Total simulations: {self.status['total_simulations']}
Completed: {len(self.status['completed'])} ({100*len(self.status['completed'])/self.status['total_simulations']:.1f}%)
Failed: {len(self.status['failed'])} ({100*len(self.status['failed'])/self.status['total_simulations']:.1f}%)
Pending: {len(self.status['pending'])} ({100*len(self.status['pending'])/self.status['total_simulations']:.1f}%)
Running: {len(self.status['running'])}

Last update: {self.status['last_update']}
"""
        
        if self.status['failed']:
            report += f"\nFailed simulations:\n"
            for sim in self.status['failed']:
                report += f"  - {sim}\n"
        
        if self.status['running']:
            report += f"\nCurrently running:\n"
            for sim in self.status['running']:
                report += f"  - {sim}\n"
        
        return report


def main():
    """Command line interface for campaign runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simulation campaigns')
    parser.add_argument('campaign', help='Campaign name or directory')
    parser.add_argument('--stage', type=int, default=4, choices=[1,2,3,4],
                       help='Execution stage (1-4)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run simulations in parallel')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--no-retry', action='store_true',
                       help='Do not retry failed simulations')
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop campaign on first failure')
    parser.add_argument('--status', action='store_true',
                       help='Show campaign status only')
    
    args = parser.parse_args()
    
    try:
        runner = CampaignRunner(args.campaign)
        
        if args.status:
            # Just show status
            print(runner.get_status_report())
            return 0
        
        # Configure runner
        runner.stage = args.stage
        runner.retry_failed = not args.no_retry
        runner.continue_on_error = not args.stop_on_error
        
        # Run campaign
        results = runner.run_campaign(
            parallel=args.parallel,
            max_workers=args.workers
        )
        
        # Return code based on failures
        if len(runner.status['failed']) == 0:
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
    