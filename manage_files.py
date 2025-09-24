#!/usr/bin/env python
"""
ABAQUS file management utility - Simplified interface for batch operations
Works with the new simulation-specific directory structure
"""

import sys
import json
from pathlib import Path
from datetime import datetime


class ABQFileOrganizer:
    def __init__(self, simulation_dir=None, job_name=None):
        """
        Initialize file organizer for new directory structure
        
        Args:
            simulation_dir: Path to simulation directory (e.g., simulations/sim_name/)
            job_name: ABAQUS job name for this simulation
        """
        if simulation_dir:
            self.sim_dir = Path(simulation_dir)
        else:
            self.sim_dir = Path('.')
        
        self.job_name = job_name
        
        # Define file categories and destinations
        self.file_mappings = {
            'models': ['.cae', '.odb'],
            'abaqus_files/inputs': ['.inp'],
            'abaqus_files/logs': ['.msg', '.sta', '.dat', '.prt', '.log'],
            'abaqus_files/temp': ['.023', '.lck', '.ipm', '.rec', '.res', 
                                  '.mdl', '.stt', '.com', '.jnl', '.rpy']
        }
        
        # Files to keep in both locations (copy instead of move)
        self.preserve_files = []
    
    def create_directory_structure(self):
        """Create organized directory structure for simulation"""
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
            self.sim_dir / 'logs'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        return directories
    
    def organize_job_files(self, source_dir='.', cleanup_temp=True):
        """
        Organize ABAQUS files for a specific job into proper structure
        
        Args:
            source_dir: Directory containing ABAQUS files
            cleanup_temp: Whether to delete temporary files
        
        Returns:
            Dict with counts of files organized by category
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Source directory not found: {source_path}")
            return {}
        
        organized = {
            'models': 0,
            'inputs': 0,
            'logs': 0,
            'temp': 0,
            'deleted': 0
        }
        
        # Process each file category
        for dest_subdir, extensions in self.file_mappings.items():
            dest_dir = self.sim_dir / dest_subdir
            
            for ext in extensions:
                # Find files with this extension
                if self.job_name:
                    pattern = f"{self.job_name}*{ext}"
                else:
                    pattern = f"*{ext}"
                
                for file_path in source_path.glob(pattern):
                    if not file_path.is_file():
                        continue
                    
                    dest_path = dest_dir / file_path.name
                    
                    try:
                        # Determine category for counting
                        if 'models' in dest_subdir:
                            category = 'models'
                        elif 'inputs' in dest_subdir:
                            category = 'inputs'
                        elif 'logs' in dest_subdir:
                            category = 'logs'
                        else:
                            category = 'temp'
                        
                        # Handle temp files
                        if category == 'temp' and cleanup_temp:
                            file_path.unlink()
                            organized['deleted'] += 1
                            print(f"Deleted temp file: {file_path.name}")
                        
                        # Preserve important files (copy instead of move)
                        elif ext in self.preserve_files:
                            import shutil
                            shutil.copy2(str(file_path), str(dest_path))
                            organized[category] += 1
                            print(f"Copied: {file_path.name} -> {dest_path.relative_to(self.sim_dir)}")
                        
                        # Move other files
                        else:
                            file_path.rename(dest_path)
                            organized[category] += 1
                            print(f"Moved: {file_path.name} -> {dest_path.relative_to(self.sim_dir)}")
                    
                    except Exception as e:
                        print(f"Error processing {file_path.name}: {e}")
        
        # Also check for ABAQUS default directories
        self._process_abaqus_subdirs(source_path, organized)
        
        # Clean up remaining temp files
        if cleanup_temp:
            self._cleanup_temp_patterns(source_path, organized)
        
        return organized
    
    def _process_abaqus_subdirs(self, source_path, organized):
        """Process files in ABAQUS-created subdirectories"""
        # Check common ABAQUS output locations
        abaqus_dirs = [
            source_path / 'abaqus_files' / 'jobs' / self.job_name if self.job_name else None,
            source_path / 'abaqus_files' / 'results',
            source_path / 'abaqus_files' / 'models'
        ]
        
        for abq_dir in abaqus_dirs:
            if abq_dir and abq_dir.exists():
                # Recursively organize files from these directories
                self.organize_job_files(abq_dir, cleanup_temp=False)
    
    def _cleanup_temp_patterns(self, source_path, organized):
        """Clean up temporary file patterns"""
        temp_patterns = [
            'abaqus.rpy*',
            'aba_*.tmp',
            '*.rec.*',
            '*.exception'
        ]
        
        for pattern in temp_patterns:
            for temp_file in source_path.glob(pattern):
                try:
                    temp_file.unlink()
                    organized['deleted'] += 1
                    print(f"Deleted: {temp_file.name}")
                except Exception as e:
                    print(f"Could not delete {temp_file.name}: {e}")
    
    def get_file_summary(self):
        """Get summary of files in organized structure"""
        summary = {}
        
        # Count files in each category
        for category in ['models', 'abaqus_files/inputs', 'abaqus_files/logs']:
            dir_path = self.sim_dir / category
            if dir_path.exists():
                files = list(dir_path.glob(f"{self.job_name}*" if self.job_name else "*"))
                summary[category] = {
                    'count': len(files),
                    'files': [f.name for f in files if f.is_file()]
                }
        
        return summary
    
    def find_odb_file(self):
        """Find and return path to ODB file"""
        # Check primary location
        odb_path = self.sim_dir / 'models' / f"{self.job_name}.odb"
        if odb_path.exists():
            return odb_path
        
        # Check alternative locations
        alt_locations = [
            Path('.') / f"{self.job_name}.odb",
            Path('abaqus_files') / 'jobs' / self.job_name / f"{self.job_name}.odb",
            Path('abaqus_files') / 'results' / f"{self.job_name}.odb"
        ]
        
        for alt_path in alt_locations:
            if alt_path.exists():
                return alt_path
        
        return None
    
    def find_cae_file(self):
        """Find and return path to CAE file"""
        # Check primary location
        cae_path = self.sim_dir / 'models' / f"{self.job_name}.cae"
        if cae_path.exists():
            return cae_path
        
        # Check current directory
        cae_path = Path('.') / f"{self.job_name}.cae"
        if cae_path.exists():
            return cae_path
        
        return None


class SimulationFileManager:
    """Manage files for simulation workflow"""
    
    def __init__(self, simulation_name=None):
        """
        Initialize file manager
        
        Args:
            simulation_name: Name of the simulation
        """
        self.simulation_name = simulation_name
        if simulation_name:
            self.sim_dir = Path('simulations') / simulation_name
        else:
            self.sim_dir = Path('.')
    
    def organize_job_files(self, job_name, source_dir='.', cleanup_temp=True):
        """
        Organize files for a specific job
        
        Args:
            job_name: ABAQUS job name
            source_dir: Source directory containing files
            cleanup_temp: Whether to delete temporary files
        
        Returns:
            Summary of organized files
        """
        print(f"\nOrganizing files for job: {job_name}")
        if self.simulation_name:
            print(f"Simulation: {self.simulation_name}")
        
        organizer = ABQFileOrganizer(self.sim_dir, job_name)
        organizer.create_directory_structure()
        
        # Organize files
        result = organizer.organize_job_files(source_dir, cleanup_temp)
        
        # Print summary
        print("\nFiles organized:")
        for category, count in result.items():
            if count > 0:
                print(f"  {category}: {count} files")
        
        return result
    
    def find_key_files(self, job_name):
        """
        Find ODB and CAE files for a job
        
        Returns:
            Dict with file paths
        """
        organizer = ABQFileOrganizer(self.sim_dir, job_name)
        
        files = {
            'odb': organizer.find_odb_file(),
            'cae': organizer.find_cae_file()
        }
        
        print(f"\nKey files for {job_name}:")
        if files['odb']:
            print(f"  ODB: {files['odb']}")
        else:
            print(f"  ODB: Not found")
        
        if files['cae']:
            print(f"  CAE: {files['cae']}")
        else:
            print(f"  CAE: Not found")
        
        return files
    
    def cleanup_workspace(self, aggressive=False):
        """
        Clean up workspace by removing temporary files
        
        Args:
            aggressive: If True, remove more file types
        """
        print("\nCleaning up workspace...")
        
        temp_patterns = [
            '*.023',
            '*.lck',
            'abaqus.rpy*',
            '*.rec',
            '*.ipm',
            '*.exception'
        ]
        
        if aggressive:
            temp_patterns.extend([
                '*.jnl',
                '*.com',
                '*.mdl',
                '*.stt'
            ])
        
        removed_count = 0
        for pattern in temp_patterns:
            for temp_file in Path('.').glob(pattern):
                try:
                    temp_file.unlink()
                    removed_count += 1
                    print(f"  Removed: {temp_file.name}")
                except Exception as e:
                    print(f"  Could not remove {temp_file.name}: {e}")
        
        print(f"Removed {removed_count} temporary files")
        return removed_count
    
    def get_simulation_status(self):
        """
        Get status of simulation files
        
        Returns:
            Status dictionary
        """
        if not self.sim_dir.exists():
            return {
                'exists': False,
                'simulation_name': self.simulation_name
            }
        
        status = {
            'exists': True,
            'simulation_name': self.simulation_name,
            'directories': {},
            'file_counts': {}
        }
        
        # Check directories
        dirs_to_check = [
            'models',
            'abaqus_files/inputs',
            'abaqus_files/logs',
            'extracted_data/nnc_gradients',
            'analysis/flux_data',
            'analysis/metrics',
            'analysis/plots',
            'summary'
        ]
        
        for dir_name in dirs_to_check:
            dir_path = self.sim_dir / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob('*')))
                status['directories'][dir_name] = True
                status['file_counts'][dir_name] = file_count
            else:
                status['directories'][dir_name] = False
                status['file_counts'][dir_name] = 0
        
        # Check for key files
        config_file = self.sim_dir / 'config.json'
        status['has_config'] = config_file.exists()
        
        if status['has_config']:
            with open(config_file, 'r') as f:
                config = json.load(f)
                status['job_name'] = config.get('job_name')
                status['parameters'] = config.get('parameters')
        
        return status
    
    def print_status(self):
        """Print formatted status of simulation"""
        status = self.get_simulation_status()
        
        print(f"\n{'='*50}")
        print(f"SIMULATION STATUS: {self.simulation_name}")
        print(f"{'='*50}")
        
        if not status['exists']:
            print("Simulation directory does not exist")
            return
        
        print("\nDirectories:")
        for dir_name, exists in status['directories'].items():
            count = status['file_counts'][dir_name]
            if exists:
                print(f"  ✓ {dir_name:<30} ({count} files)")
            else:
                print(f"  ✗ {dir_name:<30}")
        
        if status.get('has_config'):
            print(f"\nConfiguration found:")
            print(f"  Job name: {status.get('job_name')}")
            if status.get('parameters'):
                print(f"  Parameters:")
                for key, value in status['parameters'].items():
                    print(f"    {key}: {value}")
        
        print(f"{'='*50}")


def main():
    """Main entry point with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Manage ABAQUS simulation files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize files for a job in a simulation
  python manage_files.py organize --simulation my_sim --job Job_c100_s10000_o25
  
  # Find key files
  python manage_files.py find --simulation my_sim --job Job_c100_s10000_o25
  
  # Check simulation status
  python manage_files.py status --simulation my_sim
  
  # Clean up temporary files
  python manage_files.py cleanup
  
  # Clean up aggressively (removes more file types)
  python manage_files.py cleanup --aggressive
        """
    )
    
    parser.add_argument('command', 
                       choices=['organize', 'find', 'status', 'cleanup'],
                       help='Command to execute')
    parser.add_argument('--simulation', '-s',
                       help='Simulation name')
    parser.add_argument('--job', '-j',
                       help='Job name')
    parser.add_argument('--source', default='.',
                       help='Source directory for organize command')
    parser.add_argument('--no-cleanup-temp', action='store_true',
                       help='Keep temporary files when organizing')
    parser.add_argument('--aggressive', action='store_true',
                       help='Aggressive cleanup (removes more file types)')
    
    args = parser.parse_args()
    
    # Create manager
    manager = SimulationFileManager(args.simulation)
    
    # Execute command
    if args.command == 'organize':
        if not args.job:
            print("ERROR: --job required for organize command")
            return 1
        
        manager.organize_job_files(
            args.job,
            args.source,
            cleanup_temp=not args.no_cleanup_temp
        )
        
    elif args.command == 'find':
        if not args.job:
            print("ERROR: --job required for find command")
            return 1
        
        manager.find_key_files(args.job)
        
    elif args.command == 'status':
        if not args.simulation:
            print("ERROR: --simulation required for status command")
            return 1
        
        manager.print_status()
        
    elif args.command == 'cleanup':
        manager.cleanup_workspace(aggressive=args.aggressive)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
    