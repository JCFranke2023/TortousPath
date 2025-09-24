#!/usr/bin/env python
"""
ABAQUS file management utility - MVP version
Simple, focused file organization for simulation outputs
"""

import os
import sys
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional


class SimulationFileManager:
    """Minimal file manager for ABAQUS simulation outputs"""
    
    def __init__(self, simulation_name: str, base_dir: str = 'simulations'):
        """
        Initialize file manager
        
        Args:
            simulation_name: Name of the simulation
            base_dir: Base directory for simulations
        """
        self.simulation_name = simulation_name
        self.sim_dir = Path(base_dir) / simulation_name
        
        # Simple file mappings
        self.file_mappings = {
            'models': ['.cae', '.odb'],
            'abaqus_files/inputs': ['.inp'],
            'abaqus_files/logs': ['.msg', '.sta', '.dat', '.prt', '.log'],
            'abaqus_files/scripts': ['.com', '.jnl', '.rpy'],
            'abaqus_files/temp': ['.023', '.lck', '.ipm', '.rec', '.sim']
        }
    
    def create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.sim_dir / 'models',
            self.sim_dir / 'abaqus_files' / 'inputs',
            self.sim_dir / 'abaqus_files' / 'logs',
            self.sim_dir / 'abaqus_files' / 'scripts',
            self.sim_dir / 'abaqus_files' / 'temp',
            self.sim_dir / 'extracted_data' / 'nnc_gradients',
            self.sim_dir / 'extracted_data' / 'raw_json',
            self.sim_dir / 'analysis' / 'flux_data',
            self.sim_dir / 'analysis' / 'metrics',
            self.sim_dir / 'analysis' / 'plots',
            self.sim_dir / 'logs'
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def organize_initial(self, job_name: str, cleanup_temp: bool = True) -> Dict:
        """
        Initial organization - move non-critical files
        
        Args:
            job_name: ABAQUS job name
            cleanup_temp: Whether to delete temp files
        
        Returns:
            Dict with counts
        """
        print(f"\nOrganizing initial files for {job_name}...")
        self.create_directories()
        
        counts = {'moved': 0, 'deleted': 0, 'skipped': 0}
        
        # Files to skip in initial pass (handle these in final pass)
        skip_extensions = ['.odb', '.cae', '.log', '.com', '.jnl', '.rpy']
        
        # Search in current directory
        for dest_dir, extensions in self.file_mappings.items():
            dest_path = self.sim_dir / dest_dir
            
            for ext in extensions:
                # Skip critical files for now
                if ext in skip_extensions:
                    continue
                
                # Find files
                for file_path in Path('.').glob(f"{job_name}*{ext}"):
                    if not file_path.is_file():
                        continue
                    
                    # Handle temp files
                    if 'temp' in dest_dir and cleanup_temp:
                        try:
                            file_path.unlink()
                            counts['deleted'] += 1
                            print(f"  Deleted: {file_path.name}")
                        except:
                            pass
                    else:
                        # Move file
                        try:
                            target = dest_path / file_path.name
                            shutil.move(str(file_path), str(target))
                            counts['moved'] += 1
                            print(f"  Moved: {file_path.name} -> {dest_dir}")
                        except Exception as e:
                            counts['skipped'] += 1
                            print(f"  Skipped {file_path.name}: {e}")
        
        print(f"Initial: moved {counts['moved']}, deleted {counts['deleted']}, skipped {counts['skipped']}")
        return counts
    
    def organize_final(self, job_name: str) -> Dict:
        """
        Final organization - move critical files that might have been locked
        Called after all processing is complete
        
        Args:
            job_name: ABAQUS job name
        
        Returns:
            Dict with counts
        """
        print(f"\nFinal file organization for {job_name}...")
        
        counts = {'moved': 0, 'failed': 0}
        
        # Critical files to move with retries
        critical_files = {
            '.odb': 'models',
            '.cae': 'models',
            '.log': 'abaqus_files/logs',
            '.com': 'abaqus_files/scripts',
            '.jnl': 'abaqus_files/scripts',
            '.rpy': 'abaqus_files/scripts'
        }
        
        for ext, dest_dir in critical_files.items():
            # Look for exact file name
            file_path = Path(f"{job_name}{ext}")
            
            if file_path.exists() and file_path.is_file():
                dest_path = self.sim_dir / dest_dir / file_path.name
                
                # Try to move with retries
                moved = self._move_with_retry(file_path, dest_path)
                
                if moved:
                    counts['moved'] += 1
                    print(f"  Moved: {file_path.name} -> {dest_dir}")
                else:
                    counts['failed'] += 1
                    print(f"  Failed to move: {file_path.name}")
        
        # Also check for abaqus.rpy
        if Path('abaqus.rpy').exists():
            dest_path = self.sim_dir / 'abaqus_files' / 'scripts' / 'abaqus.rpy'
            if self._move_with_retry(Path('abaqus.rpy'), dest_path):
                counts['moved'] += 1
                print(f"  Moved: abaqus.rpy -> abaqus_files/scripts")
        
        print(f"Final: moved {counts['moved']}, failed {counts['failed']}")
        
        if counts['failed'] > 0:
            print("\nWARNING: Some files could not be moved. They may still be in use.")
        
        return counts
    
    def _move_with_retry(self, source: Path, dest: Path, max_retries: int = 3) -> bool:
        """
        Try to move a file with retries for locked files
        
        Args:
            source: Source file path
            dest: Destination file path
            max_retries: Number of retry attempts
        
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Check if already at destination
                if dest.exists():
                    if dest.samefile(source):
                        return True  # Already there
                    dest.unlink()  # Remove old version
                
                # Try to move
                shutil.move(str(source), str(dest))
                return True
                
            except PermissionError:
                # File is locked, wait and retry
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                    continue
                return False
                
            except FileNotFoundError:
                # File doesn't exist
                return True  # Consider it success (nothing to move)
                
            except Exception:
                return False
        
        return False
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary files in workspace
        
        Returns:
            Number of files deleted
        """
        print("\nCleaning up temporary files...")
        
        temp_patterns = [
            '*.023', '*.lck', '*.rec', '*.ipm', '*.sim',
            'abaqus.rpy.*', 'aba_*.tmp', '*.exception'
        ]
        
        deleted = 0
        for pattern in temp_patterns:
            for temp_file in Path('.').glob(pattern):
                try:
                    temp_file.unlink()
                    print(f"  Deleted: {temp_file.name}")
                    deleted += 1
                except:
                    pass
        
        print(f"Deleted {deleted} temporary files")
        return deleted
    
    def verify_files(self, job_name: str) -> Dict:
        """
        Verify that critical files are in place
        
        Args:
            job_name: Job name to check
        
        Returns:
            Dict with file locations
        """
        locations = {}
        
        # Critical files to check
        files_to_check = {
            'odb': self.sim_dir / 'models' / f"{job_name}.odb",
            'cae': self.sim_dir / 'models' / f"{job_name}.cae",
            'inp': self.sim_dir / 'abaqus_files' / 'inputs' / f"{job_name}.inp",
            'log': self.sim_dir / 'abaqus_files' / 'logs' / f"{job_name}.log",
            'com': self.sim_dir / 'abaqus_files' / 'scripts' / f"{job_name}.com",
            'jnl': self.sim_dir / 'abaqus_files' / 'scripts' / f"{job_name}.jnl"
        }
        
        for file_type, path in files_to_check.items():
            if path.exists():
                locations[file_type] = str(path)
            else:
                # Check if still in root
                root_path = Path(f"{job_name}.{file_type}")
                if root_path.exists():
                    locations[file_type] = f"STILL IN ROOT: {root_path}"
                else:
                    locations[file_type] = "NOT FOUND"
        
        return locations


def organize_initial(simulation_name: str, job_name: str, cleanup_temp: bool = True) -> Dict:
    """
    Convenience function for initial organization
    
    Args:
        simulation_name: Simulation name
        job_name: Job name
        cleanup_temp: Whether to delete temp files
    
    Returns:
        Organization counts
    """
    manager = SimulationFileManager(simulation_name)
    return manager.organize_initial(job_name, cleanup_temp)


def organize_final(simulation_name: str, job_name: str) -> Dict:
    """
    Convenience function for final organization
    
    Args:
        simulation_name: Simulation name
        job_name: Job name
    
    Returns:
        Organization counts
    """
    manager = SimulationFileManager(simulation_name)
    return manager.organize_final(job_name)


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ABAQUS file management (MVP)')
    parser.add_argument('command', 
                       choices=['initial', 'final', 'cleanup', 'verify'],
                       help='Command to execute')
    parser.add_argument('--simulation', '-s', required=True,
                       help='Simulation name')
    parser.add_argument('--job', '-j',
                       help='Job name')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep temporary files')
    
    args = parser.parse_args()
    
    manager = SimulationFileManager(args.simulation)
    
    if args.command == 'initial':
        if not args.job:
            print("ERROR: --job required for initial command")
            return 1
        manager.organize_initial(args.job, not args.no_cleanup)
        
    elif args.command == 'final':
        if not args.job:
            print("ERROR: --job required for final command")
            return 1
        manager.organize_final(args.job)
        
    elif args.command == 'cleanup':
        manager.cleanup_temp_files()
        
    elif args.command == 'verify':
        if not args.job:
            print("ERROR: --job required for verify command")
            return 1
        locations = manager.verify_files(args.job)
        print("\nFile locations:")
        for file_type, location in locations.items():
            print(f"  {file_type}: {location}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
    