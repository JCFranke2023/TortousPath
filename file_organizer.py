"""
ABAQUS file organization utility for PECVD barrier coating simulation
Creates organized directory structure and manages file cleanup
"""

import os
import shutil
import glob
from pathlib import Path

class ABQFileOrganizer:
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.abq_dir = self.base_dir / 'abaqus_files'
        
        # ABAQUS file extensions to manage
        self.abq_extensions = [
            '.odb',     # Output database
            '.sta',     # Status file
            '.msg',     # Message file
            '.dat',     # Data file
            '.fil',     # Results file
            '.res',     # Restart file
            '.prt',     # Part file
            '.com',     # Complete ABAQUS/CAE command
            '.jnl',     # Journal file
            '.rec',     # Recovery file
            '.mdl',     # Model file
            '.stt',     # Status file
            '.023',     # Temporary files
            '.ipm',     # Part manager file
            '.lck',     # Lock files
            '.log',     # Log files (ABAQUS specific)
            '.abq',     # ABAQUS files
            '.inp',     # Input files
            '.cae',     # CAE database
            '.rpy',     # Replay files
            '.env',     # Environment files
        ]
    
    def create_directory_structure(self):
        """Create organized directory structure for ABAQUS files"""
        directories = [
            self.abq_dir,
            self.abq_dir / 'jobs',          # Individual job directories
            self.abq_dir / 'models',        # CAE models
            self.abq_dir / 'results',       # ODB and result files
            self.abq_dir / 'temp',          # Temporary files
            self.abq_dir / 'logs',          # Log and status files
            self.abq_dir / 'archive'        # Completed job archives
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def move_abaqus_files(self, job_name=None):
        """Move ABAQUS files to organized structure"""
        moved_files = []
        
        # Find all ABAQUS files in current directory
        for extension in self.abq_extensions:
            pattern = f"*{extension}"
            for file_path in self.base_dir.glob(pattern):
                if file_path.is_file():
                    moved_files.append(self._move_file_by_type(file_path, job_name))
        
        # Clean up any remaining ABAQUS temporary files
        self._cleanup_temp_files()
        
        return [f for f in moved_files if f is not None]
    
    def _move_file_by_type(self, file_path, job_name=None):
        """Move file to appropriate subdirectory based on extension"""
        filename = file_path.name
        extension = file_path.suffix.lower()
        
        # Determine destination directory
        if extension in ['.odb', '.fil', '.dat']:
            dest_dir = self.abq_dir / 'results'
        elif extension in ['.cae', '.mdl']:
            dest_dir = self.abq_dir / 'models'
        elif extension in ['.sta', '.msg', '.log', '.prt']:
            dest_dir = self.abq_dir / 'logs'
        elif extension in ['.023', '.lck', '.ipm']:
            dest_dir = self.abq_dir / 'temp'
        else:
            dest_dir = self.abq_dir / 'jobs'
        
        # Create job-specific subdirectory if job_name provided
        if job_name and extension in ['.odb', '.sta', '.msg', '.dat', '.fil']:
            dest_dir = dest_dir / job_name
            dest_dir.mkdir(exist_ok=True)
        
        # Move file
        dest_path = dest_dir / filename
        try:
            shutil.move(str(file_path), str(dest_path))
            print(f"Moved: {filename} -> {dest_path.relative_to(self.base_dir)}")
            return dest_path
        except Exception as e:
            print(f"Error moving {filename}: {e}")
            return None
    
    def _cleanup_temp_files(self):
        """Remove ABAQUS temporary files"""
        temp_patterns = [
            'abaqus.rpy*',
            '*.023',
            '*.lck',
            'aba_*'
        ]
        
        for pattern in temp_patterns:
            for file_path in self.base_dir.glob(pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        print(f"Removed temp file: {file_path.name}")
                    except Exception as e:
                        print(f"Could not remove {file_path.name}: {e}")
    
    def organize_job(self, job_name):
        """Organize files for a specific job"""
        print(f"Organizing files for job: {job_name}")
        
        # Create job-specific directory
        job_dir = self.abq_dir / 'jobs' / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Move job-specific files
        moved_files = []
        for extension in self.abq_extensions:
            pattern = f"{job_name}*{extension}"
            for file_path in self.base_dir.glob(pattern):
                if file_path.is_file():
                    dest_path = job_dir / file_path.name
                    try:
                        shutil.move(str(file_path), str(dest_path))
                        moved_files.append(dest_path)
                        print(f"Moved: {file_path.name} -> {dest_path.relative_to(self.base_dir)}")
                    except Exception as e:
                        print(f"Error moving {file_path.name}: {e}")
        
        return moved_files
    
    def archive_completed_job(self, job_name):
        """Archive a completed job to reduce clutter"""
        job_dir = self.abq_dir / 'jobs' / job_name
        archive_dir = self.abq_dir / 'archive' / job_name
        
        if job_dir.exists():
            try:
                shutil.move(str(job_dir), str(archive_dir))
                print(f"Archived job: {job_name}")
                return archive_dir
            except Exception as e:
                print(f"Error archiving {job_name}: {e}")
                return None
        else:
            print(f"Job directory not found: {job_dir}")
            return None
    
    def cleanup_all_abaqus_files(self):
        """Clean up all ABAQUS files in current directory"""
        print("Cleaning up all ABAQUS files...")
        self.create_directory_structure()
        moved_files = self.move_abaqus_files()
        print(f"Moved {len(moved_files)} files to organized structure")
        return moved_files
    
    def get_abaqus_files_in_current_dir(self):
        """Get list of ABAQUS files in current directory"""
        files = []
        for extension in self.abq_extensions:
            pattern = f"*{extension}"
            files.extend(list(self.base_dir.glob(pattern)))
        return files

# Usage functions
def setup_abaqus_organization():
    """Setup ABAQUS file organization structure"""
    organizer = ABQFileOrganizer()
    organizer.create_directory_structure()
    return organizer

def organize_current_job(job_name):
    """Organize files for current job"""
    organizer = ABQFileOrganizer()
    organizer.create_directory_structure()
    return organizer.organize_job(job_name)

def cleanup_abaqus_files():
    """Clean up all ABAQUS files in current directory"""
    organizer = ABQFileOrganizer()
    return organizer.cleanup_all_abaqus_files()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize ABAQUS files')
    parser.add_argument('--setup', action='store_true', help='Setup directory structure')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup all ABAQUS files')
    parser.add_argument('--job', help='Organize specific job files')
    parser.add_argument('--archive', help='Archive completed job')
    
    args = parser.parse_args()
    
    organizer = ABQFileOrganizer()
    
    if args.setup:
        organizer.create_directory_structure()
    elif args.cleanup:
        organizer.cleanup_all_abaqus_files()
    elif args.job:
        organizer.organize_job(args.job)
    elif args.archive:
        organizer.archive_completed_job(args.archive)
    else:
        # Default: setup and cleanup
        organizer.cleanup_all_abaqus_files()
        