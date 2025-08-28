#!/usr/bin/env python
"""
ABAQUS file management utility
Quick script to organize ABAQUS files and clean up the workspace
"""

import sys
from pathlib import Path
from file_organizer import ABQFileOrganizer

def main():
    organizer = ABQFileOrganizer()
    
    if len(sys.argv) == 1:
        # Default: setup and cleanup
        print("Setting up ABAQUS file organization and cleaning up current directory...")
        files = organizer.cleanup_all_abaqus_files()
        print(f"Organized {len(files)} ABAQUS files")
        
    elif sys.argv[1] == "setup":
        print("Setting up ABAQUS file organization structure...")
        organizer.create_directory_structure()
        print("Directory structure created")
        
    elif sys.argv[1] == "cleanup":
        print("Cleaning up all ABAQUS files in current directory...")
        files = organizer.cleanup_all_abaqus_files()
        print(f"Organized {len(files)} ABAQUS files")
        
    elif sys.argv[1] == "status":
        print("Current ABAQUS files in directory:")
        files = organizer.get_abaqus_files_in_current_dir()
        if files:
            for f in files:
                print(f"  {f.name}")
        else:
            print("  No ABAQUS files found")
        
    elif sys.argv[1] == "job" and len(sys.argv) > 2:
        job_name = sys.argv[2]
        print(f"Organizing files for job: {job_name}")
        files = organizer.organize_job(job_name)
        print(f"Organized {len(files)} files for job {job_name}")
        
    elif sys.argv[1] == "archive" and len(sys.argv) > 2:
        job_name = sys.argv[2]
        print(f"Archiving job: {job_name}")
        archive_path = organizer.archive_completed_job(job_name)
        if archive_path:
            print(f"Job archived to: {archive_path}")
        
    else:
        print("Usage:")
        print("  python manage_files.py                 # Setup and cleanup all files")
        print("  python manage_files.py setup          # Create directory structure")
        print("  python manage_files.py cleanup        # Organize all ABAQUS files")
        print("  python manage_files.py status         # Show current ABAQUS files")
        print("  python manage_files.py job <name>     # Organize specific job")
        print("  python manage_files.py archive <name> # Archive completed job")

if __name__ == "__main__":
    main()
    