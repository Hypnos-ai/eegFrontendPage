import subprocess
import re
from pathlib import Path

def read_requirements(file_path):
    """Read requirements.txt and extract package names without version numbers."""
    with open(file_path, 'r') as file:
        # Read lines and remove empty lines
        lines = [line.strip() for line in file if line.strip()]
        
        # Extract package names (remove version numbers)
        packages = [re.split('[=<>]', line)[0] for line in lines]
        
        return packages

def get_package_version(package):
    """Get the installed version of a package using pip show."""
    try:
        result = subprocess.run(['pip', 'show', package], 
                             capture_output=True, 
                             text=True)
        if result.returncode == 0:
            # Extract version from pip show output
            for line in result.stdout.split('\n'):
                if line.startswith('Version: '):
                    return line.split('Version: ')[1].strip()
        return None
    except Exception as e:
        print(f"Error getting version for {package}: {str(e)}")
        return None

def update_requirements_file(file_path):
    """Update requirements.txt with actual installed versions."""
    packages = read_requirements(file_path)
    updated_requirements = []
    duplicates = set()
    
    print("Checking installed packages and updating versions...")
    print("=" * 80)
    
    for package in packages:
        # Skip if we've already processed this package
        if package in duplicates:
            continue
        
        version = get_package_version(package)
        if version:
            requirement = f"{package}=={version}"
            updated_requirements.append(requirement)
            print(f"Found {requirement}")
        else:
            print(f"Warning: Could not determine version for {package}")
            updated_requirements.append(package)
        
        duplicates.add(package)
    
    print("\nWriting updated requirements to file...")
    
    # Create backup of original file
    backup_path = Path(file_path).with_suffix('.txt.backup')
    Path(file_path).rename(backup_path)
    
    # Write updated requirements
    with open(file_path, 'w') as file:
        file.write('\n'.join(sorted(updated_requirements)) + '\n')
    
    print(f"Original file backed up to: {backup_path}")
    print(f"Updated requirements written to: {file_path}")
    print("=" * 80)

def main():
    requirements_path = 'eeg_tool/requirements.txt'
    update_requirements_file(requirements_path)

if __name__ == "__main__":
    main() 