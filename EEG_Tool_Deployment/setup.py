from setuptools import setup, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from multiprocessing import freeze_support
import numpy
import os
import platform
import glob

# Detect OS and architecture
IS_WINDOWS = platform.system() == "Windows"
IS_64BIT = platform.architecture()[0] == "64bit"

# Optimization flags
if IS_WINDOWS:
    EXTRA_COMPILE_ARGS = ["/O2", "/openmp", "/favor:AMD64" if IS_64BIT else ""]
    EXTRA_LINK_ARGS = []
else:
    EXTRA_COMPILE_ARGS = ["-O3", "-fopenmp", "-march=native", "-ffast-math"]
    EXTRA_LINK_ARGS = ["-fopenmp"]

# Get version from __init__.py
def get_version():
    with open(os.path.join("src", "__init__.py"), "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read requirements from requirements.txt
def get_requirements():
    requirements = []
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open("requirements.txt", "r", encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Remove any inline comments and version specifiers
                        line = line.split("#")[0].strip()
                        if line:  # Only append non-empty lines
                            requirements.append(line)
            break  # If successful, break the loop
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            # If requirements.txt doesn't exist, return default requirements
            return [
                
            ]
    
    if not requirements:  # If no encoding worked
        raise RuntimeError("Could not read requirements.txt with any supported encoding")
    
    return requirements

def get_extension_modules():
    python_files = glob.glob("src/**/*.py", recursive=True)
    print("Python files to be compiled:")
    extensions = []
    
    for file in python_files:
        module_name = file.replace(os.path.sep, '.')[:-3]
        print(f"  - {file} -> {module_name}")
        
        # Set specific compiler directives based on file
        compiler_directives = {
            "language_level": "3",
            "binding": True,
            "embedsignature": True,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
            "profile": False,
        }
        
        
        extensions.append(file)
    
    return extensions

def setup_package():
    extensions = get_extension_modules()
    
    if not extensions:
        print("ERROR: No Python files found to compile!")
        return

    try:
        setup(
            name="NeuroSync",
            version=get_version(),
            author="Sriram NC",
            author_email="sriramnallani35@gmail.com",
            description="Advanced EEG Analysis and Visualization Tool - Merging the brain and AI",
            
            
            # Package configuration
            packages=find_packages(include=["src", "src.*"]),
            package_data={
                "src": ["*.pyx", "*.pxd", "*.h", "*.c", "data/*", "sample_data/*", "*.pkl"]
            },
            
            # Cython configuration
            ext_modules=cythonize(
                extensions,
                compiler_directives={
                    "language_level": "3",
                    "binding": True,
                    "embedsignature": True,
                    "initializedcheck": False,
                    "nonecheck": False,
                    "cdivision": True,
                    "profile": False,
                    # Let individual files override these
                    "wraparound": True,
                    "boundscheck": True,
                },
                compile_time_env={
                    "OPENMP": True,
                    "SSE": True,
                    "AVX": True,
                    "FAST_MATH": True,
                },
                nthreads=os.cpu_count(),
                force=True,
            ),
            
            # Build configuration
            cmdclass={
                "build_ext": build_ext,
            },
            
            # Dependencies from requirements.txt
            install_requires=get_requirements(),
            
            # Python requirements
            python_requires=">=3.12",
            
            # Compilation settings
            include_dirs=[
                numpy.get_include(),
                "src/include",
            ],
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=EXTRA_LINK_ARGS,
            
            # Entry points
            entry_points={
                "console_scripts": [
                    "neurosync=src.launcher:main",
                ],
            },
            
            # Classifiers for PyPI
            classifiers=[
                "Development Status :: 4 - Beta",
                "Intended Audience :: Science/Research",
                "Topic :: Scientific/Engineering :: Bio-Informatics",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3.12",
                "Programming Language :: Cython",
                "Operating System :: OS Independent",
                "Environment :: X11 Applications :: Qt",
            ],
            
            # Additional metadata
            keywords="eeg, neuroscience, brain-computer interface, machine learning, artificial intelligence, neural networks",
            
            # Build options
            zip_safe=False,
            platforms=["any"],
        )
        print("\nCompilation completed successfully!")
        
        # Verify compiled files
        compiled_files = glob.glob("src/**/*.pyd", recursive=True)
        print("\nCompiled files:")
        for file in compiled_files:
            print(f"  - {file}")
            
    except Exception as e:
        print(f"\nERROR during compilation: {str(e)}")
        raise

if __name__ == '__main__':
    freeze_support()
    setup_package() 