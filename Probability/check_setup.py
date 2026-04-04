"""
Quick test script to verify installation
Run this to check if all dependencies are properly installed
"""

import sys

def check_imports():
    """Check if all required packages can be imported"""
    
    print("Checking Python version...")
    print(f"Python {sys.version}")
    print()
    
    packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn'
    }
    
    print("Checking package installations:")
    print("-" * 50)
    
    all_good = True
    for package, name in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = module.__version__
            
            print(f"✓ {name:<20} {version}")
        except ImportError:
            print(f"✗ {name:<20} NOT INSTALLED")
            all_good = False
    
    print("-" * 50)
    
    if all_good:
        print("\n✓ All packages installed successfully!")
        print("\nYou can now run the examples:")
        print("  python run_examples.py")
        print("\nOr run individual scripts:")
        print("  python examples/distributions.py")
        print("  python examples/conditional_probability.py")
        print("  python examples/bias_variance.py")
    else:
        print("\n✗ Some packages are missing.")
        print("\nPlease run: pip install -r requirements.txt")
    
    return all_good


if __name__ == "__main__":
    print("=" * 50)
    print("  PROBABILITY PROJECT - INSTALLATION CHECK")
    print("=" * 50)
    print()
    
    success = check_imports()
    
    print()
    print("=" * 50)
    
    sys.exit(0 if success else 1)
