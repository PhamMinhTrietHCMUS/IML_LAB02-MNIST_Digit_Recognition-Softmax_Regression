"""
Test script to verify the installation and data availability
"""
import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import numpy as np
        print("✓ NumPy installed:", np.__version__)
    except ImportError:
        print("✗ NumPy not found. Run: pip install numpy")
        return False
    
    try:
        import flask
        print("✓ Flask installed:", flask.__version__)
    except ImportError:
        print("✗ Flask not found. Run: pip install Flask")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow installed")
    except ImportError:
        print("✗ Pillow not found. Run: pip install Pillow")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib installed:", matplotlib.__version__)
    except ImportError:
        print("✗ Matplotlib not found. Run: pip install matplotlib")
        return False
    
    return True


def test_data_files():
    """Test if MNIST data files exist"""
    print("\nTesting MNIST data files...")
    data_dir = 'archive_2'
    required_files = [
        'train-images.idx3-ubyte',
        'train-labels.idx1-ubyte',
        't10k-images.idx3-ubyte',
        't10k-labels.idx1-ubyte'
    ]
    
    all_found = True
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[OK] {filename} found ({size_mb:.2f} MB)")
        else:
            print(f"[FAIL] {filename} not found")
            all_found = False
    
    return all_found


def test_directory_structure():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    # Check src directory
    if os.path.exists('src'):
        print("✓ src/ directory exists")
        
        # Check src files
        src_files = ['softmax_regression.py', 'data_loader.py', 'train.py']
        for filename in src_files:
            filepath = os.path.join('src', filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename} exists")
            else:
                print(f"  ✗ {filename} missing")
    else:
        print("✗ src/ directory not found")
    
    # Check templates directory
    if os.path.exists('templates'):
        print("✓ templates/ directory exists")
        if os.path.exists('templates/index.html'):
            print("  ✓ index.html exists")
        else:
            print("  ✗ index.html missing")
    else:
        print("✗ templates/ directory not found")
    
    # Check static directory
    if os.path.exists('static'):
        print("✓ static/ directory exists")
        static_files = ['style.css', 'script.js']
        for filename in static_files:
            filepath = os.path.join('static', filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename} exists")
            else:
                print(f"  ✗ {filename} missing")
    else:
        print("✗ static/ directory not found")
    
    # Check app.py
    if os.path.exists('app.py'):
        print("✓ app.py exists")
    else:
        print("✗ app.py missing")
    
    return True


def test_load_data():
    """Test if data can be loaded"""
    print("\nTesting data loading...")
    try:
        sys.path.append('src')
        from data_loader import load_mnist_data
        
        X_train, y_train, X_test, y_test = load_mnist_data('archive_2')
        print(f"✓ Data loaded successfully")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Image shape: {X_train.shape[1]}x{X_train.shape[2]}")
        return True
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("MNIST Softmax Regression - System Check")
    print("=" * 60)
    
    # Run tests
    imports_ok = test_imports()
    data_ok = test_data_files()
    structure_ok = test_directory_structure()
    
    print("\n" + "=" * 60)
    
    if imports_ok and data_ok and structure_ok:
        print("✓ All basic checks passed!")
        print("\nTrying to load MNIST data...")
        load_ok = test_load_data()
        
        if load_ok:
            print("\n" + "=" * 60)
            print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
            print("=" * 60)
            print("\nYou're ready to go! Next steps:")
            print("1. Train the model: python src/train.py")
            print("2. Run the web app: python app.py")
            print("=" * 60)
        else:
            print("\n⚠ Data loading failed. Check error messages above.")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nMissing packages? Run: pip install -r requirements.txt")
    
    print()


if __name__ == '__main__':
    main()
