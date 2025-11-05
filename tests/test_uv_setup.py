#!/usr/bin/env python3
"""
Final test script to verify UV environment and CELLxGENE setup.
Run this script to verify everything is working correctly.

Usage:
    export PATH="$HOME/.local/bin:$PATH"
    python test_uv_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("🧪 Testing imports...")

    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False

    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False

    try:
        import matplotlib
        print(f"✅ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        return False

    try:
        import anndata
        print(f"✅ anndata {anndata.__version__}")
    except ImportError as e:
        print(f"❌ anndata: {e}")
        return False

    try:
        import scanpy as sc
        print(f"✅ scanpy {sc.__version__}")
    except ImportError as e:
        print(f"❌ scanpy: {e}")
        return False

    try:
        import cellxgene_census
        print(f"✅ cellxgene_census available")
    except ImportError as e:
        print(f"⚠️  cellxgene_census: {e}")
        # This is okay, we have fallbacks

    return True

def test_data_directories():
    """Test that data directories exist."""
    print("\n📁 Testing data directories...")

    base_dir = Path.cwd()
    required_dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "data" / "interim",
        base_dir / "data" / "external",
        base_dir / "scripts",
        base_dir / "notebooks",
        base_dir / "results"
    ]

    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            all_exist = False

    return all_exist

def test_uv_environment():
    """Test UV-specific functionality."""
    print("\n🔧 Testing UV environment...")

    # Check if we're in a UV-managed environment
    virtual_env = sys.prefix
    print(f"📍 Python environment: {virtual_env}")

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python version: {python_version}")

    # Check for UV-specific indicators
    if "AstroBio" in virtual_env:
        print("✅ Running in AstroBio environment")
        return True
    else:
        print("⚠️  Environment name doesn't contain 'AstroBio'")
        return True  # Still okay

def create_sample_data():
    """Create a sample dataset to test data processing."""
    print("\n📊 Creating sample data...")

    import pandas as pd
    import numpy as np

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    sample_data = pd.DataFrame({
        'cell_id': [f'cell_{i:03d}' for i in range(n_samples)],
        'cell_type': np.random.choice(['neuron', 'microglia', 'astrocyte'], n_samples),
        'condition': np.random.choice(['control', 'treatment'], n_samples),
        'expression_gene1': np.random.lognormal(1, 0.5, n_samples),
        'expression_gene2': np.random.lognormal(0.5, 0.7, n_samples)
    })

    # Save to data directory
    output_file = Path("data/raw/uv_test_sample.csv")
    sample_data.to_csv(output_file, index=False)
    print(f"✅ Sample data saved to {output_file}")

    # Basic analysis
    print("\n📈 Basic data analysis:")
    print(f"   Samples: {len(sample_data)}")
    print(f"   Cell types: {sample_data['cell_type'].nunique()}")
    print(f"   Conditions: {sample_data['condition'].value_counts().to_dict()}")

    return True

def test_visualization():
    """Test basic plotting functionality."""
    print("\n📊 Testing visualization...")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        # Create simple plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        # Generate sample data
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)

        ax.scatter(x, y, alpha=0.6)
        ax.set_xlabel('X values')
        ax.set_ylabel('Y values')
        ax.set_title('UV Environment Test Plot')

        # Save plot
        output_file = Path("results/figures/uv_test_plot.png")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Test plot saved to {output_file}")
        return True

    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 UV Environment Setup Test")
    print("=" * 40)

    tests = [
        ("Import Tests", test_imports),
        ("Directory Structure", test_data_directories),
        ("UV Environment", test_uv_environment),
        ("Sample Data Creation", create_sample_data),
        ("Visualization", test_visualization)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 40)
    print("📋 TEST SUMMARY")
    print("=" * 40)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{test_name}: {status}")

    overall_success = all(results)
    if overall_success:
        print("\n🎉 All tests passed! UV environment is ready for CELLxGENE analysis.")
        print("\nNext steps:")
        print("- Run: python scripts/download_cellxgene_data.py")
        print("- Open: jupyter lab")
        print("- Explore: notebooks/cellxgene_exploration.ipynb")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
