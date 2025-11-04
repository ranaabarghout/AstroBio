#!/usr/bin/env python3
"""
Simple test script to check imports and basic functionality.
"""

print("Testing imports...")

try:
    import cellxgene_census
    print("✓ cellxgene_census imported successfully")
except ImportError as e:
    print(f"✗ Failed to import cellxgene_census: {e}")

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pandas: {e}")

try:
    from pathlib import Path
    print("✓ pathlib imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pathlib: {e}")

try:
    import logging
    print("✓ logging imported successfully")
except ImportError as e:
    print(f"✗ Failed to import logging: {e}")

print("\nTesting basic cellxgene_census functionality...")
try:
    # Test opening the census (this might take a moment)
    print("Attempting to open census...")
    with cellxgene_census.open_soma() as census:
        print("✓ Successfully opened census")
        print(f"Available datasets: {list(census['census_data'].keys())}")
except Exception as e:
    print(f"✗ Failed to open census: {e}")

print("Test complete!")
