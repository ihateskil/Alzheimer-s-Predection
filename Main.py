#!/usr/bin/env python3
"""
Alzheimer's Disease Prediction Tool
Main entry point for the application
"""

import sys
import os

# Add the src directory to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the main function from the src package
from src import main

if __name__ == "__main__":
    main()