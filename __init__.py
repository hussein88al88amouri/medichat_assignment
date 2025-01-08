# __init__.py

# Import necessary modules or functions from submodules.
# This is where you can aggregate the public API of your package.

# Example:
# from .module_name import function_name, ClassName

# You can also import specific components to make them available directly from the package level.
# For example:
# from .subpackage.module_name import function_name
from src import llama3_finetune
from src import main

# Initialize any package-level variables or constants
# For example, if you have any version number or author info, you can define it here.

__version__ = "0.1.0"  # Replace with your actual package version
__author__ = "Hussein El Amouri"  # Replace with your name or the author name

# You can include initialization code here, if your package requires any.
# For example, setting up logging, initializing global variables, etc.

# Example:
# import logging
# logging.basicConfig(level=logging.INFO)

# Define a list of publicly exposed items (optional)
# This list is used to specify which functions, classes, or variables
# should be available when `from package_name import *` is used.

# Example:
__all__ = [
    'function_name',  # List the names of the functions, classes, or variables you want exposed
    'ClassName',
]

# If your package uses a specific function or submodule as the primary entry point,
# you can set that here.
# For example, if the main function of the package is in a submodule called 'main.py',
# you can import that here:
# from .main import run

# Initialize any necessary package-specific code here, if needed
# Example for adding environment setup, database initialization, etc.

# If your package contains a command-line interface (CLI), you can import it here,
# so it can be executed as a script if the package is installed:
# from .cli import main

# Any other necessary imports that users should be aware of can go here.
