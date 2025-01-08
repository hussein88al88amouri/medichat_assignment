from setuptools import setup, find_packages
from pathlib import Path

# Read the requirements from the requirements.txt file
def parse_requirements():
    requirements_path = Path(__file__).parent / 'requirements.txt'
    with open(requirements_path, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

setup(
    # The name of your package.
    name='medichat',

    # A version number for your package.
    version='0.1.0',

    # A brief summary of what your package does.
    description='A fine-tuned LLM for medical consultations based on the Meta-Llama 3.1 8B model.',

    # The URL of your project's homepage.
    url='https://github.com/hussein88al88amouri/medichat',

    # The author’s name.
    author='Hussein El Amouri',

    # The author’s email address.
    author_email='alamourihusein@gmail.com',

    # This defines which packages should be included in the distribution.
    packages=find_packages(),

    # Read dependencies from the requirements.txt
    install_requires=parse_requirements(),

    # Additional classification of your package.
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

    # A license for your package.
    license='MIT',

    # You can add entry points for command-line tools if your package includes such functionality.
    entry_points={
        'console_scripts': [
            'medichat=medichat.cli:main',  # Adjust to your actual CLI entry point, if any
        ],
    },

    # If you have data files (like configuration files), you can specify them here.
    data_files=[
        # Example of configuration files for saving the model, etc.
        ('share/config', ['config/config.json']),
    ],

    # If your package has specific testing requirements or needs test dependencies, list them here.
    extras_require={
        'dev': ['pytest', 'tox'],  # Optional dependencies for development or testing
        'docs': ['sphinx'],  # Optional dependencies for documentation generation
    },

    # Specify your package's minimum supported Python version
    python_requires='>=3.8',

    # If your package includes command-line scripts, you can list them here
    scripts=['scripts/cli_script.py'],  # Update path if you have a script to run

    # If your package includes C extensions or other modules, specify them here.
    ext_modules=[],
)
