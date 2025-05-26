import pydoc
import os
import importlib
import sys

sys.path.insert(0, '../../')

def generate_docs(package_name, output_dir="_pages/"):
    """
    Generate HTML documentation for the given package and save it to output_dir.
    
    Args:
        package_name (str): The name of the package to document.
        output_dir (str): The directory where the HTML files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Import the package
    package = importlib.import_module(package_name)
    
    # Generate documentation for the package itself
    package_doc = pydoc.HTMLDoc().docmodule(package)
    with open(os.path.join(output_dir, f"{package_name}.html"), "w") as f:
        f.write(package_doc)
    
    # Recursively generate documentation for submodules and subpackages
    for root, _, files in os.walk(os.path.abspath(package.__path__[0])):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_name = f"{package_name}.{os.path.splitext(file)[0]}"
                print(root, files, module_name)
                module = importlib.import_module(module_name)
                module_doc = pydoc.HTMLDoc().docmodule(module)
                
                # Save the documentation to an HTML file
                with open(os.path.join(output_dir, f"{module_name}.html"), "w") as f:
                    f.write(module_doc)
    
    print(f"Documentation generated in the '{output_dir}' directory.")

if __name__ == "__main__":
    # Replace 'your_package' with the name of the package you want to document
    generate_docs('lattice_data_tools')
