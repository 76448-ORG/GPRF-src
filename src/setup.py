import os
import sys
import subprocess
from multiprocessing import Pool
from typing import List, Tuple


# --- Configuration ---
mainDir = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(mainDir, "requirements.txt")
MAX_PARALLEL_PROCESSES = 5
PIP_COMMAND = [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet"]


def get_dependencies(filename: str) -> List[str]:
    """Reads dependencies from the requirements file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return []

    with open(filename, 'r') as f:
        dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    return dependencies


def install_package(package_name: str) -> Tuple[str, bool]:
    """
    Installs a single package using a separate pip process.
    Returns (package_name, success_status).
    """
    command = PIP_COMMAND + [package_name]

    try:
        _ = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        return (package_name, True)

    except subprocess.CalledProcessError as e:
        print(f"\nError installing {package_name}:")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return (package_name, False)

    except Exception as e:
        print(f"\nAn unexpected error occurred during installation of {package_name}: {e}")
        return (package_name, False)


def main():
    """Main function to orchestrate parallel dependency installation."""
    dependencies = get_dependencies(REQUIREMENTS_FILE)

    if not dependencies:
        print("No dependencies found or requirements file is missing.")
        return

    print(f"Found {len(dependencies)} dependencies in {REQUIREMENTS_FILE}.")
    print(f"Installing using {MAX_PARALLEL_PROCESSES} parallel processes...")

    with Pool(processes=MAX_PARALLEL_PROCESSES) as pool:
        results = pool.map(install_package, dependencies)

    print("\n--- Installation Summary ---")
    failed_packages = []
    success_count = 0

    for package, success in results:
        if success:
            success_count += 1

        else:
            failed_packages.append(package)

    if not failed_packages:
        print(f"SUCCESS: All {success_count} dependencies installed successfully.")

    else:
        print(f"WARNING: {success_count} packages installed successfully.")
        print(f"FAILURE: {len(failed_packages)} packages failed to install:")

        for pkg in failed_packages:
            print(f"  - {pkg}")

        sys.exit(1)
