import sys
import subprocess

PACKAGES = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "imblearn"]

print(f"Python version : {sys.version}")


def install_package(package, version=None):
    if version:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", f"{package}=={version}"]
        )
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    for package in PACKAGES:
        install_package(package)
