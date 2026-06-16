"""
Run this once to download the dataset and build all embeddings + clustering models.
Usage: python setup.py
"""
import subprocess, sys

steps = [
    ("Preparing corpus + embeddings", [sys.executable, "part1_prepare.py"]),
    ("Running fuzzy clustering",       [sys.executable, "part2_clustering.py"]),
]

for label, cmd in steps:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFailed at: {label}")
        sys.exit(result.returncode)

print("\nSetup complete. Start the server with:")
print("  uvicorn part4_api:app --reload")
