"""
DashClarity AI - Automated Weights Downloader
Verifies local pretrained model weights against SHA-256 checksums and downloads missing checkpoints.
"""
import sys
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.weights_manager import WEIGHTS_REGISTRY, get_weights_path

def main():
    print("=" * 70)
    print(" DashClarity AI — Pretrained Weights Verification & Downloader")
    print("=" * 70)
    
    for model_name, info in WEIGHTS_REGISTRY.items():
        print(f"\nChecking: {model_name} ({info['filename']})...")
        try:
            path = get_weights_path(model_name, auto_download=True)
            print(f"  [OK] Ready at: {path}")
        except Exception as e:
            if info.get("local_only"):
                print(f"  [NOTE] {model_name} is marked local_only (optional comparison checkpoint): {e}")
            else:
                print(f"  [ERROR] Failed to fetch {model_name}: {e}")
                
    print("\n" + "=" * 70)
    print(" Weights verification complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
