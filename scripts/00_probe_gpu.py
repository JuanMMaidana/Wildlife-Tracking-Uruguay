#!/usr/bin/env python3
"""
GPU and CUDA Probe Script
Quick test to verify CUDA, PyTorch, and Ultralytics setup
"""

import torch
import sys

def check_cuda():
    """Check CUDA availability and configuration"""
    print("=== CUDA Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        
        # Test GPU memory allocation
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.randn(1000, 1000, device=device)
            print(f"GPU memory test: SUCCESS")
            print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        except Exception as e:
            print(f"GPU memory test: FAILED - {str(e)}")
    else:
        print("CUDA not available - will use CPU")
    
    print()

def check_ultralytics():
    """Check Ultralytics YOLO installation"""
    print("=== Ultralytics Check ===")
    
    try:
        from ultralytics import YOLO
        print("Ultralytics YOLO: INSTALLED")
        
        # Test model loading (without weights)
        try:
            # This will download YOLOv8n if not present, but just for testing
            print("Testing model initialization...")
            model = YOLO('yolov8n.pt')  # Small model for testing
            print("Model initialization: SUCCESS")
            
            # Test inference on dummy data
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(dummy_image, verbose=False)
            print("Dummy inference test: SUCCESS")
            
        except Exception as e:
            print(f"Model test: FAILED - {str(e)}")
            
    except ImportError:
        print("Ultralytics YOLO: NOT INSTALLED")
        print("Install with: pip install ultralytics")
    
    print()

def check_opencv():
    """Check OpenCV installation"""
    print("=== OpenCV Check ===")
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        print("OpenCV: INSTALLED")
        
        # Test video reading capability
        try:
            # Test basic video capture (without actual video)
            cap = cv2.VideoCapture()
            print("Video capture capability: AVAILABLE")
        except Exception as e:
            print(f"Video capture test: FAILED - {str(e)}")
            
    except ImportError:
        print("OpenCV: NOT INSTALLED")
        print("Install with: pip install opencv-python")
    
    print()

def check_other_deps():
    """Check other required dependencies"""
    print("=== Other Dependencies ===")
    
    deps = [
        ('numpy', 'np'),
        ('yaml', 'yaml'),  
        ('tqdm', 'tqdm'),
        ('pathlib', None),
        ('json', None)
    ]
    
    for dep_name, import_name in deps:
        import_name = import_name or dep_name
        try:
            __import__(import_name)
            print(f"{dep_name}: INSTALLED")
        except ImportError:
            print(f"{dep_name}: NOT INSTALLED")
    
    print()

def main():
    """Run all checks"""
    print("MegaDetector Pipeline - System Check")
    print("=" * 50)
    print()
    
    check_cuda()
    check_ultralytics()
    check_opencv()
    check_other_deps()
    
    # Summary
    print("=== Summary ===")
    if torch.cuda.is_available():
        print("✅ System ready for GPU-accelerated processing")
    else:
        print("⚠️  System will use CPU (slower processing)")
    
    print("\nNext steps:")
    print("1. Download MegaDetector weights to models/detectors/")
    print("2. Copy videos to data/videos_raw/")
    print("3. Run: python scripts/10_run_md_batch.py")

if __name__ == "__main__":
    main()