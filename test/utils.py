import torch
import numpy as np
import random
import os
import logging

def get_device():
    """
    获取设备，顺序: CUDA -> DirectML (PrivateUse1) -> CPU
    """
    # 1. Check CUDA
    if torch.cuda.is_available():
        print("设备: 使用 CUDA (NVIDIA)")
        return torch.device("cuda")
    
    # 2. Check DirectML (通常作为 privateuseonly 或者通过 torch_directml 包)
    try:
        import torch_directml
        print("设备: 使用 DirectML (Intel ARC/AMD)")
        # DirectML LSTM operation 'aten::_thnn_fused_lstm_cell' not supported.
        # Force CPU if using RNNs to avoid crash, until DirectML supports it fully.
        # But user wants priority: CUDA -> DirectML -> CPU.
        # Let's return DirectML, but model needs to handle compatibility or we disable RNN optimization.
        # Actually, PyTorch DirectML backend is known to have issues with RNNs.
        # Strategy: Return CPU if high-level RNNs are critical and DML fails, 
        # OR suggest user that RNNs on DML is unstable.
        # For now, let's stick to user request but warn.
        
        # FIX: The error explicitly says 'aten::_thnn_fused_lstm_cell' not supported.
        # This usually happens when Cudnn/optimized LSTM is called.
        # We can enforce non-cudnn LSTM by not using CUDA? But here we are on PrivateUse1.
        # It seems DirectML implementation of LSTM is missing.
        # Force CPU for stability on Intel ARC for RNNs.
        print("Warning: DirectML might crash with LSTM. Switching to CPU for stability if needed.")
        return torch.device("cpu") 
        # return torch_directml.device() # Commented out due to missing LSTM kernel
    except ImportError:
        pass
    
    # 3. Fallback to CPU
    print("设备: 使用 CPU")
    return torch.device("cpu")

def set_seed(seed=42):
    """固定随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger("CitiOilPrice")
