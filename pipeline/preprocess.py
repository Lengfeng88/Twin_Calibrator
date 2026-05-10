import numpy as np, cv2

def preprocess_frame(frame: np.ndarray,
                     size=(64, 64)) -> np.ndarray:
    """BGR uint8 → ONNX NCHW float32，ImageNet 归一化"""
    img = cv2.resize(frame, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    img  = (img - mean) / std
    return np.transpose(img,(2,0,1))[np.newaxis,...]