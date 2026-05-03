"""
Export trained ResNet-18 weights → ONNX
Places model.onnx in model_repository/chest_xray/1/
"""

import os
import torch
import torch.nn as nn
from torchvision import models

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
ONNX_OUT     = os.path.join(
    os.path.dirname(__file__), "..", "model_repository", "chest_xray", "1", "model.onnx"
)
NUM_CLASSES  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights_path: str) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


def export(model: nn.Module, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"[DONE] ONNX model saved → {out_path}")
    # Quick size check
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"       File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    print(f"[INFO] Loading weights from {WEIGHTS_PATH}")
    model = load_model(WEIGHTS_PATH)
    export(model, ONNX_OUT)

    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])
        dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out = sess.run(None, {"input": dummy_np})
        print(f"[OK]   ONNX inference test passed. Output shape: {out[0].shape}")
    except ImportError:
        print("[WARN] onnxruntime not installed — skipping verification")
