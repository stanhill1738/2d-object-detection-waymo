import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model checkpoint to a path in the mounted GCSFuse directory.
    """
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)

def list_local_files(prefix):
    """
    Recursively list .pt files under the given prefix directory.
    """
    pt_files = []
    for root, _, files in os.walk(prefix):
        for f in files:
            if f.endswith('.pt'):
                pt_files.append(os.path.join(root, f))
    return pt_files
