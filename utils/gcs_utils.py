import torch
import gcsfs

fs = gcsfs.GCSFileSystem(token='google_default')

def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }
    with fs.open(path, 'wb') as f:
        torch.save(checkpoint, f)

def list_gcs_files(prefix):
    return [f.split('/')[-1] for f in fs.ls(prefix) if f.endswith('.pt')]
