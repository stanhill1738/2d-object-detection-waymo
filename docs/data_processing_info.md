# Data Processing

## Waymo Dataset

The Waymo Perception dataset conists of a `training` folder, a `validation` folder and a `testing` folder.
We utilise the `training` and `validation`sets because the `testing` set comes without labels, due to the fact that it is used for the public competitions.

Within each of `training` and `validation`, there are 799 and 203 parquet files respectively for both `camera_image` (the image) and `camera_box` (ie. the label). Each parquet file consists of 198 unique frames, per camera (5 cameras), meaning that there are 990 frames in total per parquet.

## Converting Parquet Files to .pt Samples

To generate `.pt` files for training from the Waymo Open Dataset, I used a custom Python script (see `scripts/process_batch.py`) that processes `.parquet` files stored in Google Cloud Storage (GCS). The script extracts RGB images and their associated 2D bounding boxes, then converts them into PyTorch tensors and saves them as `.pt` files.

### Summary of the Script Workflow

- The script loads and processes data from three parquet sources:
  - `camera_image/` for RGB image data
  - `camera_box/` for 2D bounding box annotations
  - `stats/` for metadata (only partially used)

- The dataset is processed in **parallel batches** using a job ID (`BATCH_ID`) passed via the command line.
- From each `.parquet` file, the script:
  - Extracts individual frames and corresponding camera images
  - Converts the image to a PyTorch tensor
  - Parses bounding boxes and object class labels
  - Saves each frame as a `.pt` file containing:  
    `{"image", "boxes", "labels", "meta"}`

- The `.pt` files are saved locally to `/tmp/waymo_data/<split>/`.

- A **global limit** is enforced to only produce 18,000 `.pt` files across all batches, with each batch restricted to a proportional number (`900` per batch if using 20 jobs).

### Why This Was Done

- The output `.pt` files are used to train a Faster R-CNN object detection model.
- Processing in parallel batches enabled scalable conversion from GCS-stored parquet format to a local PyTorch-ready dataset.
- This script allowed me to efficiently sample and prepare a subset of the dataset for prototyping and experimentation.

### Example Usage

To run batch 0 out of 20:

```bash
python scripts/process_batch.py 0 
```

## Usage of .pt files

**Training set** - this is used as training data for the hyperparameter tuning and final model training. 18,000 .pt files were generated for training. Only a subset was used for hyperparameter tuning, while the full set was used for the final model training.

**Validation set** - this is used for validation measurements during the hyperparameter tuning, in order to find the optimal combination. It is also used during the final model training to measure improvements between epochs and to initiate early stoppage in the training. Again, only a subset was used for hyperparameter tuning, while the full set was used for the final model training.

**Testing set** - this was generated from the wider Waymo validation set too, as the testing Waymo set doesn't have labels. I made sure to change the random seed when pulling samples at random, to minimise duplicates with the validation set, then used a bash scrip to confirm uniqueness. This set is used for evaluating the final model.

## Cleaning up missing data

Although the Waymo dataset is pretty clean on the whole, I used a bash script to check that the `.pt` files were of similar size, to have an estimation that they are uniform. There were a handful of samples that were much smaller in size than the others, which would indicate issues with the download or transformation, or from the original sample. These were discarded to ensure clean data.

