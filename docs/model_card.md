# Model Card

## Model Description

**Input:**  
- RGB images (camera frames) from the Waymo Open Perception Dataset  
- Each image is accompanied by bounding box annotations for detected objects (vehicles, pedestrians, cyclists)  

**Output:**  
- A set of 2D bounding boxes per image, each with a predicted class label (vehicle, pedestrian, cyclist) and confidence score  

**Model Architecture:**  
- The model is a **Faster R-CNN** (Region-based Convolutional Neural Network), a two-stage object detector from the PyTorch `torchvision` library.  
- It first proposes object regions and then classifies them while refining bounding box coordinates.  
- The backbone used is a convolutional neural network pre-trained on COCO, fine-tuned on the Waymo data for 3 custom classes.

---

## Performance

**Evaluation Metric:**  
- **Mean Average Precision (mAP) at IoU=0.5** was used to evaluate model performance.  
- This metric measures the accuracy of bounding box predictions and class labels.

**Validation Dataset:**  
- A locally sampled subset of the Waymo validation set (balanced to contain vehicles, pedestrians, and cyclists)

**Best Performance Achieved:**  
- `mAP@0.5`: _[insert your actual value here]_  
- Class imbalance was observed: most labels are for vehicles, fewer for pedestrians, and very few for cyclists.

---

## Limitations

- The dataset is heavily imbalanced: vehicles dominate the label distribution.
- Cyclists and pedestrians may be underrepresented, leading to reduced detection performance on these classes.
- The model is only trained to detect **three classes**; other object types are ignored or labeled as background.
- Only front-facing camera images are used (not full 360° coverage).

---

## Trade-offs

- Faster R-CNN offers high accuracy but is relatively slow during inference compared to single-shot detectors (e.g., YOLO).
- A trade-off was made between detection accuracy and training time: subset training on 20k images provided faster feedback but may limit generalization.
- Batch size and number of epochs were reduced for faster tuning, which could affect convergence and final model performance.
- No augmentation strategies (e.g., image flipping, scaling) were applied, which may limit robustness to varied environments.

