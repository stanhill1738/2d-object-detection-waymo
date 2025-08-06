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
- **Mean Average Precision (mAP)** and **Mean Average Recall (mAR)** at various IoU thresholds were used to evaluate model performance.  
- These metrics measure the accuracy and completeness of bounding box predictions across different object sizes and confidence levels.

**Validation Dataset:**  
- A locally sampled subset of the Waymo validation set (balanced to contain vehicles, pedestrians, and cyclists)

**Final Test Set Results:**

- **Overall mAP:** 0.3047  
- **mAP@0.50:** 0.5129  
- **mAP@0.75:** 0.3142  

**Object Size Breakdown:**
- `mAP_small`: 0.0564  
- `mAP_medium`: 0.2959  
- `mAP_large`: 0.5829  

**Recall Metrics:**
- `mAR_100`: 0.3959  
- `mAR_small`: 0.1212  
- `mAR_medium`: 0.4199  
- `mAR_large`: 0.6822  

**Per-Class mAP:**
- Vehicle (Class 1): 0.3848  
- Pedestrian (Class 2): 0.2978  
- Cyclist (Class 3): 0.2314  

These results show that the model performs well on large and medium-sized objects, especially vehicles. Performance drops for smaller objects and less-represented classes such as cyclists, which is expected given the class imbalance in the training data.

---

## Limitations

- The dataset is heavily imbalanced: vehicles dominate the label distribution.
- Cyclists and pedestrians may be underrepresented, leading to reduced detection performance on these classes.
- The model is only trained to detect **three classes**; other object types are ignored or labeled as background.
- Only front-facing camera images are used (not full 360° coverage).
- Small objects (e.g., distant pedestrians or cyclists) are particularly challenging for the model to detect accurately.

---

## Trade-offs

- Faster R-CNN offers high accuracy but is relatively slow during inference compared to single-shot detectors (e.g., YOLO).
- A trade-off was made between detection accuracy and training time: subset training on 20k images provided faster feedback but may limit generalization.
- Batch size and number of epochs were reduced for faster tuning, which could affect convergence and final model performance.
- No augmentation strategies (e.g., image flipping, scaling) were applied, which may limit robustness to varied environments.

---

## Improving Future Performance

To improve these scores, a few things could be done:
1. Train on a much larger dataset — only a subset was used.
2. Run more hyperparameter tests to have greater confidence in the hyperparameters chosen for the final training.
3. Create class-balanced datasets, which may impact the accuracy of some classes but improve performance for under-represented classes.

---

## Accuracy & Safety in Context

Accuracy metrics like mAP and mAR are essential for safety in autonomous vehicle applications, as they reflect the system’s ability to detect and correctly classify objects in dynamic environments. Reliable detection of **all classes** — including less-represented ones like pedestrians and cyclists — is critical for making safe driving decisions. Failing to detect or properly localize vulnerable road users could lead to unsafe or even life-threatening outcomes. Improving class balance and object detection performance directly supports safer, more robust perception systems for real-world deployment.
