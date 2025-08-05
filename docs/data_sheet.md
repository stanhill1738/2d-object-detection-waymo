# Waymo Open Dataset - Datasheet

This is a datasheet to summarise the [Waymo Open Dataset](https://waymo.com/open/), specifically the Perception dataset.

## Motivation

### For what purpose was the dataset created?

As per the Waymo Open Dataset [Website](https://waymo.com/open/):
"We have released the Waymo Open Dataset publicly to aid the research community in investigating a wide range of interesting aspects of machine perception and autonomous driving technology.

The Waymo Open Dataset is composed of three datasets - the Perception Dataset with high resolution sensor data and labels for 2,030 segments, the Motion Dataset with object trajectories and corresponding 3D maps for 103,354 segments, and the End-to-End Driving Dataset with camera images providing 360-degree coverage and routing instructions for 5,000 segments."

For the purpose of thr 2D bounding box use case, the dataset used is the Perception Dataset.

### Who created the dataset?

This dataset was created for and by Waymo.

Citation: @InProceedings{Sun_2020_CVPR, author = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and Vasudevan, Vijay and Han, Wei and Ngiam, Jiquan and Zhao, Hang and Timofeev, Aleksei and Ettinger, Scott and Krivokon, Maxim and Gao, Amy and Joshi, Aditya and Zhang, Yu and Shlens, Jonathon and Chen, Zhifeng and Anguelov, Dragomir}, title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset}, booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, month = {June}, year = {2020} }
 
## Composition

### What do the instances that comprise the dataset represent? 

Details of the Perception Dataset can be found [here](https://waymo.com/open/data/perception/).
For the purpose of the 2D bounding box use case, the data used is the camera images and the corresponding camera boxes (ie. the labels), along with some metadata about the vehicle pose (eg. which city is was taken in).

The labels correspond to vehicles, pedestrians and cyclists.

### How many instances of each type are there?

While the distribution of each label is not published, it is possible to find it out from running a script over each frame and tallying up all the different labels. Due to the time constraints of this project, and the large size of the dataset (ca. 800,000 frames), this has not been done.
However, a distribution was calculated of a 4000 randomly sampled subet, and the results were as follows:
- Class 1 (Vehicles): 77.96%
- Class 2 (Pedestrians): 21.42%
- Class 4 (Cyclists): 0.62%

You may notice that there is no Class 3. As per the [labels.proto](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/label.proto#L63), Class 3 should correspond to "sign", however, there is no such data.

### Is there any missing data?

As mentioned above, there is some confusion caused by the label to class mapping. Class 3 ("sign") is missing.

### Does the dataset contain data that might be considered confidential?

It seems that confidential information, such as the faces of people and car number plates, are blurred.

## Collection process

### How was the data acquired?

Camera images come from the Waymo data-collection vehicles.
The 2D bounding box annotations are created in the following way:

- 3D bounding boxes are first generated using LiDAR data.
- These 3D boxes are then projected into each camera's image plane using calibrated camera intrinsics and extrinsics.
- The projection results in axis-aligned 2D bounding rectangles around the visible portion of each object in the image.
- Human annotators manually review and refine the 2D boxes to correct errors caused by occlusion, truncation, or projection inaccuracies.
- Final 2D annotations tightly fit the object's visible extent in each camera view.

### If the data is a sample of a larger subset, what was the sampling strategy?

For the hyperparameter experimentation and model training, the data was sampled down.
This process is explained [here]().

### Over what time frame was the data collected?

- Data was collected over several months across multiple U.S. cities (Phoenix, Mountain View, San Francisco, Seattle, Detroit).
- The dataset captures a wide range of lighting (day, night, dusk/dawn) and weather conditions.
- Each driving segment lasts 20 seconds, sampled at 10 Hz (200 frames per segment).
- While exact collection dates are not specified, the data was gathered prior to the initial public release in late 2019.

## Preprocessing/cleaning/labelling

Details about data processing can be found [here](). 
 
## Uses

### What other tasks could the dataset be used for? 

Beyond 2D object detection and tracking, the Waymo Open Dataset can be used for:

- **Domain Adaptation**: Studying how perception models generalize across different geographic regions, lighting, and weather conditions.
- **Sensor Fusion**: Research in combining camera and LiDAR data for more robust object detection, segmentation, or tracking.
- **Semantic & Instance Segmentation**: Especially with the panoptic segmentation labels available in later dataset versions.
- **Depth Estimation**: Learning depth from monocular images or stereo-like setups using synchronized LiDAR data as supervision.
- **Scene Understanding**: Modeling interactions between road users, predicting intentions, and inferring scene layouts.
- **Simultaneous Localization and Mapping (SLAM)**: Using sequences of LiDAR and camera frames for mapping or localization tasks.
- **Trajectory Prediction**: Leveraging object tracks to model and predict future motion of vehicles, pedestrians, and cyclists.
- **NeRF and 3D Reconstruction**: Utilizing multi-view images and LiDAR to reconstruct photorealistic scenes (especially in later versions that include camera intrinsics and per-ray data).
- **Adverse Condition Robustness**: Evaluating and improving model robustness under rare or difficult driving conditions (e.g. occlusion, night, rain).

These capabilities make the dataset valuable for both academic research and industrial-scale autonomous driving development.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

#### Dataset Composition and Collection Risks

- **Geographic Bias**: The dataset is collected in specific U.S. cities (e.g., Phoenix, San Francisco, Mountain View, Seattle, Detroit), which may lead to over-representation of certain road layouts, signage, driving behaviors, and demographic distributions. It may underrepresent rural areas, developing countries, or areas with differing infrastructure or cultural driving norms.

- **Weather and Lighting Diversity**: While the dataset includes various weather and lighting conditions, it is still relatively biased toward clear, daylight conditions. Extreme conditions (heavy snow, dense fog, severe rain) are rare or absent, potentially limiting model robustness in such scenarios.

- **Demographic Representation**: There is no explicit labeling or control for human demographic attributes (e.g., age, race, clothing style), which may cause unintentional bias in pedestrian or cyclist detection and tracking.

- **Sensor Setup Bias**: The sensor suite (multiple LiDARs + 5 cameras) is specific to Waymo's hardware. Models trained on this dataset may not generalize well to other platforms with different sensor configurations or lower-cost sensors.

#### Labeling and Annotation Considerations

- **Projection Inaccuracies**: 2D bounding boxes are projected from 3D annotations and refined by human annotators, but errors may remain—particularly for occluded, truncated, or small objects.

- **Human Annotation Bias**: Human labelers may unconsciously reflect their own cultural or perceptual biases in tasks like box refinement, class labeling, or occlusion judgment.

- **Undersampled Edge Cases**: Rare behaviors or vulnerable road users (e.g., people in wheelchairs, construction workers, animals) may be underrepresented or mislabeled, reducing model fairness or safety in these cases.

#### Risks and Harms for Downstream Use

- **Unfair Treatment**: Models trained on this dataset could exhibit performance disparities across demographic or geographic lines, potentially resulting in disproportionate safety risks to underrepresented groups.

- **Legal/Regulatory Risks**: Deployment of models trained on this dataset in jurisdictions outside the U.S. without local adaptation could lead to legal or ethical issues due to mismatched traffic laws or cultural expectations.

- **Financial or Operational Harm**: Over-reliance on this dataset without evaluating generalization could lead to product failures or safety incidents in deployment, especially in edge cases or new markets.

#### Mitigation Strategies

- **Geographic and Demographic Audits**: Conduct targeted evaluation and model testing on new datasets from different regions or communities to assess generalization and fairness.

- **Diverse Supplementation**: Augment training with data from other regions, conditions, and sensor setups to improve robustness.

- **Bias and Fairness Analysis**: Implement tools and protocols to monitor for disparities in detection/tracking performance across different populations and settings.

- **Domain Adaptation Techniques**: Use domain adaptation or domain generalization methods to improve model transferability across different real-world scenarios.

- **Transparency in Deployment**: Clearly document training datasets, limitations, and known biases when deploying perception models into safety-critical applications.

### Are there tasks for which the dataset should not be used? If so, please provide a description.

While the Waymo Open Dataset is highly valuable for autonomous vehicle perception tasks, there are certain applications for which it is not suitable or may pose ethical, legal, or technical risks:

#### Biometric Identification or Face Recognition
- The dataset includes people (e.g., pedestrians, cyclists), but it does **not provide facial details or consent** for identity recognition.
- Using the dataset for **biometric analysis, facial recognition, or re-identification** of individuals would be ethically inappropriate and likely violate privacy norms.

#### Demographic or Behavioral Profiling
- The dataset was **not designed to support inferences about demographic attributes** (e.g., age, gender, race, behavior).
- Attempting to infer or model such attributes could lead to **unfair stereotyping, discrimination, or privacy harms**, especially since these labels are not included and annotators were not trained to make such judgments.

#### Applications Outside Driving Contexts
- The dataset was collected using a **sensor suite mounted on a vehicle**, in urban and suburban U.S. traffic environments.
- It should **not be used to train models for unrelated domains** (e.g., indoor robotics, aerial drones, healthcare applications), as the assumptions about geometry, object types, and sensor input will not hold.

#### Safety-Critical Deployment Without Validation
- Models trained solely on this dataset **should not be deployed in safety-critical systems** (e.g., real-world autonomous driving, ADAS) **without extensive testing and validation** in the target environment.
- The dataset lacks some edge cases, rare hazards, and region-specific driving norms that are essential for safe deployment.

## Distribution

### How has the dataset already been distributed?

The Waymo Open Dataset has been publicly distributed by Waymo LLC since December 2019 via the official website:

🔗 [https://waymo.com/open](https://waymo.com/open)

- **Access Requirements**:
  - Users must agree to a **research-only, non-commercial license**.
  - Distribution is governed by specific terms of use, prohibiting redistribution and certain forms of downstream usage (e.g., commercial use, identity recognition).

- **Distribution Format**:
  - The dataset is available for download as **compressed TFRecord files**, organized into training, validation, and test segments.
  - Supplemental assets include tools for data loading, visualization, and evaluation (available on GitHub).

- **Versions Released**:
  - **Perception v1.0 (Dec 2019)** — the version described in the original paper.
  - Subsequent updates include **v1.1**, **v1.2**, **v1.3**, **v1.4.3**, and **v2.0.1**, adding improvements to labeling quality, new metadata (e.g., camera intrinsics), and additional data formats (e.g., panoptic segmentation, NeRF-ready image data).
  - New subsets have been released for **motion prediction**, **occupancy forecasting**, and **simulated scenarios**.

- **Community Use**:
  - The dataset has been used in **academic research, benchmarks, and public challenges** (e.g., the Waymo Open Dataset Challenges at CVPR and NeurIPS).
  - Citations and references to the dataset appear in hundreds of peer-reviewed papers.

### Is it subject to any copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The Waymo Open Dataset is protected under **copyright** and is distributed under a **custom research-only license** governed by Waymo LLC.

- **License Type**:
  - The dataset is released for **non-commercial, academic research use only**.
  - Users must agree to the **Waymo Dataset License Agreement**, which outlines permitted and prohibited uses.

- **Permitted Uses**:
  - Academic research, model development, benchmarking, and publication, as long as the usage is non-commercial.
  - Participation in challenges hosted by Waymo (e.g., CVPR, NeurIPS competitions) under the challenge rules.

- **Prohibited Uses**:
  - **Commercial use** (e.g., training products or services for sale or deployment).
  - **Redistribution** of the dataset or derived data.
  - **Re-identification**, **facial recognition**, or any use attempting to infer personal identity or demographic information.
  - Any usage that violates privacy, ethical standards, or applicable laws.

- **Attribution**:
  - Any publications or derived works must cite the official paper:
    > Sun et al., "Scalability in Perception for Autonomous Driving: Waymo Open Dataset", CVPR 2020. [arXiv:1912.04838](https://arxiv.org/abs/1912.04838)

- **Access Requirements**:
  - Users must register and agree to the license via Waymo’s official portal: [https://waymo.com/open](https://waymo.com/open)

## Maintenance

The **Waymo Open Dataset** is maintained by **Waymo LLC**, an autonomous driving technology company and subsidiary of Alphabet Inc. (Google's parent company).

- **Maintaining Organization**:  
  **Waymo LLC**  
  [https://waymo.com/open](https://waymo.com/open)

- **Maintainer Roles**:
  - Curating and updating the dataset with improved labels, new versions, and additional subsets (e.g., motion, occupancy, simulation).
  - Providing documentation, tools, and evaluation scripts.
  - Hosting annual challenges (e.g., at CVPR, NeurIPS) to benchmark progress on core tasks.
  - Responding to community feedback and bug reports via official GitHub repositories and mailing lists.

- **Latest Updates**:
  - Waymo has released multiple updated versions (v1.0 through v2.0.1), adding features like improved annotation quality, new segmentation labels, and NeRF-style camera intrinsics.



