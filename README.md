# Autonomous Drone Landing Zone Identification Using Deep Learning

## Project Description

This project focuses on identifying optimal landing zones for drones in real-time using semantic segmentation of aerial images. The goal is to distinguish safe landing areas (like grass, paved areas, dirt, gravel) from obstacles (like trees, buildings, water bodies, vehicles) to enhance the safety and autonomy of drone operations, especially in dynamic or unknown environments. The primary model used is a U-Net architecture with a pre-trained EfficientNetB0 backbone, which demonstrated superior performance in segmenting images compared to an FCN model with a ResNet50 backbone and a non-pretrained U-Net. The model predicts a segmentation mask for an input aerial image, and a post-processing step identifies the largest safe area furthest from detected obstacles as the optimal landing zone.

## Dataset

* **Source**: The Semantic Drone Dataset, available at [TU Graz](https://www.tugraz.at/index.php?id=22387). This dataset contains annotated aerial images suitable for semantic segmentation tasks.
* **Real-World Testing**: Additional real-world images captured by a personal drone were used for evaluating performance in dynamic scenarios.

## Dependencies

The Python code relies on the following libraries:

* `tensorflow` (including `keras`)
* `opencv-python` (cv2)
* `numpy`
* `pandas`
* `scikit-learn`
* `albumentations`
* `matplotlib`
* `scikit-image`

## Code Functionality & Usage

All project code, including data loading, preprocessing, model definition, training, evaluation, and visualization, is contained within the Jupyter Notebook: `drone-landing-unet.ipynb`.

### Functionality within `drone-landing-unet.ipynb`
* **Data Handling**: Loads dataset paths, class definitions (`class_dict_seg.csv`), splits data, defines augmentation pipelines (`albumentations`), and implements data loading/preprocessing (RGB to one-hot encoding) via a generator.
* **Model Definition**: Defines the U-Net architecture with an EfficientNetB0 backbone, Dice coefficient/loss, and MeanIoU metric.
* **Training**: Configures and runs model training using Keras, including callbacks like `ModelCheckpoint`, `ReduceLROnPlateau`, `CSVLogger`, `LearningRateScheduler`, and a custom `DisplayCallback` for visualization. Supports loading a saved model to continue training. Utilizes mixed-precision and `tf.distribute.MirroredStrategy`. Saves the best model (`unet_model.keras`) and logs (`model_training_csv.log`).
* **Metrics Plotting**: Loads training history from the log file and generates plots for key metrics (Loss, IoU, Dice Coefficient), saving the output (`metrics_plot.png`).
* **Landing Zone Identification**: Implements the logic (`find_landing_zone`) using `skimage.measure` to analyze predicted masks, apply buffer zones around obstacles via morphological dilation, find safe regions, and select the optimal landing centroid based on distance from obstacles and size.
* **Evaluation & Visualization**: Loads the trained model, runs predictions on dataset samples and real-world images, scales coordinates, visualizes segmentation masks, and marks identified landing zones on images using `matplotlib` and `cv2`.

### How to Run
1.  Ensure all dependencies listed above are installed.
2.  Download the Semantic Drone Dataset from the provided URL and place it in the expected directory structure (e.g., `/kaggle/input/drone-landing-dataset/`). Adjust paths within the notebook if necessary.
3.  Place any real-world test images in the directory specified in the notebook (e.g., `/kaggle/input/drone-images/`).
4.  Open and run the cells sequentially within the `drone-landing-unet.ipynb` Jupyter Notebook using a compatible environment (like Jupyter Lab, Google Colab, or VS Code with Python extensions).

## Model Architecture

* **Encoder**: EfficientNetB0 pre-trained on ImageNet, with frozen weights initially. Features from specific blocks are extracted for skip connections.
* **Decoder**: A typical U-Net expansive path employing `Conv2DTranspose` layers for upsampling, concatenation with corresponding encoder feature maps, followed by standard Convolutional blocks (`Conv2D`, `BatchNormalization`, `ReLU`).
* **Output Layer**: A 1x1 Convolution layer with a Softmax activation function produces the final pixel-wise segmentation map with probabilities for each class.
* **Loss Function**: Categorical Crossentropy.
* **Metrics**: Mean Intersection over Union (MeanIoU) and Dice Coefficient.

## Results

* The U-Net model incorporating the EfficientNetB0 backbone demonstrated effective segmentation accuracy and successfully identified safe landing zones within both dataset images and real-world drone captures.
* Performance metrics documented in the training logs (`model_training_csv.log`) and visualizations (`metrics_plot.png`) confirm the model's capabilities, showing better results than FCN-ResNet50 and non-pretrained U-Net baselines according to the report.

## Future Work

Based on the accompanying report, potential future directions include:

* **Integration with Real-Time Systems**: Deploying the model onto drone hardware for live navigation and landing assistance.
* **Enhancement with More Data**: Augmenting the training dataset with more varied aerial imagery to improve robustness and generalization.
* **Exploring Other Architectures**: Evaluating alternative advanced segmentation models like DeepLabV3+ or PSPNet.
* **Advanced Obstacle Avoidance**: Integrating more complex algorithms to dynamically handle diverse and moving obstacles.
