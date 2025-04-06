# Autonomous Drone Landing Zone Identification Using Deep Learning

## Project Description

This project focuses on identifying optimal landing zones for drones in real-time using semantic segmentation of aerial images[cite: 1, 61]. The goal is to distinguish safe landing areas (like grass, paved areas, dirt, gravel) from obstacles (like trees, buildings, water bodies, vehicles) to enhance the safety and autonomy of drone operations, especially in dynamic or unknown environments[cite: 2, 7, 62, 67, 68].

The primary model used is a U-Net architecture with a pre-trained EfficientNetB0 backbone, which demonstrated superior performance in segmenting images compared to an FCN model with a ResNet50 backbone and a non-pretrained U-Net[cite: 2, 3, 4, 17, 18, 62, 63, 64, 65]. The model predicts a segmentation mask for an input aerial image, and a post-processing step identifies the largest safe area furthest from detected obstacles as the optimal landing zone.

## Dataset

* **Source**: The Semantic Drone Dataset was used for training and evaluation[cite: 12, 73].
* **Content**: It contains 400 high-resolution aerial images (6000x4000 pixels) with semantic annotations for 20+ urban classes (e.g., paved-area, dirt, grass, gravel, vegetation, roof, tree, car, obstacle)[cite: 13, 74].
* **Splits**: The data was divided into training (64%), validation (16%), and testing (20%) sets[cite: 15, 16, 17, 77, 78, 79].
* **Real-World Testing**: 12 additional real-world images captured by a personal drone were used for evaluating performance in dynamic scenarios[cite: 14, 75].

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

## Code Structure & Usage

The provided code includes several Python scripts likely intended to be run sequentially or as separate modules within a larger project (e.g., a Jupyter Notebook):

1.  **Training Script (`train_unet_efficientnet.py` - *assumed name*)**:
    * Loads the dataset paths and class dictionary (`class_dict_seg.csv`).
    * Splits data into training, validation, and test sets.
    * Defines data augmentation using `albumentations`.
    * Implements data loading/preprocessing (`load_image_mask`, `preprocess`, `rgb_to_onehot`) and a generator (`generator`).
    * Defines the U-Net model with an EfficientNetB0 backbone (`unet`), Dice coefficient/loss (`dice_coefficient`, `dice_loss`), and uses MeanIoU metric.
    * Sets up callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `CSVLogger`, `LearningRateScheduler` (with exponential decay), and a custom `DisplayCallback` to show predictions during training.
    * Loads a previously saved model (`unet_model.keras`) and continues training from a specified epoch (`current_epoch = 34`).
    * Uses mixed-precision training and `tf.distribute.MirroredStrategy` for potential multi-GPU training.
    * Saves the best model during training.
    * Plots and saves training metrics (Loss, IoU, Dice) after completion.

2.  **Metrics Plotting Script (`plot_metrics.py` - *assumed name*)**:
    * Loads training history from the CSV log (`model_training_csv.log`).
    * Plots Dice Coefficient and IoU curves for training and validation sets.
    * Saves the plot (`metrics_plot.png`).

3.  **Evaluation/Visualization Script (`evaluate_model.py` - *assumed name*)**:
    * Loads the trained model (`unet_model.keras`) and class dictionary.
    * Defines safe and obstacle classes based on the class dictionary.
    * Implements `find_landing_zone` function:
        * Takes a predicted mask (resized to 128x128).
        * Identifies safe areas and obstacles.
        * Creates a buffer zone around obstacles using dilation (`skimage.morphology.dilation`).
        * Finds connected components (`skimage.measure.label`, `regionprops`) in the valid safe areas.
        * Filters regions by size and calculates the distance to the nearest obstacle for remaining regions.
        * Selects the best region (furthest from obstacles, largest area).
        * Returns the centroid coordinates of the best region.
    * Includes helper functions `draw_x` (to mark the landing zone on plots) and `visualize_mask` (to convert label masks to color).
    * Loads sample images and ground truth masks from the dataset.
    * Preprocesses images, predicts masks using the loaded model.
    * Finds the landing zone coordinates using `find_landing_zone` and scales them back to the original image size.
    * Visualizes the original image, true mask (if available), and predicted mask, all marked with the identified landing zone using `matplotlib`.

4.  **Real-World Application Script (`apply_on_real_images.py` - *assumed name*)**:
    * Loads the trained model.
    * Defines paths and image size constants.
    * Includes functions for preprocessing real images (`preprocess_image`), visualizing masks (`visualize_mask`), and finding/drawing the landing zone (`find_and_draw_landing_zone` - slightly different implementation than script 3, using `cv2.dilate` and a simpler region selection logic).
    * Processes images from a specified directory (`REAL_IMAGES_DIR`).
    * Predicts segmentation masks for each image.
    * Identifies and draws the landing zone 'X' on the original image and the colorized prediction mask using `cv2` drawing functions.
    * Displays the original image with the landing zone and the predicted mask with the landing zone side-by-side using `matplotlib`.

**To Run:**

1.  Ensure all dependencies are installed (`pip install -r requirements.txt` - you might need to create this file).
2.  Download the Semantic Drone Dataset and place it in the expected directory structure (e.g., `/kaggle/input/drone-landing-dataset/`). Adjust paths in the code if necessary.
3.  Place real-world test images in the specified directory (e.g., `/kaggle/input/drone-images/`).
4.  Run the training script first to train or continue training the model. This will save `unet_model.keras` and `model_training_csv.log`.
5.  Run the plotting script to visualize training progress.
6.  Run the evaluation script to see predictions on the test set samples and how landing zones are identified.
7.  Run the real-world application script to test the model on your own drone images.

## Model Architecture

* **Encoder**: EfficientNetB0 pre-trained on ImageNet, with frozen weights initially[cite: 2, 17, 63]. Layers from specific blocks are used as skip connections.
* **Decoder**: Standard U-Net expansive path using `Conv2DTranspose` for upsampling, concatenation with corresponding encoder feature maps, followed by `Conv2D` blocks with ReLU activation and Batch Normalization.
* **Output Layer**: A 1x1 Convolution layer with Softmax activation to produce pixel-wise class probabilities[cite: 10].
* **Loss Function**: Categorical Crossentropy.
* **Metrics**: Mean Intersection over Union (MeanIoU) and Dice Coefficient[cite: 2].

## Results

* The U-Net model with the EfficientNetB0 backbone achieved good segmentation accuracy and effectively identified safe landing zones in both dataset images and real-world drone footage[cite: 3, 5, 64, 66].
* It outperformed an FCN-ResNet50 model and a non-pretrained U-Net[cite: 3, 4, 64, 65].
* Training logs (`model_training_csv.log`) and plots (`metrics_plot.png`) provide detailed performance metrics over epochs.

## Future Work

* Integrate the model into real-time drone navigation systems.
* Expand the dataset with more diverse aerial images.
* Explore other segmentation architectures (e.g., DeepLabV3+, PSPNet).
* Incorporate more advanced dynamic obstacle avoidance algorithms.
