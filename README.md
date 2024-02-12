# Mtech_Thesis
Efficient Local Similarity based Hyperspectral Imagery Classification using CNN

## Exploratory Data Analysis

### 1. Spectral Signature Visualization
- **Function Name:** Inline plotting code (not encapsulated in a function)
- **Function Objective:** To visualize the spectral signature of randomly selected pixels. The spectral signature is a plot of reflectance values across different spectral bands for a single pixel.
- **Inferences from the Plot:**
- **Types of Inferences:** Identification of unique spectral patterns that may correspond to different materials or land cover types.
- **Plot Type:** Line plots displaying the spectral signature, which helps in understanding how reflectance varies across different wavelengths for individual pixels.
### 2. Class Distribution Visualization
- **Function Name:** Inline plotting code (specifically for class distribution, not encapsulated in a function)
- **Function Objective:** To plot the distribution of different classes within the dataset, providing insights into class balance or imbalance.
- **Inferences from the Plot:**
- **Types of Inferences:** Understanding whether the dataset is balanced or if certain classes are underrepresented, which could impact model training and generalization.
- **Plot Type:** Histogram or bar chart showing the frequency of each class within the dataset.
### 3. Visualize Samples from Each Class
- **Function Name:** visualize_samples
- **Function Objective:** To display sample images from each class within the dataset. This function is designed to showcase the spatial characteristics and differences between classes by displaying a specified number of samples for each class.
- **Inferences from the Plot:**
- **Types of Inferences:** Visual inspection of the variance within and across classes, which can reveal how distinguishable the classes are based on visual features alone. It can also highlight potential challenges in classifying similar-looking classes.
- **Plot Type:** Grid of images, with each row or column representing a different class. This spatial plot provides a visual representation of samples, enhancing understanding of class characteristics.
### 4. General Observations and Inferences
- **Understanding Spectral and Spatial Features:** The combined use of spectral signature plots and spatial image displays allows for a comprehensive understanding of both the spectral and spatial characteristics of the hyperspectral data. This dual perspective is crucial for tasks such as classification, anomaly detection, and material identification in hyperspectral imagery.
- **Dataset Insights:** The visualizations collectively provide insights into the dataset's complexity, including the variability within classes, the distinction between different classes, and potential preprocessing needs (e.g., normalization, augmentation) to address class imbalances or enhance model robustness.

