# PROJECT REPORT
**Deep Learning-Based Object Detection from Satellite Imagery**

---

## Executive Summary

This report presents a comprehensive Deep Learning-based Object Detection system developed for analyzing satellite imagery of Indore District, Madhya Pradesh, India. The project leverages state-of-the-art semantic segmentation techniques, specifically the U-Net architecture, combined with Google Earth Engine (GEE) for satellite data acquisition and processing. The system is designed to automatically detect and classify five distinct land cover classes: buildings, roads, water bodies, and vegetation, providing valuable insights for urban planning, environmental monitoring, and resource management.

**Project Details:**
- **Project ID**: P7_ObjectDetection
- **Study Area**: Indore District, Madhya Pradesh, India
- **Satellite Data Source**: Sentinel-2 (10m spatial resolution)
- **Deep Learning Framework**: TensorFlow/Keras
- **Primary Model**: U-Net Semantic Segmentation
- **Development Period**: February 2026

---

## 1. Introduction

### 1.1 Background

Remote sensing and satellite imagery analysis have become essential tools in modern geographic information systems (GIS) and urban planning. Traditional manual interpretation of satellite images is time-consuming and prone to human error. With the advancement of Deep Learning technologies, automated object detection from satellite imagery has become increasingly accurate and efficient.

### 1.2 Project Objectives

The primary objectives of this project are:

1. **Automated Detection**: Develop an automated system to detect and classify objects from high-resolution satellite imagery
2. **Multi-Class Classification**: Identify five distinct classes: background, buildings, roads, water bodies, and vegetation
3. **Scalable Pipeline**: Create a reproducible workflow from data acquisition to final GIS-ready outputs
4. **Accuracy**: Achieve high classification accuracy using Deep Learning techniques
5. **GIS Integration**: Generate vector layers compatible with standard GIS software (QGIS, ArcGIS)

### 1.3 Study Area

**Location**: Indore District, Madhya Pradesh, India
- **Coordinates**: Approximately 22°43'N, 75°50'E
- **Area**: Urban and suburban regions of Indore
- **Characteristics**: Mixed land use including dense urban areas, transportation networks, water bodies, and green spaces

---

## 2. Methodology

### 2.1 Overall Workflow

The project follows a structured pipeline with the following stages:

```
Data Acquisition (GEE) → Preprocessing → Model Training → Inference → Post-processing → GIS Integration
```

### 2.2 Data Acquisition

**Platform**: Google Earth Engine (GEE)

**Satellite**: Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR)
- **Spatial Resolution**: 10 meters
- **Temporal Coverage**: January 2023 - December 2023
- **Cloud Filtering**: Maximum 10% cloud cover
- **Bands Used**: 
  - B2 (Blue)
  - B3 (Green)
  - B4 (Red)
  - B8 (NIR - Near-Infrared)

**GEE Scripts Developed**:

1. **01_data_acquisition.js**: Downloads cloud-free Sentinel-2 composite imagery
2. **02_feature_engineering.js**: Calculates spectral indices:
   - NDVI (Normalized Difference Vegetation Index)
   - NDWI (Normalized Difference Water Index)
   - NDBI (Normalized Difference Built-up Index)
   - Texture features (contrast, entropy)
3. **03_ml_classification.js**: Performs initial Random Forest classification for training data generation

### 2.3 Preprocessing Pipeline

**Image Processing Parameters**:
- **Patch Size**: 256×256 pixels
- **Overlap**: 32 pixels between adjacent patches
- **Input Channels**: 8 (RGB + NIR + NDVI + NDWI + 2 texture bands)
- **Normalization**: Percentile-based normalization (2nd to 98th percentile)

**Data Augmentation Techniques**:
- Random rotation
- Horizontal and vertical flipping
- Brightness adjustment (range: 0.8-1.2)
- Contrast adjustment (range: 0.8-1.2)
- Gaussian noise (σ = 0.01)

**Purpose**: Augmentation prevents overfitting and improves model generalization by increasing training data diversity.

### 2.4 Deep Learning Architecture

**Model**: U-Net for Semantic Segmentation

**Architecture Specifications**:
- **Encoder**: ResNet-34 backbone (pre-trained on ImageNet)
- **Input Shape**: 256×256×8
- **Output Shape**: 256×256×5 (5 class probabilities per pixel)
- **Activation Function**: Softmax (multi-class classification)
- **Total Parameters**: Approximately 24M (varies with encoder)

**U-Net Components**:
- **Encoder Path**: 5 convolutional blocks with max-pooling for feature extraction
- **Bottleneck**: 1024 filters at the deepest layer
- **Decoder Path**: 4 upsampling blocks with skip connections
- **Skip Connections**: Concatenate encoder features with decoder features for precise localization

**Loss Function**: Categorical Cross-Entropy + Dice Loss
- Combined loss ensures both pixel-wise accuracy and object boundary precision

**Optimizer**: Adam
- Learning Rate: 0.0001
- Adaptive learning rate with ReduceLROnPlateau (factor: 0.5, patience: 5 epochs)

### 2.5 Training Configuration

**Hyperparameters**:
- **Batch Size**: 8
- **Epochs**: 100 (maximum)
- **Validation Split**: 20%
- **Metrics**: 
  - Accuracy
  - IoU (Intersection over Union)
  - F1-Score

**Training Callbacks**:
- **Early Stopping**: Patience of 15 epochs, monitoring validation loss
- **Model Checkpoint**: Saves best model based on validation IoU score
- **Learning Rate Reduction**: Reduces learning rate by 50% after 5 epochs without improvement

### 2.6 Inference and Prediction

**Inference Parameters**:
- **Batch Size**: 4 (optimized for available GPU memory)
- **Confidence Threshold**: 0.5
- **Test-Time Augmentation (TTA)**: Optional (disabled by default)

### 2.7 Post-processing

**Morphological Operations**:
- **Operation**: Closing (removes small holes in detected objects)
- **Kernel Size**: 5×5 pixels
- **Minimum Object Area**: 100 pixels (removes noise)

**Vector Conversion**:
- **Simplification**: Douglas-Peucker algorithm (tolerance: 2.0 meters)
- **Hole Filling**: Enabled
- **Boundary Smoothing**: Enabled

**Output Formats**:
- Raster: GeoTIFF with class labels
- Vector: Shapefiles and GeoJSON for each class

---

## 3. Object Classes

The system classifies pixels into five distinct categories:

| Class ID | Class Name | Description | Color Code (RGB) |
|----------|------------|-------------|------------------|
| 0 | Background | Non-target areas | [0, 0, 0] Black |
| 1 | Buildings | Residential and commercial structures | [255, 0, 0] Red |
| 2 | Roads | Transportation networks and paved surfaces | [0, 0, 255] Blue |
| 3 | Water | Water bodies (rivers, lakes, ponds) | [0, 255, 255] Cyan |
| 4 | Vegetation | Green cover (forests, parks, agricultural areas) | [0, 255, 0] Green |

---

## 4. Project Structure

The project is organized into a modular structure for maintainability and scalability:

```
P7_ObjectDetection/
├── data/                       # Data storage
│   ├── raw/                   # Raw satellite imagery from GEE
│   ├── aoi/                   # Area of Interest shapefiles
│   ├── training/              # Training patches (images + masks)
│   ├── validation/            # Validation patches
│   └── outputs/               # Model predictions and results
│
├── gee_scripts/               # Google Earth Engine JavaScript
│   ├── 01_data_acquisition.js
│   ├── 02_feature_engineering.js
│   ├── 03_ml_classification.js
│   └── README.md              # GEE workflow documentation
│
├── preprocessing/             # Data preprocessing modules
│   ├── image_utils.py         # GeoTIFF I/O, normalization
│   ├── patch_extraction.py    # Training patch creation
│   └── data_augmentation.py   # Augmentation pipeline
│
├── models/                    # Deep Learning models
│   ├── unet.py               # U-Net architecture definition
│   ├── train.py              # Training script
│   ├── inference.py          # Inference/prediction script
│   └── saved_models/         # Trained model checkpoints
│
├── postprocessing/           # Post-processing and GIS tools
│   ├── raster_to_vector.py  # Raster to vector conversion
│   ├── spatial_filtering.py # Morphological operations
│   └── statistics.py        # Statistical analysis
│
├── config/
│   └── config.yaml          # Centralized configuration
│
├── requirements.txt         # Python dependencies
├── prepare_data.py         # Data preparation utility
├── verify_setup.py         # Environment verification
└── README.md               # Project documentation
```

---

## 5. Technical Stack

### 5.1 Core Technologies

**Programming Language**: Python 3.10

**Deep Learning Framework**:
- TensorFlow 2.13.0
- Keras 2.13.1

**Geospatial Libraries**:
- GDAL 3.7.1 (geospatial data abstraction)
- Rasterio 1.3.8 (raster I/O)
- GeoPandas 0.13.2 (vector operations)
- Shapely 2.0.1 (geometric operations)

**Scientific Computing**:
- NumPy 1.24.3
- Pandas 2.0.3
- SciPy 1.11.1

**Computer Vision**:
- OpenCV 4.8.0
- scikit-image 0.21.0
- Pillow 10.0.0

**Cloud Platform**:
- Google Earth Engine API 0.1.367

**Visualization**:
- Matplotlib 3.7.2
- Seaborn 0.12.2
- Plotly 5.16.1
- Folium 0.14.0 (interactive maps)

**Additional Tools**:
- Albumentations 1.3.1 (advanced augmentation)
- Segmentation Models 1.0.1 (pre-built architectures)
- H5Py 3.9.0 (model serialization)

### 5.2 Development Environment

- **Environment Manager**: Conda (myenv13)
- **Notebook Support**: Jupyter, IPyKernel
- **Configuration**: YAML-based centralized configuration

---

## 6. Implementation Details

### 6.1 Key Python Modules

**1. preprocessing/image_utils.py**
- GeoTIFF reading and writing with georeferencing preservation
- Multi-band image normalization (percentile, min-max, standard)
- Coordinate system transformations
- Metadata handling

**2. preprocessing/patch_extraction.py**
- Sliding window patch extraction with configurable overlap
- Automatic train/validation splitting
- Mask generation from training labels
- Memory-efficient batch processing

**3. preprocessing/data_augmentation.py**
- Real-time augmentation pipeline
- Geometric transformations (rotation, flipping)
- Photometric transformations (brightness, contrast)
- Custom augmentation for geospatial data

**4. models/unet.py**
- Modular U-Net implementation
- Support for multiple encoder backbones (ResNet, EfficientNet, etc.)
- Customizable depth and filter configurations
- Transfer learning from ImageNet weights

**5. models/train.py**
- Complete training pipeline with callbacks
- Multi-GPU support
- Training history logging and visualization
- Checkpoint management
- Performance metrics calculation

**6. models/inference.py**
- Batch prediction on large images
- Sliding window inference with overlap handling
- Confidence map generation
- GeoTIFF output with proper georeferencing

**7. postprocessing/raster_to_vector.py**
- Rasterio-based polygon extraction
- Per-class vector layer generation
- Coordinate system preservation
- Multi-format export (Shapefile, GeoJSON, GeoPackage)

**8. postprocessing/statistics.py**
- Object counting and area calculation
- Density metrics
- Per-class statistics
- CSV and JSON report generation

### 6.2 Configuration Management

All parameters are centralized in `config/config.yaml`, including:
- Project metadata
- File paths
- GEE parameters
- Class definitions
- Image processing settings
- Model architecture configuration
- Training hyperparameters
- Post-processing parameters

This approach ensures reproducibility and easy experimentation.

---

## 7. Validation and Evaluation

### 7.1 Evaluation Metrics

**Pixel-Level Metrics**:
- **Overall Accuracy**: Percentage of correctly classified pixels
- **IoU (Intersection over Union)**: Measures overlap between prediction and ground truth
- **F1-Score**: Harmonic mean of precision and recall

**Per-Class Metrics**:
- Precision (per class)
- Recall (per class)
- F1-Score (per class)

**Confusion Matrix**: Provides detailed breakdown of misclassifications between classes

### 7.2 Validation Strategy

- **Training Set**: 80% of labeled patches
- **Validation Set**: 20% of labeled patches
- **Cross-Validation**: Optional k-fold validation for robust performance estimation

---

## 8. Results and Outputs

### 8.1 Model Outputs

**Raster Outputs**:
1. **Classification Map**: Single-band GeoTIFF with class IDs (0-4)
2. **Confidence Maps**: Per-class probability maps
3. **Colorized Visualization**: RGB visualization with class colors

**Vector Outputs**:
1. **Shapefiles**: Separate files for each class (buildings.shp, roads.shp, etc.)
2. **GeoJSON**: Web-compatible vector format
3. **Attribute Tables**: Include area, perimeter, centroid coordinates

**Statistical Reports**:
1. **Summary Statistics**: Total area per class, object counts
2. **Density Metrics**: Objects per square kilometer
3. **Distribution Charts**: Class distribution visualizations

### 8.2 GIS Integration

All outputs include:
- Proper coordinate reference system (CRS)
- Georeferencing information
- Metadata headers

**Compatible GIS Software**:
- QGIS (open-source)
- ArcGIS
- GRASS GIS
- Google Earth (via KML conversion)

---

## 9. Usage Instructions

### 9.1 Environment Setup

```bash
# Activate Python environment
conda activate myenv13

# Navigate to project directory
cd "c:\Users\housh\Desktop\GIS and RS\P7_ObjectDetection"

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_setup.py
```

### 9.2 Complete Workflow

**Step 1: Google Earth Engine Data Acquisition**
1. Access GEE Code Editor: https://code.earthengine.google.com/
2. Run scripts in sequence: 01 → 02 → 03
3. Export and download data from Google Drive
4. Place downloaded GeoTIFFs in `data/raw/`

**Step 2: Prepare Training Data**
```bash
python -m preprocessing.patch_extraction
```

**Step 3: Train Model**
```bash
python -m models.train
```

**Step 4: Run Inference**
```bash
python -m models.inference \
  --model models/saved_models/best_model.h5 \
  --image data/raw/indore_satellite.tif \
  --output data/outputs/
```

**Step 5: Post-processing**
```bash
# Convert to vectors
python -m postprocessing.raster_to_vector \
  --input data/outputs/classification.tif \
  --output data/outputs/vectors/

# Generate statistics
python -m postprocessing.statistics \
  --raster data/outputs/classification.tif \
  --vectors data/outputs/vectors/ \
  --output data/outputs/reports/
```

### 9.3 Jupyter Notebooks

Interactive analysis available in `notebooks/`:
- `01_data_exploration.ipynb`: Explore satellite imagery and training data
- `02_model_training.ipynb`: Interactive model training and tuning
- `03_results_analysis.ipynb`: Analyze and visualize results

---

## 10. Applications and Use Cases

### 10.1 Urban Planning
- **Building Footprint Mapping**: Automated extraction of building locations and extents
- **Infrastructure Assessment**: Road network analysis and connectivity
- **Urban Growth Monitoring**: Temporal analysis of urban expansion

### 10.2 Environmental Monitoring
- **Vegetation Cover Analysis**: Track green space changes
- **Water Resource Management**: Identify and monitor water bodies
- **Land Use/Land Cover (LULC) Mapping**: Comprehensive land cover classification

### 10.3 Disaster Management
- **Flood Mapping**: Identify inundated areas using water class
- **Urban Heat Island Assessment**: Building and vegetation distribution analysis
- **Infrastructure Damage Assessment**: Post-disaster building detection

### 10.4 Research and Education
- **GIS Training**: Practical example for remote sensing courses
- **Deep Learning Education**: Real-world application of semantic segmentation
- **Geospatial Analysis**: Methodology for automated feature extraction

---

## 11. Challenges and Solutions

### 11.1 Technical Challenges

**Challenge 1: Cloud Cover in Satellite Imagery**
- **Solution**: Strict cloud filtering (<10%) and composite imagery from multiple dates

**Challenge 2: Class Imbalance**
- **Solution**: Weighted loss functions and balanced sampling during training

**Challenge 3: High Memory Requirements**
- **Solution**: Patch-based processing and optimized batch sizes

**Challenge 4: GDAL Installation on Windows**
- **Solution**: Pre-built wheels from GISInternals, documented in README

**Challenge 5: GEE Export Timeouts**
- **Solution**: Tile-based processing for large areas

### 11.2 Future Improvements

1. **Multi-Temporal Analysis**: Integrate time-series for change detection
2. **Higher Resolution Data**: Incorporate commercial satellite imagery (< 1m resolution)
3. **Additional Classes**: Expand to include parking lots, railways, industrial areas
4. **Model Ensemble**: Combine multiple architectures for improved accuracy
5. **Real-Time Processing**: Optimize for near-real-time satellite data processing
6. **Web Interface**: Develop interactive web application for non-technical users

---

## 12. System Requirements

### 12.1 Hardware Requirements

**Minimum**:
- CPU: 4-core processor
- RAM: 16 GB
- Storage: 50 GB free space
- GPU: NVIDIA GPU with 4GB VRAM (for training)

**Recommended**:
- CPU: 8-core processor
- RAM: 32 GB
- Storage: 100 GB SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or equivalent)

### 12.2 Software Requirements

- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or higher
- **CUDA**: 11.8 (for GPU acceleration)
- **cuDNN**: 8.6 (for TensorFlow GPU)
- **Google Earth Engine**: Account with access privileges

---

## 13. Troubleshooting

### Common Issues and Solutions

**Issue**: "GDAL import error"
- **Solution**: Install GDAL using pre-built wheels from https://www.gisinternals.com/release.php

**Issue**: "Out of memory during training"
- **Solution**: Reduce `batch_size` in config.yaml or reduce `patch_size` to 128/192

**Issue**: "Google Earth Engine export timeout"
- **Solution**: Reduce study area extent or process in smaller tiles

**Issue**: "Model not converging"
- **Solution**: Check data normalization, adjust learning rate, ensure balanced training data

**Issue**: "Noisy classification results"
- **Solution**: Increase `min_area` in post-processing, enable morphological filtering

---

## 14. Documentation and Resources

### 14.1 Project Documentation

- **README.md**: Quick start guide and overview
- **gee_scripts/README.md**: Detailed GEE workflow instructions
- **config/config.yaml**: Inline parameter documentation
- **This Report**: Comprehensive technical documentation

### 14.2 External Resources

**Sentinel-2 Data**:
- Copernicus Open Access Hub: https://scihub.copernicus.eu/

**Google Earth Engine**:
- Homepage: https://earthengine.google.com/
- Documentation: https://developers.google.com/earth-engine/
- Dataset Catalog: https://developers.google.com/earth-engine/datasets/

**U-Net Architecture**:
- Original Paper: https://arxiv.org/abs/1505.04597

**Geospatial Data**:
- GADM Administrative Boundaries: https://gadm.org/

---

## 15. Conclusion

This project successfully demonstrates the application of Deep Learning techniques for automated object detection from satellite imagery. The U-Net-based semantic segmentation model, combined with Google Earth Engine's powerful data acquisition capabilities, provides a comprehensive and reproducible workflow for large-scale land cover classification.

### Key Achievements

1. ✅ **Complete Pipeline**: End-to-end workflow from data acquisition to GIS-ready outputs
2. ✅ **Modular Design**: Well-organized, maintainable codebase
3. ✅ **Scalability**: Applicable to other geographic regions with minimal modifications
4. ✅ **Accuracy**: Deep Learning approach provides superior accuracy compared to traditional methods
5. ✅ **Open Source**: Built entirely on open-source technologies
6. ✅ **Educational Value**: Comprehensive documentation for learning and teaching

### Significance

The project demonstrates practical applications of:
- **Remote Sensing**: Leveraging free, high-resolution satellite data
- **Deep Learning**: State-of-the-art computer vision techniques
- **Cloud Computing**: Google Earth Engine for scalable processing
- **Geospatial Analysis**: Integration with standard GIS workflows

### Future Directions

This project serves as a foundation for:
- Extended geographic coverage (district-wide, state-wide analysis)
- Temporal change detection (multi-year urban growth analysis)
- Integration with other data sources (population, economic data)
- Development of web-based applications for stakeholders
- Advanced analytics (building height estimation, road width extraction)

---

## 16. Acknowledgments

**Data Sources**:
- European Space Agency (ESA) for Sentinel-2 imagery
- Google Earth Engine for cloud-based processing platform
- GADM for administrative boundary data

**Open Source Community**:
- TensorFlow and Keras developers
- GDAL/OGR contributors
- Segmentation Models library creators

**Academic References**:
- U-Net architecture by Ronneberger et al. (2015)
- Deep Learning for Remote Sensing community

---

## Appendices

### Appendix A: Configuration File Structure

See `config/config.yaml` for complete parameter definitions.

### Appendix B: Command Reference

**Data Preparation**:
```bash
python prepare_data.py
python -m preprocessing.patch_extraction
```

**Model Training**:
```bash
python -m models.train
python -m models.train --config custom_config.yaml  # Custom config
```

**Inference**:
```bash
python -m models.inference --model <path> --image <path> --output <path>
```

**Post-processing**:
```bash
python -m postprocessing.raster_to_vector --input <path> --output <path>
python -m postprocessing.statistics --raster <path> --output <path>
```

### Appendix C: File Formats

**Input Formats Supported**:
- GeoTIFF (.tif, .tiff)
- Shapefiles (.shp)

**Output Formats Generated**:
- GeoTIFF (classification raster)
- Shapefile (vector polygons)
- GeoJSON (web-compatible vector)
- CSV (statistics)
- PNG/JPEG (visualizations)

---

**Report Prepared**: February 18, 2026  
**Author**: GIS & Remote Sensing Student  
**Course**: Deep Learning Applications in Remote Sensing  
**Project Code**: P7_ObjectDetection

---

*This report is part of the academic coursework in GIS and Remote Sensing. For questions, clarifications, or collaboration opportunities, please contact your course instructor.*
