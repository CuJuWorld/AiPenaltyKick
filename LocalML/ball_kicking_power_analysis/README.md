# Ball Kicking Power Analysis

This project aims to analyze ball speed and kicking power using supervised machine learning techniques. The analysis is based on raw video data, from which features are extracted and used to train machine learning models.

## Project Structure

- **data/**: Contains all data files.
  - **raw/**: Contains the raw video data used for analysis.
    - `video_data.mp4`: The raw video data used for analyzing ball speed and kicking power.
  - **processed/**: Contains processed feature data.
    - `features.csv`: Processed feature data extracted from the raw video data for training the models.
  - **labels/**: Contains labels for supervised learning.
    - `kicking_power_labels.csv`: Labels corresponding to the kicking power.

- **models/**: Contains trained machine learning models.
  - `ball_speed_model.pkl`: Trained model for predicting ball speed.
  - `kicking_power_model.pkl`: Trained model for predicting kicking power.

- **notebooks/**: Jupyter notebooks for various stages of the project.
  - `data_exploration.ipynb`: Notebook for exploring and visualizing the raw data.
  - `feature_engineering.ipynb`: Notebook for feature extraction and processing.
  - `model_training.ipynb`: Notebook for training the machine learning models.

- **src/**: Source code for data processing, feature extraction, model definition, training, and evaluation.
  - `data_processing.py`: Functions for loading and processing raw video data.
  - `feature_extraction.py`: Functions for extracting relevant features from video data.
  - `model.py`: Defines machine learning models for predicting ball speed and kicking power.
  - `training.py`: Functions for training the models using processed features and labels.
  - `evaluation.py`: Functions for evaluating model performance.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ball_kicking_power_analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the data:
   - Place your raw video data in the `data/raw/` directory.

4. Run the Jupyter notebooks for data exploration, feature engineering, and model training.

## Usage

- Use the `data_exploration.ipynb` notebook to understand the characteristics of the raw video data.
- Use the `feature_engineering.ipynb` notebook to extract and process features from the raw data.
- Use the `model_training.ipynb` notebook to train the machine learning models.

## File Descriptions

- **data/raw/video_data.mp4**: Raw video data for analysis.
- **data/processed/features.csv**: Processed features for model training.
- **data/labels/kicking_power_labels.csv**: Labels for supervised learning.
- **models/ball_speed_model.pkl**: Model for predicting ball speed.
- **models/kicking_power_model.pkl**: Model for predicting kicking power.
- **notebooks/**: Jupyter notebooks for exploration, feature engineering, and training.
- **src/**: Source code for data processing, feature extraction, model definition, training, and evaluation.
- **README.md**: Documentation for the project.
- **requirements.txt**: Lists project dependencies.