"""
# CMT Speaker Identification Project

This project implements a speaker identification system for the CMT Data Science interview.

## Problem Overview
- Classify audio utterances (represented as LPC Cepstrum coefficients) to identify speakers
- 9 male speakers (labels 0-8) speaking the same vowel
- Variable-length time series data (7-29 time points)
- 12 LPC coefficients per time point

## Solution Approach

### 1. Data Loading and Parsing
- Parse non-standard format with blocks separated by double newlines
- Handle variable-length sequences
- Load training labels from separate file

### 2. Feature Engineering
- Statistical features: mean, std, median, min, max, range
- Temporal features: first/last values, overall change, slopes
- Trajectory features: curvature, trends
- Global features: sequence length, energy, cross-correlations

### 3. Model Selection
- Ensemble approach with RandomForest as primary model
- StandardScaler for feature normalization
- Cross-validation for robust evaluation

### 4. Evaluation Strategy
- Focus on worst-performing speaker (as per grading criteria)
- Balanced performance across all speakers
- Cross-validation for generalization assessment

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run this command for in-depth visualization and interpretability, including pattern analysis and feature importance insights.
python analysis.py

# Run the command to execute the complete pipeline and generate the test Predictions label
python speaker_identification.py

```

## Files Required
- train.txt: Training data
- test.txt: Test data  
- train_block_labels.txt: Training labels

## Output
- cmt_speaker_predictions.csv: Test set predictions
- data_exploration.png: Data visualizations
- Console output: Performance metrics and analysis

## Key Features
- Robust feature extraction for variable-length sequences
- Comprehensive data visualization
- Cross-validation evaluation
- Balanced performance optimization
- Professional code structure with error handling
"""