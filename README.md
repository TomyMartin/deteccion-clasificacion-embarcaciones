# Vessel Detection and Classification using Passive Acoustic Monitoring

Machine Learning project developed as my final thesis for the Sound Engineering degree at Universidad Nacional de Tres de Febrero (UNTREF).

## Overview
This project explores the automatic detection and classification of vessels using acoustic characteristics extracted from Passive Acoustic Monitoring (PAM) recordings.

The objective was to automate the analysis of large volumes of underwater acoustic data and evaluate machine learning models capable of distinguishing vessel activity under realistic conditions.

## Problem
Passive Acoustic Monitoring systems can generate large volumes of underwater recordings. Manually reviewing these recordings is time-consuming and difficult to scale.

This project investigates acoustic descriptors and machine learning techniques to automatically detect and classify vessel-generated sounds.

## Methodology

The project was divided into three main stages:

1. Construction of a diverse acoustic database.
2. Extraction and evaluation of acoustic descriptors.
3. Development and evaluation of machine learning models for vessel detection and classification.

## Acoustic Features

The analysis included several acoustic descriptors:

- Mel-Frequency Cepstral Coefficients (MFCC)
- Mel Spectrogram
- Spectral Contrast
- Acoustic Signature
- Tonality Index
- Spectral Centroid
- Spectral Bandwidth
- Spectral Flatness
- Zero-Crossing Rate

The Tonality Index was identified as a particularly useful descriptor for distinguishing vessel and non-vessel recordings.

## Machine Learning Models

The following algorithms were evaluated:

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- Decision Tree

The models were implemented using Python and Scikit-learn.

## Results

### Vessel Detection

All evaluated detection models achieved:

**F1-score > 0.95**

The models were also evaluated using independent long-duration recordings to assess their generalization capability under realistic conditions.

### Vessel Classification

Support Vector Machine achieved the best overall classification performance:

**SVM: F1-score > 0.80**

K-Nearest Neighbors achieved the second-best performance:

**KNN: F1-score > 0.75**

## Tech Stack

- Python
- Scikit-learn
- NumPy
- Acoustic Signal Processing
- Machine Learning
- Passive Acoustic Monitoring
- Audio Feature Extraction

## Project Pipeline

Acoustic recordings
↓
Signal preprocessing
↓
Acoustic feature extraction
↓
Dataset preparation
↓
Machine Learning training
↓
Model evaluation
↓
Vessel detection / classification

## Repository Structure

src/
    Source code used for data processing and machine learning models

docs/
    Full thesis document

README.md
    Project overview

## Thesis

The complete thesis is available in the `docs/` folder:

**Detección y clasificación automática de embarcaciones por parámetros acústicos**

Final thesis presented for the Sound Engineering degree at Universidad Nacional de Tres de Febrero (UNTREF).

## Author

Tomás Martín
Sound Engineer
