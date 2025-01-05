# SocialSense: Decoding Digital Emotions Across Meta, Instagram, and Twitter

## Project Overview

SocialSense is an advanced sentiment analysis project that aims to decode and classify emotions expressed in social media posts across Facebook, Instagram, and Twitter. By leveraging various machine learning techniques, from traditional approaches to state-of-the-art deep learning models, this project provides insights into the emotional context of digital expressions.

## Features

- Multi-platform sentiment analysis (Facebook, Instagram, Twitter)
- Implementation of five different models:
  1. Multinomial Naive Bayes
  2. Logistic Regression
  3. Basic LSTM
  4. LSTM with GloVe embeddings
  5. BERT (fine-tuned)
- Comprehensive data preprocessing pipeline
- Performance comparison across different model architectures

## Dataset

The project uses a curated dataset from Kaggle, comprising social media posts from Twitter, Instagram, and Facebook. Key characteristics include:

- Temporal coverage: 2023
- Geographical distribution: USA, Canada, UK, Australia
- 15 sentiment categories
- Rich metadata including text content, timestamps, and engagement metrics

## Model Performance

| Model                 | Validation Accuracy | Test Accuracy |
|-----------------------|---------------------|---------------|
| Naive Bayes           | 0.58                | 0.55          |
| Logistic Regression   | 0.5471              | 0.5849        |
| Basic LSTM            | 0.58                | 0.5471        |
| LSTM with GloVe       | 0.5283              | 0.6038        |
| BERT                  | 0.774               | 0.849         |

## Installation

```bash
git clone https://github.com/yourusername/SocialSense.git
cd SocialSense
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the required format
2. Download GLoVe.6B.100d file from [Zenodo/GLoVe.6B.100D](https://zenodo.org/records/4925376)
3. Run the python notebook (require GPU for BERT fine-tuning)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- PyTorch
- Transformers
- Scikit-learn
- NLTK
- Pandas
- NumPy

## Future Work

- Implement multi-modal analysis incorporating images and hashtags
- Develop platform-specific models
- Explore ensemble methods
- Expand dataset to include more languages and regions
- Investigate zero-shot learning for emerging sentiment categories
