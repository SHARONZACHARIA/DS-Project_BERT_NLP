# Overview

This project explores the use of **transformer-based models** (BERT, T5) in comparison to **classical NLP techniques** (Bag-of-Words, TF-IDF) for tasks related to **text summarization** 
The goal is to understand the performance of each models and the computational efficieny of each model.
## Project Description

This project compares the performance of **transformer-based models** (BERT and T5) with **traditional NLP techniques** such as **Bag-of-Words** (BoW) and **TF-IDF** for:**Text Summarization**: Condensing long news articles into concise summaries.By evaluating these models, we aim to highlight the advantages and limitations of transformers in comparison to classical methods for these tasks.

## Dataset

This project uses the **CNN/Daily Mail** dataset, a popular dataset for training and evaluating text summarization models. 

- Dataset: [CNN/Daily Mail Dataset](https://www.tensorflow.org/datasets/community_catalog/huggingface/cnn_dailymail)

## Methodology

### Data Preprocessing

- **Text Cleaning**: Removing unnecessary punctuation, special characters, and formatting.
- **Tokenization**: Splitting the text into words or subwords for model processing.

### Models

- **Traditional NLP Models**:
  - **Bag-of-Words (BoW)**: A simple method of vectorizing text based on word frequency.
  - **TF-IDF**: Measures the importance of words in a document relative to a corpus.

- **Transformer Models**:
  - **BERT**: A pre-trained transformer model fine-tuned for summarization 
  - **T5**: A model designed for text-to-text tasks, including summarization 

### Evaluation Metrics

- **ROUGE Score**: Measures the quality of the summaries.
-**Inference Time** : For calculating computational efficiency

## Key Libraries

- `transformers` (for transformer models like BERT and T5)
- `torch` (for deep learning model implementations)
- `sklearn` (for traditional NLP methods and metrics)
- `pandas` (for dataset handling and manipulation)
- `nltk` (for text preprocessing and tokenization)

## Results

The project aims to compare the performance of BERT, T5, BoW, and TF-IDF on text summarization and question answering:

- **BERT** and **T5** are expected to outperform the traditional methods

## Setup and Installation

### Prerequisites
Ensure that Python 3.7 or higher is installed on your machine.

