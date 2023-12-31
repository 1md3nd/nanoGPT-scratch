# Nano GPT Language Model

This repository contains an implementation of a Nano GPT (Generative Pre-trained Transformer) Language Model using PyTorch. The Nano GPT model is a simplified version of the GPT architecture designed for educational purposes.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Generation](#generation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

The Nano GPT Language Model implemented here is a lightweight version of the GPT model, suitable for training on small datasets and understanding the basics of transformer architectures.

## Requirements

- Python 3.x
- PyTorch
- tqdm (for progress bars)

## Usage

Clone the repository and install the required dependencies:

`
git clone https://github.com/1md3nd/nanoGPT-scratch.git
cd nanoGPT-scratch
pip install -r requirements.txt
`


## Model Architecture

The Nano GPT Language Model consists of the following components:

- **Embedding layers:** Token embeddings to represent words in the sequence.

- **Transformer Blocks:** Stacks of simplified transformer blocks with self-attention mechanisms.

- **Layer Normalization:** Normalization applied to the output of each transformer block.

- **Linear Layer:** A linear layer for predicting the next token in the sequence.

## Training

The model is trained on a provided text dataset (`Rabindranath.txt`). The training loop involves updating the model's parameters to minimize the cross-entropy loss.

To train the model, run:

` python train.py `

Adjust hyperparameters in the script or pass them as arguments to customize the training process.

## Evaluation

The model's performance is evaluated on both training and validation datasets. The evaluation includes estimating the loss over multiple iterations.

To evaluate the model, run:

`python evaluate.py`


## Generation

The trained model can generate new sequences of text by predicting the next token given an input sequence.

To generate text, run:

`python generate.py`

Adjust the input sequence and other parameters in the script to control the generated text.

## Results

After training, the model's performance, including the final training and validation losses, will be printed. Additionally, the generated text based on a specified input sequence will be displayed.

## Conclusion

This Nano GPT Language Model provides a simplified introduction to transformer architectures and can serve as a starting point for understanding more complex language models. Feel free to explore and modify the code for your own experiments and applications.

If you have any questions or suggestions, please reach out to [anurag.botmaster@outlook.com].

Happy coding!
