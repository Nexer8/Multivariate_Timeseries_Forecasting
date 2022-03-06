# Multivariate Time Series Forecasting

Link to [Codalab competition](https://codalab.lisn.upsaclay.fr/competitions/621).

## Authors

- *[Aleksandra Krajnović](https://github.com/akrajnovic)*
- *[Iva Milojkovic](https://github.com/ivamilojkovic)*
- *[Mariusz Wiśniewski](https://github.com/Nexer8)*

## Description

The aim of this project to predict future samples of a multivariate time series. The goal is to design and implement forecasting models to learn how to exploit past observations in the input sequence to correctly predict the future.

## Data

The data can be found under the [link](https://drive.google.com/drive/folders/14YIaBj7Hm9wjqc8notvB0gW4V8PHO8mR?usp=sharing). The provided time series have a *uniform* sampling rate.

### Dataset Details

- Length of the time series (number of samples in the training set): 68528.
- Number of features: 7.
- Name of the features: *'Sponginess'*, *'Wonder level'*, *'Crunchiness'*, *'Loudness on impact'*, *'Meme creativity'*, *'Soap slipperiness'*, *'Hype root'*.

## Approaches

- Convolutional Bidirectional LSTM
- Convolutional Bidirectional LSTM using *Bahdanau's Attention*
- Convolutional Bidirectional LSTM using *Luong's Attention*
- Seq2Seq LSTM using *Luong's Attention*
- Stacked LSTM
- Stacked SCINet
