# Topic research

- [Important remarks](#important-remarks)
- [An overview of time series forecasting models](#an-overview-of-time-series-forecasting-models)
  - [Prophet](#prophet)
  - [NNETAR](#nnetar)
  - [LSTM](#lstm)
- [Lecture](#lecture)
  - [Conv BiLSTM ✅](#conv-bilstm)
- [How to Develop LSTM Models for Time Series Forecasting](#how-to-develop-lstm-models-for-time-series-forecasting)
  - [Vanilla LSTM ✅](#vanilla-lstm)
  - [Stacked LSTM ✅](#stacked-lstm)
  - [Bidirectional LSTM ✅](#bidirectional-lstm)
  - [CNN LSTM ✅](#cnn-lstm)
  - [ConvLSTM ✅](#convlstm)
  - [Encoder-Decoder Model ✅](#encoder-decoder-model)
- [Forecasting With Tensorflow: Comparing Model Architectures](#forecasting-with-tensorflow-comparing-model-architectures)
  - [FNN ✅](#fnn)
  - [CNN ✅](#cnn)
  - [LSTM with two LSTM layers ✅](#lstm-with-two-lstm-layers)
  - [CNN + LSTM + FN ✅](#cnn--lstm--fn)
  - [CNN + LSTM + FN with a skip connection ✅](#cnn--lstm--fn-with-a-skip-connection)
  - [Results](#results)
- [Other interesting approaches](#other-interesting-approaches)
  - [Residual connections](#residual-connections)
  - [Seq2Seq LSTM with Luong Attention ✅](#seq2seq-lstm-with-luong-attention)
  - [Multivariate Time Series Forecasting with Transformers](#multivariate-time-series-forecasting-with-transformers)
  - [LSTMs with Self-Attention ✅](#lstms-with-self-attention)
  - [LSTMs with Multi-Head Attention ✅](#lstms-with-multi-head-attention)
  - [Autocorrelation](#autocorrelation)
  - [Moving Average](#moving-average)
  - [Decomposition](#decomposition)

## Important remarks

- **Warning:** do not shuffle the time-series data while you are preparing the test and train sets. Thus, avoid using *scikit-learn* cross-validation or k-fold tools since these functions are implicitly shuffling the data during the test and train split process.
- [Stackoverflow: stateful vs stateless LSTMs](https://stackoverflow.com/a/43090574)
    > when two sequences in two batches have connections (e.g. prices of one stock), you'd better use **stateful** mode, else (e.g. one sequence represents a complete sentence) you should use **stateless** mode.
- [Tensorflow documentation: LSTM layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
  - **return_state:** Boolean. Whether to return the last state in addition to the output. Default: **False**.
  - **stateful:** Boolean (default **False**). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
- [When to use GRU over LSTM?](https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm)
  - > **GRUs train faster** and perform better than LSTMs on **less training data** if you are doing language modeling (not sure about other tasks).
  - > **LSTMs** should in theory **remember longer sequences** than GRUs and outperform them in tasks requiring modeling long-distance relations.
- **Data preparation** consists of: *cleaning*, *scaling*, *feature extraction*, and *splitting*.
- **Autoregression** is done via setting *telescope=1* and appending the predictions to the previous ones inside a loop. 👍

## An overview of time series forecasting models

- *Source:* [An overview of time series forecasting models](https://towardsdatascience.com/an-overview-of-time-series-forecasting-models-a2fa7a358fcb)

**Traditional methods for time series forecasting like ARIMA has its limitation as it can only be used for univariate data and one step forecasting.**

### Prophet

> Prophet is another forecasting model which allows to deal with multiple **seasonalities**. It is an open source software released by Facebook’s Core Data Science team. The model fitting is framed as a curve-fitting exercise, therefore it does not explicitly take into account the temporal dependence structure in the data. This also allows to have irregularly spaced observations. There are two options for trend time series: a saturating growth model, and a piecewise linear model. The multi-period seasonality model relies on Fourier series. The effect of known and custom holydays can be easily incorporated into the model. For a complete introduction to Prophet model, see [Prophet](https://facebook.github.io/prophet/).

- [Stackoverflow: FB Prophet for multivariate multi-step forecasting](https://stackoverflow.com/questions/54544285/is-it-possible-to-do-multivariate-multi-step-forecasting-using-fb-prophet)

### NNETAR

> The NNETAR model is a fully connected neural network. The acronym stands for Neural NETwork AutoRegression. The NNETAR model takes in input the last elements of the sequence up to time t and outputs the forecasted value at time t+1. To perform multi-steps forecasts the network is applied iteratively. In presence of seasonality, the input may include also the seasonally lagged time series. For a complete introduction to NNETAR models, see [Neural network models](https://otexts.com/fpp2/nnetar.html).

- [Stackoverflow: Do Python have a model which is similar to nnetar in R's package forecast?](https://stackoverflow.com/a/67718775)
  - > Any NN model that uses 1 or more hidden layers is a multi-layer perceptron model, and for that case it is trivial to make it extendable to N layers. So any library that you pick will support it. My guess for you not picking a complex library like pytorch/Tensorflow is its size.

### LSTM

> LSTM models can be used to forecast time series (as well as other Recurrent Neural Networks). LSTM is an acronym that stands for Long-Short Term Memories. The state of a LSTM network is represented through a state space vector. This technique allows to **keep tracks of dependencies** of new observations with past ones (even very far ones). Generally speaking, **LSTMs are complex models and they are rarely used for predicting a single time-series, because they require a large amount of data to be estimated.** However, they are commonly used when predictions are needed for a large number of time-series (check [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)).

## Lecture

### Conv BiLSTM

```py
input_layer = tfkl.Input(shape=input_shape, name='Input')
convlstm = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True))(input_layer)
convlstm = tfkl.Conv1D(128, 3, padding='same', activation='relu')(convlstm)
convlstm = tfkl.MaxPool1D()(convlstm)
convlstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True))(convlstm)
convlstm = tfkl.Conv1D(256, 3, padding='same', activation='relu')(convlstm)
convlstm = tfkl.GlobalAveragePooling1D()(convlstm)
convlstm = tfkl.Dropout(.5)(convlstm)

# In order to predict the next values for more than one sensor,
# we can use a Dense layer with a number given by telescope*num_sensors,
# followed by a Reshape layer to obtain a tensor of dimension 
# [None, telescope, num_sensors]
dense = tfkl.Dense(output_shape[-1]*output_shape[-2], activation='relu')(convlstm)
output_layer = tfkl.Reshape((output_shape[-2],output_shape[-1]))(dense)
output_layer = tfkl.Conv1D(output_shape[-1], 1, padding='same')(output_layer)

# Connect input and output through the Model class
model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

# Compile the model
model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam(), metrics=['mae'])
```

## How to Develop LSTM Models for Time Series Forecasting

- *Source:* [How to Develop LSTM Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)

> The objective of this tutorial is to provide standalone examples of each model on each type of time series problem as a template that you can copy and adapt for your specific time series forecasting problem.

**Multivariate time series** data means data where there is more than one observation for each time step.

Any of the varieties of LSTMs in the previous section can be used, such as a Vanilla, Stacked, Bidirectional, CNN, or ConvLSTM model.

A time series forecasting problem that requires a prediction of multiple time steps into the future can be referred to as multi-step time series forecasting. Specifically, these are problems where the forecast horizon or interval is more than one time step.

### Vanilla LSTM

#### Model

```py
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
```

### Stacked LSTM

#### Model

```py
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
```

### Bidirectional LSTM

#### Model

```py
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
```

### CNN LSTM

> A convolutional neural network, or CNN for short, is a type of neural network developed for working with two-dimensional image data.
>
> The CNN can be very effective at automatically extracting and learning features from one-dimensional sequence data such as univariate time series data.
>
> A CNN model can be used in a hybrid model with an LSTM backend where the CNN is used to interpret subsequences of input that together are provided as a sequence to an LSTM model to interpret. This hybrid model is called a CNN-LSTM.

> We want to reuse the same CNN model when reading in each sub-sequence of data separately.
>
> This can be achieved by wrapping the entire CNN model in a **TimeDistributed** wrapper that will apply the entire model once per input, in this case, once per input subsequence.

#### Model

```py
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                        input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
```

#### Another model

- *Source:* [How To Do Multivariate Time Series Forecasting Using LSTM](https://analyticsindiamag.com/how-to-do-multivariate-time-series-forecasting-using-lstm/)

```py
lstm_model = tf.keras.models.Sequential([
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True), 
                                input_shape=x_train.shape[-2:]),
     tf.keras.layers.Dense(20, activation='tanh'),
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
     tf.keras.layers.Dense(20, activation='tanh'),
     tf.keras.layers.Dense(20, activation='tanh'),
     tf.keras.layers.Dropout(0.25),
     tf.keras.layers.Dense(units=horizon),
 ])
 lstm_model.compile(optimizer='adam', loss='mse')
```

### ConvLSTM

> A type of LSTM related to the CNN-LSTM is the ConvLSTM, where the convolutional reading of input is built directly into each LSTM unit.
>
> The ConvLSTM was developed for reading two-dimensional spatial-temporal data, but can be adapted for use with univariate time series forecasting.

#### Model

```py
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', 
                            input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
```

### Encoder-Decoder Model

> A model specifically developed for forecasting variable length output sequences is called the Encoder-Decoder LSTM.
>
> The model was designed for prediction problems where there are both input and output sequences, so-called sequence-to-sequence, or seq2seq problems, such as translating text from one language to another.
>
> This model can be used for multi-step time series forecasting.
>
> As its name suggests, the model is comprised of two sub-models: the encoder and the decoder.
>
> The encoder is a model responsible for reading and interpreting the input sequence. The output of the encoder is a fixed length vector that represents the model’s interpretation of the sequence. **The encoder is traditionally a Vanilla LSTM model, although other encoder models can be used such as Stacked, Bidirectional, and CNN models.**

> We can use the same output layer or layers to make each one-step prediction in the output sequence. This can be achieved by wrapping the output part of the model in a **TimeDistributed** wrapper.

#### Model

```py
model = Sequential()
# encoder
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
#decoder
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
```

#### Another model

- *Source:* [Multivariate Multi-step Time Series Forecasting using Stacked LSTM sequence to sequence Autoencoder in Tensorflow 2.0 / Keras](https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/)

```py
# n_features ==> no of features at each timestep in the data.

encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]

decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])

decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)

model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
```

***Note:** The results vary with respect to the dataset. If we stack more layers, it may also lead to overfitting. So the number of layers to be stacked acts as a hyperparameter.*

## Forecasting With Tensorflow: Comparing Model Architectures

- *Source:* [Kaggle: Multi-Variate Time Series Forecasting Tensorflow](https://www.kaggle.com/nicholasjhana/multi-variate-time-series-forecasting-tensorflow/notebook)

### FNN

```py
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(n_steps, n_features)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_horizon)
], name='dnn')
    
loss=tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(lr=lr)

model.compile(loss=loss, optimizer='adam', metrics=['mae'])
```

### CNN

```py
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu', input_shape=(n_steps,n_features)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_horizon)
], name="CNN")

loss= tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(lr=lr)

model.compile(loss=loss, optimizer='adam', metrics=['mae'])
```

### LSTM with two LSTM layers

```py
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(72, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
    tf.keras.layers.LSTM(48, activation='relu', return_sequences=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_horizon)
], name='lstm')

loss = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(lr=lr)

model.compile(loss=loss, optimizer='adam', metrics=['mae'])
```

### CNN + LSTM + FN

```py
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu', input_shape=(n_steps,n_features)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.LSTM(72, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(48, activation='relu', return_sequences=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_horizon)
], name="lstm_cnn")

loss = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(lr=lr)

model.compile(loss=loss, optimizer='adam', metrics=['mae'])
```

### CNN + LSTM + FN with a skip connection

```py
inputs = tf.keras.layers.Input(shape=(n_steps,n_features), name='main')
    
conv1 = tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu')(inputs)
max_pool_1 = tf.keras.layers.MaxPooling1D(2)(conv1)
conv2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(max_pool_1)
max_pool_2 = tf.keras.layers.MaxPooling1D(2)(conv2)
lstm_1 = tf.keras.layers.LSTM(72, activation='relu', return_sequences=True)(max_pool_2)
lstm_2 = tf.keras.layers.LSTM(48, activation='relu', return_sequences=False)(lstm_1)
flatten = tf.keras.layers.Flatten()(lstm_2)

skip_flatten = tf.keras.layers.Flatten()(inputs)

concat = tf.keras.layers.Concatenate(axis=-1)([flatten, skip_flatten])
drop_1 = tf.keras.layers.Dropout(0.3)(concat)
dense_1 = tf.keras.layers.Dense(128, activation='relu')(drop_1)
drop_2 = tf.keras.layers.Dropout(0.3)(dense_1)
output = tf.keras.layers.Dense(n_horizon)(drop_2)

model = tf.keras.Model(inputs=inputs, outputs=output, name='lstm_skip')

loss = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(lr=lr)

model.compile(loss=loss, optimizer='adam', metrics=['mae'])
```

### Results

The lower the score, the better the model.

|           | mae      | error_mw    |
| --------- | -------- | ----------- |
| dnn       | 0.117158 | 3363.850678 |
| cnn       | 0.081110 | 2328.828899 |
| lstm      | 0.085401 | 2452.055694 |
| lstm_cnn  | 0.083816 | 2406.540358 |
| lstm_skip | 0.089378 | 2566.219683 |

## Other interesting approaches

### Residual connections

- *Source:* [Tensorflow: Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series#advanced_residual_connections)

> The Baseline model from earlier took advantage of the fact that the sequence doesn't change drastically from time step to time step. Every model trained in this tutorial so far was randomly initialized, and then had to learn that the output is a a small change from the previous time step.
>
> While you can get around this issue with careful initialization, it's simpler to build this into the model structure.
>
> It's common in time series analysis to build models that instead of predicting the next value, predict how the value will change in the next time step. Similarly, residual networks — or **ResNets** — in deep learning refer to architectures where each layer adds to the model's accumulating result.
>
> That is how you take advantage of the knowledge that the change should be small.

#### Model

```py
class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta

residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small.
        # Therefore, initialize the output layer with zeros.
        kernel_initializer=tf.initializers.zeros())
]))
```

### Seq2Seq LSTM with Luong Attention

- *Source:* [Building Seq2Seq LSTM with Luong Attention in Keras for Time Series Forecasting](https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb)

#### Simple Seq2Seq LSTM Model

```py
input_train = Input(shape=(X_input_train.shape[1], X_input_train.shape[2]-1))
output_train = Input(shape=(X_output_train.shape[1], X_output_train.shape[2]-1)

encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
    n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, 
    return_sequences=False, return_state=True)(input_train)

encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
decoder = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, return_state=False, return_sequences=True)(
    decoder, initial_state=[encoder_last_h1, encoder_last_c])

out = TimeDistributed(Dense(output_train.shape[2]))(decoder)

model = Model(inputs=input_train, outputs=out)
opt = Adam(lr=0.01, clipnorm=1)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
```

#### Seq2Seq LSTM Model with Luong Attention

```py
input_train = Input(shape=(X_input_train.shape[1], X_input_train.shape[2]-1))
output_train = Input(shape=(X_output_train.shape[1], X_output_train.shape[2]-1))

encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
    n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, 
    return_state=True, return_sequences=True)(input_train)

encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
decoder_stack_h = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
    return_state=False, return_sequences=True)(
    decoder_input, initial_state=[encoder_last_h, encoder_last_c])

attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
attention = Activation('softmax')(attention)

context = dot([attention, encoder_stack_h], axes=[2,1])
context = BatchNormalization(momentum=0.6)(context)

decoder_combined_context = concatenate([context, decoder_stack_h])

out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)

model = Model(inputs=input_train, outputs=out)
opt = Adam(lr=0.01, clipnorm=1)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
```

### Multivariate Time Series Forecasting with Transformers

- *Source:* [Multivariate Time Series Forecasting with Transformers](https://towardsdatascience.com/multivariate-time-series-forecasting-with-transformers-384dc6ce989b)

### LSTMs with Self-Attention

- *Source:* [Package: Keras Self-Attention](https://pypi.org/project/keras-self-attention/)

#### Example model

```py
inputs = keras.layers.Input(shape=(None,))
embd = keras.layers.Embedding(input_dim=32,
                              output_dim=16,
                              mask_zero=True)(inputs)
lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=16,
                                                    return_sequences=True))(embd)
att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(lstm)
dense = keras.layers.Dense(units=5, name='Dense')(att)
model = keras.models.Model(inputs=inputs, outputs=[dense])
model.compile(
    optimizer='adam',
    loss={'Dense': 'sparse_categorical_crossentropy'},
    metrics={'Dense': 'categorical_accuracy'},
)
```

### LSTMs with Multi-Head Attention

- *Source:* [Package: Keras Multi-Head](https://pypi.org/project/keras-multi-head/)

#### Example model

```py
model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=5, output_dim=3, name='Embed'))
model.add(MultiHead(
    layer=keras.layers.Bidirectional(keras.layers.LSTM(units=16), name='LSTM'),
    layer_num=5,
    reg_index=[1, 4],
    reg_slice=(slice(None, None), slice(32, 48)),
    reg_factor=0.1,
    name='Multi-Head-Attention',
))
model.add(keras.layers.Flatten(name='Flatten'))
model.add(keras.layers.Dense(units=2, activation='softmax', name='Dense'))
model.build()
```

### Autocorrelation

- *Source:* [A Gentle Introduction to Autocorrelation and Partial Autocorrelation](https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/)

> Statistical correlation summarizes the strength of the relationship between two variables.
>
> We can assume the distribution of each variable fits a Gaussian (bell curve) distribution. If this is the case, we can use the Pearson’s correlation coefficient to summarize the correlation between the variables.
>
> The Pearson’s correlation coefficient is a number between -1 and 1 that describes a negative or positive correlation respectively. A value of zero indicates no correlation.
>
> We can calculate the correlation for time series observations with observations with previous time steps, called lags. Because the correlation of the time series observations is calculated with values of the same series at previous times, this is called a serial correlation, or an **autocorrelation**.
>
> A plot of the autocorrelation of a time series by lag is called the AutoCorrelation Function, or the acronym ACF. This plot is sometimes called a correlogram or an autocorrelation plot.
>
> A partial autocorrelation is a summary of the relationship between an observation in a time series with observations at prior time steps with the relationships of intervening observations removed.

### Moving Average

- *Source:* [Time Series forecasting using LSTM/ARIMA/Moving Average use case (Single/Multi-variate) with code](https://ranasinghiitkgp.medium.com/time-series-forecasting-using-lstm-arima-moving-average-use-case-single-multi-variate-with-code-5dd41e32d1fc)

> The predicted closing price for each day will be the average of a set of previously observed values. Instead of using the simple average, we will be using the moving average technique which uses the latest set of values for each prediction. In other words, for each subsequent step, the predicted values are taken into consideration while removing the oldest observed value from the set. Here is a simple figure that will help you understand this with more clarity.
>
> Given last ‘k’ values of temp-observations (only one feature <=> univariate), predict the next observation. Basically, Average the previous k values to predict the next value.
>
> ‘Average’ is easily one of the most common things we use in our day-to-day lives. For instance, finding the average temperature of the past few days to get an idea about today’s temperature.
>
> The predicted temp will be the average of a set of previously observed values. Instead of using the simple average, we will be using the moving average technique which uses the latest set of values for each prediction. In other words, for each subsequent step, the predicted values are taken into consideration while removing the oldest observed value from the set. Here is a simple figure that will help you understand this with more clarity.

### Decomposition

- *Source:* [Dynamic Mode Decomposition for Multivariate Time Series Forecasting](https://towardsdatascience.com/dynamic-mode-decomposition-for-multivariate-time-series-forecasting-415d30086b4b)

> Dynamic mode decomposition (DMD) is a data-driven dimensionality reduction algorithm developed by Peter Schmid in 2008 (paper published in 2010, see [1, 2]), which is similar to matrix factorization and principle component analysis (PCA) algorithms. Given a multivariate time series data set, DMD computes a set of dynamic modes in which each mode is associated with a fixed oscillation frequency and decay/growth rate. Due to the intrinsic temporal behaviors underlying each dynamic mode, DMD indeed differs from commonly used dimensionality reduction algorithms like PCA. DMD allows one to interpret temporal behaviors of data with physically meaningful modes. One important feature is that DMD is capable of performing **multivariate time series forecasting**.
>
> DMD has a strong connect with vector autoregressive (VAR) model.
