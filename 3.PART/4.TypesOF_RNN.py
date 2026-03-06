"""
Types of RNN:


Many-to-Many RNN
e.g. Machine Translation, Video Classification.

Many-to-One RNN
e.g. Sentiment Analysis, Rating Prediction.


One-to-Many RNN 
e.g. Music Generation from a seed.


1. Simple RNN
2. LSTM
3. GRU

Description of each:

1. Simple RNN: The basic architecture where the output from the previous time step is fed as input to the current step. It works well for short-term dependencies but suffers from the vanishing gradient problem in longer sequences.
   - Example: Simple character-level text prediction or short-term weather forecasting.

2. LSTM (Long Short-Term Memory): An advanced RNN that uses a cell state and three gates (input, forget, and output) to control the flow of information, allowing it to learn and retain long-term dependencies.
   - Example: Machine translation, complex speech recognition, and long-form document summarization.

3. GRU (Gated Recurrent Unit): A simplified version of LSTM that combines the forget and input gates into a single update gate and merges the cell state and hidden state, making it more computationally efficient.
   - Example: Real-time sentiment analysis and sequence-to-sequence modeling on smaller datasets.

"""