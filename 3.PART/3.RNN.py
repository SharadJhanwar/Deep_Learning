"""
RNN (Recurrent Neural Network)

Need & Why:
Traditional neural networks assume inputs are independent, which fails for sequential data where context matters. RNNs are designed to handle data where the order of elements is critical, allowing the model to "remember" previous information to influence current processing.

How:
RNNs incorporate a feedback loop where the output of a hidden layer at a specific time step is fed back into the same layer as an input for the next time step. This creates a hidden state that acts as a memory of the sequence history.

Use Cases:
- auto complete
- NER (Named Entity Recognition)
- Language Translation
- Sentiment Analysis

3 Issues using ANN for sequence problems:
1. variable size of input/output neurons
2. Too much computation
3. No parameter sharing


Examples:
- Natural Language Processing (NLP): Machine translation, sentiment analysis, and text generation.
- Time Series Forecasting: Predicting stock prices or weather patterns.
- Speech Recognition: Converting audio signals into sequences of text.


RNN for NER:

Ex - Dhaval Loves baby yoda

Input (x):  [Dhaval] -> [Loves] -> [baby]  -> [yoda]
               |           |          |          |
RNN Cell:    [ h0 ]  ->  [ h1 ] ->  [ h2 ]  -> [ h3 ]
               |           |          |          |
Output (y):  [B-PER]     [  O  ]    [B-PER]    [I-PER]



"""