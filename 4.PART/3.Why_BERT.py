"""
Issue With Word2Vec:

1. Word2Vec cannot handle the polysemy of words.
2. Word2Vec cannot handle the out-of-vocabulary words.

Solution:

BERT (Bidirectional Encoder Representations from Transformers)

How BERT solves the issue:

1. BERT can handle the polysemy of words.
2. BERT can handle the out-of-vocabulary words.

But how ?

1. BERT uses the concept of self attention mechanism .

What more BERT can do ?
2. BERT uses Masked Language Modeling (MLM) to capture bidirectional context by hiding certain words and predicting them.
3. BERT uses Next Sentence Prediction (NSP) to understand the relationship between two sentences.
4. BERT can be fine-tuned for a wide range of downstream tasks such as Sentiment Analysis, Question Answering, and Named Entity Recognition (NER).

complete BERT article:
http://jalammar.github.io/illustrated-bert/

BERT was trained on Wikipedia and BookCorpus using two approaches:
1. Masked Language Modeling (MLM)
2. Next Sentence Prediction (NSP)

Today Google search is powered by BERT.



"""