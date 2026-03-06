"""
NN cannot understand the word , it can only understand the numbers.

So we have to convert the words to numbers before training the model.

BUT HOW?

OPTION 1 : Convert the words to unique numbers using vocabulary.

OPTION 2 : Convert the words to numbers using OneHotEncoder.

     - NOt possible bcz computational complexity is high.

OPTION 3 : Convert the words to numbers using Word Embeddings.
  - WAYS:
    1. TF-IDF
    2. Word2Vec
    3. GloVe
    4. FastText

"""