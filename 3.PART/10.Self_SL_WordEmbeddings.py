"""
Self Supervised Learning Technique for Word Embedding:

1. Word2Vec
2. GloVe
3. FastText


1. Word2Vec: Developed by Google, it utilizes two primary architectures—Continuous Bag of Words (CBOW) and Skip-gram—to learn vector representations by predicting words based on their local context.

CBOW: Predicts the target word based on its surrounding context words.
Skip-gram: Predicts context words based on the target word.

2. GloVe (Global Vectors): Developed by Stanford, it aggregates global word-word co-occurrence statistics from a corpus, combining the advantages of global matrix factorization and local context window methods.

3. FastText: Developed by Facebook, it extends Word2Vec by treating each word as a bag of character n-grams. This allows the model to capture internal word structure and generate embeddings for out-of-vocabulary words.

"""