"""
Tensorflow Input Pipeline:

Step 1: Build Data Pipeline 
Perform ETL (Extract, Transform, Load)

Step 2: Train the Model
model.fit(tf_dataset)

Dataset -> tf.data input pipeline -> Model

Benifits of Tensorflow input pipeline
1. Handle huge dataset by streaming from disk usin batching
2. Apply transformations to make dataset ready for model training

"""