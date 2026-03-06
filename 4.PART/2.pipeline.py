import tensorflow as tf

daily_sales_numbers = [21, 22, 23, -1, 25, -9, 27, 28, 29, 30]

# Create a tf.data.Dataset
tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales_numbers)

for sales in tf_dataset.as_numpy_iterator():
    print(sales)

for sales in tf_dataset.take(3):
    print(sales.numpy())

#remove negative values
tf_dataset = tf_dataset.filter(lambda x: x>0)

#convert to indian ruppee
tf_dataset = tf_dataset.map(lambda x: x*90)

#shuffle the dataset
tf_dataset = tf_dataset.shuffle(buffer_size=3)

# Batch the dataset
batch_size = 3
batched_dataset = tf_dataset.batch(batch_size)

# Iterate over the batched dataset
for batch in batched_dataset:
    print(batch.numpy())

# Pipeline in one single line
final_dataset = tf.data.Dataset.from_tensor_slices(daily_sales_numbers).filter(lambda x: x>0).map(lambda x: x*90).shuffle(buffer_size=3).batch(batch_size)

for batch in final_dataset:
    print(batch.numpy())


images_ds = tf.data.Dataset.list_files("images/*/*",shuffle=False)

for file in images_ds.take(3):
    print(file.numpy())

class_names = ["cat","dog"]

image_count = len(images_ds)
print(image_count)

train_size = int(image_count * 0.8)

#split into train and validation
train_ds = images_ds.take(train_size)
val_ds = images_ds.skip(train_size)

print(len(train_ds))
print(len(val_ds))

