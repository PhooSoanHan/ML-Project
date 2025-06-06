import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model("garbage_classifier_model.keras")

# Defining class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# Load the test_images
img_path = "C:/Users/Phoo Soan Han/ML-Project/garbage-classification/test_images/test180.jpg"
img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class = class_names[np.argmax(score)]

print(f"This image most likely belong to '{predicted_class}' with a {100 * np.max(score):.2f}% confidence.")

# Show image
plt.imshow(img)
plt.title(predicted_class)
plt.axis('off')
plt.show()