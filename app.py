from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os


app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'C:/Users/HP/Desktop/SEM_Projects/FCV/Project/SkinCancer_App/skincancer_backend/Upload_Folder'


json_file = open(
    'C:/Users/HP/Desktop/SEM_Projects/FCV/Project/SkinCancer_v3_Models/resnetV2_model_v2/resnet_v2_model_v2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
loaded_model.load_weights(
    "C:/Users/HP/Desktop/SEM_Projects/FCV/Project/SkinCancer_v3_Models/resnetV2_model_v2/resnet_v2_model_v2_Weights.h5")
loaded_model.compile(
    optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
print("Loaded model from disk")


@app.route('/api/submit', methods=["POST"])
def index():
    image = request.files['image']  # get the uploaded image file

    if image is None:
        return 'No image file provided', 400
    newfilename = "testImage.jpg"  # get the file name
    # save the image to a file

    image.save(os.path.join(app.config['UPLOAD_FOLDER'], newfilename))
    # os.chmod('C:/Users/HP/Desktop/SEM_Projects/FCV/Project/SkinCancer_App/skincancer_backend/Upload_Folder/testImage.jpg', 666)

    # Set up the ImageDataGenerator object
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)

    # Load the image file
    image = tf.keras.preprocessing.image.load_img(
        'C:/Users/HP/Desktop/SEM_Projects/FCV/Project/SkinCancer_App/skincancer_backend/Upload_Folder/testImage.jpg', target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(
        image)  # convert the image to an array
    # add an extra dimension to the array
    image_array = np.expand_dims(image_array, axis=0)

    # Generate image data from the image array
    data = data_generator.flow(image_array)

    # Predict the class of image
    prediction = loaded_model.predict(data)
    pred_class = np.argmax(prediction, axis=1)

    result = {
        "predicted_Class": pred_class.tolist()
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
