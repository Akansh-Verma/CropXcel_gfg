from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.utils import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt

class_dict = ["Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Septoria_leaf_spot", "Tomato___healthy", "Tomato___Late_blight", "Tomato___Tomato_mosaic_virus",
              "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Early_blight", "Tomato___Bacterial_spot", "Tomato___Target_Spot", "Tomato___Leaf_Mold"]


class Predict:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, image_path):
        # Load the image
        img = load_img(image_path, target_size=(224, 224))

        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        preds = self.model.predict(x)
        pred_idx = np.argmax(preds, axis=1)[0]
        prediction = class_dict[pred_idx]

        # Convert image to NumPy array
        img_array = np.array(img)

        return prediction, img_array


# # load the model we saved
# model_vgg16 = load_model(
#     'D:/GithubDesktop/CropXcel_gfg/cropxcel/prediction/CropXcel.h5')

# # Load the image
# img = load_img(
#     'D:/GithubDesktop/CropXcel_gfg/cropxcel/prediction/new_tomato.jpg', target_size=(224, 224))

# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)

# preds = model_vgg16.predict(x)
# pred_idx = np.argmax(preds, axis=1)[0]
# preds = class_dict[pred_idx]

# print(preds)
# # Convert image to NumPy array
# img_array = np.array(img)
# plt.imshow(img_array)
# plt.show()
# plt.clf()
