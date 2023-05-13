from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.utils import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt

# class_dict = ["Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Septoria_leaf_spot", "Tomato___healthy", "Tomato___Late_blight", "Tomato___Tomato_mosaic_virus",
              #"Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Early_blight", "Tomato___Bacterial_spot", "Tomato___Target_Spot", "Tomato___Leaf_Mold"]
# class_dict = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
             # "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
# class_dict = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
# class_dict={'Tomato Bacterial spot': 0,
#             'Tomato Early blight': 1,
#             'Tomato Late blight': 2,
#             'Tomato Leaf Mold': 3,
#             'Tomato Septoria leaf spot': 4,
#             'Tomato Spider mites Two-spotted spider mite': 5,
#             'Tomato Target Spot': 6,
#             'Tomato Tomato Yellow Leaf Curl Virus': 7,
#             'Tomato Tomato mosaic virus': 8,
#             'Tomato healthy': 9}
class_dict = {
    0: "Tomato___Bacterial_spot",
    1: "Tomato___Early_blight",
    2: "Tomato___healthy",
    3: "Tomato___Late_blight",
    4: "Tomato___Leaf_Mold",
    5: "Tomato___Septoria_leaf_spot",
    6: "Tomato___Spider_mites Two-spotted_spider_mite",
    7: "Tomato___Target_Spot",
    8: "Tomato___Tomato_mosaic_virus",
    9: "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}

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
        # prediction = class_dict[pred_idx]
        if pred_idx in class_dict:
            prediction = class_dict[pred_idx]
        else:
            prediction = "Unknown class"

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
