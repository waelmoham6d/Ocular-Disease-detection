from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

eyes_model=load_model(r"C:\Users\mwael\OneDrive\Desktop\home\course\Eyes\eyes_model.h5")

image_path=r'C:\Users\mwael\OneDrive\Desktop\home\course\Eyes\ODIR-5K\Training Images\4_left.jpg'

img=image.load_img(image_path,target_size=(224,224))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = eyes_model.predict(img_array)

predicted_class = np.argmax(prediction, axis=1)
real_pred=['N','D','G','C',"A","H",'M','O']
print(f"The image belongs to class: {prediction}")

print(f"The image belongs to class: {real_pred[predicted_class[0]]}")































































