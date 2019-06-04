
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image


jsonfile =open('malaria_model.json', 'r')
loadedmodeljson = jsonfile.read()
jsonfile.close()

loadedmodel = model_from_json(loadedmodeljson)
loadedmodel.load_weights('malaria_model.h5')
loadedmodel.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

test_image = image.load_img('random3.png', target_size=(100,100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

result = loadedmodel.predict(test_image)
print(result[0][0])
if result[0][0] !=0:
    prediction = 'uninfected'
else :
    prediction ='infected'

print(prediction)


