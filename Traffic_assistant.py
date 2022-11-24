import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import  pyttsx3


car =  pyttsx3.init()
voices = car.getProperty('voices')
car.setProperty('voice', voices[1].id)
car.setProperty('rate', 150)

def Speak(text):
    car.say(text)
    car.runAndWait()

def check(speed,result):
  if result in [0,1,2,3,4,5,6,7,8]:
    if result == 0:
      if speed > 20:
        Message = ('Slow down Speed limit is 20Km/h')
    elif result == 1:
      if speed > 30:
        Message = ('Slow down Speed limit is 30Km/h')
    elif result == 2:
      if speed > 50:
        Message = ('Slow down Speed limit is 50Km/h')
    elif result == 3:
      if speed > 60:
        Message = ('Slow down Speed limit is 60Km/h')
    elif result == 4:
      if speed > 70:
        Message = ('Slow down Speed limit is 70Km/h')
    elif result == 5:
      if speed > 80:
        Message = ('Slow down Speed limit is 80Km/h')
    elif result == 6:
      if speed > 80:
        Message = ('Slow down Speed limit is 80Km/h')
    elif result == 7:
      if speed > 100:
        Message = 'Slow down Speed limit is 100Km/h'
    elif result == 8:
      if speed > 120:
        Message = ('Slow down Speed limit 120Km/h')
    print(Message)
    Speak(Message + 'Your current speed is '+ str(speed) + 'Km/h')
    cv2.putText(image, Message, (0, 100 ), font, 1, (0, 0, 255),2)


classes = [ 'Speed limit 20km/h',
            'Speed limit 30km/h',
            'Speed limit 50km/h',
            'Speed limit 60km/h',
            'Speed limit 70km/h',
            'Speed limit 80km/h',
            'End of speed limit 80km/h',
            'Speed limit 100km/h',
            'Speed limit 120km/h',
            'No passing',
            'No passing veh over 3.5 tons',
            'Right-of-way at intersection',
            'Priority road',
            'Yield',
            'Stop',
            'No vehicles',
            'Veh > 3.5 tons prohibited',
            'No entry',
            'General caution',
            'Dangerous curve left',
            'Dangerous curve right',
            'Double curve',
            'Bumpy road',
            'Slippery road',
            'Road narrows on the right',
            'Road work',
            'Traffic signals',
            'Pedestrians',
            'Children crossing',
            'Bicycles crossing',
            'Beware of ice/snow',
            'Wild animals crossing',
            'End speed + passing limits',
            'Turn right ahead',
            'Turn left ahead',
            'Ahead only',
            'Go straight or right',
            'Go straight or left',
            'Keep right',
            'Keep left',
            'Roundabout mandatory',
            'End of no passing',
            'End no passing veh > 3.5 tons' ]


def identify(img):
  model_path = "model_krizz_4.h5"
  loaded_model = tf.keras.models.load_model(model_path)


  image = img
  image_fromarray = Image.fromarray(image, 'RGB')
  resize_image = image_fromarray.resize((30, 30))
  expand_input = np.expand_dims(resize_image,axis=0)
  input_data = np.array(expand_input)
  input_data = input_data/255

  pred = loaded_model.predict(input_data)
  result = pred.argmax()
  Message = classes[result] + ' sign detected'
  Speak(Message)
  print(Message)
  return(result)


# define a video capture object
cam = cv2.VideoCapture(0)

while True:

    success, image = cam.read()
    cv2.waitKey(1)
    speed = 900 #int(input("Enter your current speed :"))
    result = identify(image)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, str(classes[result]), (0, 50), font, 1, (0, 0, 0), 2)
    check(speed, result)
    cv2.imshow("image",image)
    print(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
# Destroy all the windows
cam.destroyAllWindows()