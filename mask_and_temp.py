import RPi.GPIO as GPIO
import time
import board
import busio
import adafruit_mlx90614
from Adafruit_CharLCD import Adafruit_CharLCD
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Set up servo motor
servo_pin = 18
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz (20 ms PWM period)

# Set up buzzer
buzzer_pin = 23
GPIO.setup(buzzer_pin, GPIO.OUT)

# Initialize LCD
lcd = Adafruit_CharLCD(rs=26, en=19, d4=13, d5=6, d6=5, d7=21, cols=16, lines=2)
lcd.clear()
lcd.message("WELCOME")
GPIO.output(buzzer_pin, GPIO.LOW)

# Display text on LCD
def display_lcd(message):
    lcd.clear()
    lcd.message(message)


# Initialize servo motor
def angle_to_duty_cycle(angle):
    return (angle / 18) + 2.5  # 2.5% to 12.5% duty cycle for 0 to 180 degrees

def set_angle(angle):
    duty_cycle = angle_to_duty_cycle(angle)
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.3)  # Allow time for the servo to move

# Initialize temperature sensor
def get_temperature():
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90614.MLX90614(i2c)
    temperature = mlx.object_temperature
    print(f"Temperature: {temperature} °C")
    return temperature

# Load face detector and mask detector
def load_models():
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    print("[INFO] loading face mask detector model...")
    maskNet = load_model("MaskDetector.h5")

    return faceNet, maskNet

# Detect faces and predict masks
def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame
    (h, w) = frame.shape[:2]

    # construct a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and get the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize lists to store faces, their locations, and predictions
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding box falls within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, resize it to 224x224, preprocess it, and add it to the faces list
            face = cv2.resize(frame[startY:endY, startX:endX], (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a prediction if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Main function
def main():
    try:
        pwm.start(0)

        # Load models
        faceNet, maskNet = load_models()

        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(0)

        time.sleep(2.0)

        while True:
            frame = vs.read()[1]
            frame = imutils.resize(frame, width=500)

            # Detect faces and predict masks
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask Detected" if mask > withoutMask else "No Mask Detected"
                if mask > withoutMask:
                    display_lcd("Mask Detected.")
                    # Measure body temperature
                    temperature = get_temperature()
                    # Act based on temperature and mask detection
                    if temperature < 37.5:              
                        set_angle(90)
                        time.sleep(10)
                        set_angle(0)
                        display_lcd("Normal Temperature.\n now go")
                    else:
                        GPIO.output(buzzer_pin, GPIO.HIGH)  # Turn on buzzer
                        time.sleep(5)  # Buzzer on for 5 seconds
                        GPIO.output(buzzer_pin, GPIO.LOW)  # Turn off buzzer
                else:
                    display_lcd("NO MASK.\n Plz STOP")

                color = (0, 255, 0) if label == "Mask Detected" else (0, 0, 255)

                cv2.putText(frame, label, (startX - 50, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Mask Detector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            time.sleep(1)  # Wait before the next iteration

    except KeyboardInterrupt:
        pass

    finally:
        pwm.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        vs.release()

if __name__ == "__main__":
    main()