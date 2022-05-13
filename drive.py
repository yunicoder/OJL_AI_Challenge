import os
import argparse
import base64
import shutil
import socketio
import eventlet
import eventlet.wsgi

from PIL import Image
from flask import Flask
from datetime import datetime
from io import BytesIO

import numpy as np
from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Current car angle
        steering_angle = float(data["steering_angle"])
        # Current car throttle
        throttle = float(data["throttle"])
        # Current car speed
        speed = float(data["speed"])
        # Current central camera images
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # If path to the directory is specified, save images
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
            
        try:
            # resize to model input
            image = image.resize((model.input_shape[2], model.input_shape[1]))

            image = np.asarray(image) # PIL image to numpy array
            image = np.array([image])
            # Predicting angle
            steering_angle = float(model.predict(image, batch_size=1))

            # Controlling speed
            MAX_SPEED = 20
            MIN_SPEED = 8
            
            zero_steearing = 0.05
            if -zero_steearing < steering_angle < zero_steearing:
                steering_angle = 0
            
            throttle = 1.0
            
            if float(speed) > MAX_SPEED:
                throttle = -1.0
            elif float(speed) < MIN_SPEED:
                throttle = 1.0
            else:
                throttle = 0.2

            print('steering_angle:{} throttle:{} speed:{}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
        
    else:
        # Please do not edit this line!
        sio.emit('manual', data={}, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='h5 fileのパスを入力'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='image folder pass'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Create a new folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING")
    else:
        print("NOT RECORDING")

    # Activating WSGI server
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    # eventlet.wsgi.server(eventlet.listen(('', 55021)), app)
