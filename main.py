
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import time
import json
import numpy as np


with open("passwd.json","r") as passwd:
    pwd = json.load(passwd)
    
source = pwd["source1"] 
cap = cv2.VideoCapture(source)

app = FastAPI()
def gen_frames():
    while True:
 #   frame frame loop read the data of the camera
        success, frame = cap.read()
        if not success:
            break
        else:
 # Code data of each frame and store it in Memory
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
 # Use the yield statement to return the frame data as the responder, Content-Type is image / jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
@app.get('/rtsp')
def video_start():
 # By returning an image of a frame of frame, it reaches the purpose of watching the video. Multipart / X-Mixed-Replace is a single HTTP request - response mode, if the network is interrupted, it will cause the video stream to terminate, you must reconnect to recover
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)