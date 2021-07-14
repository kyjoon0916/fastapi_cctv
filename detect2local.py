import argparse
import time
import uvicorn

from time import sleep
from pathlib import Path
from fastapi import FastAPI, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import  StaticFiles
from detect import run
from rtsp_ import detect
import os
import tensorflow
import json
import cv2
import pytz 
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import pandas as pd
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

app = FastAPI()
# app.mount("static",StaticFiles(directory="static",name="static"))
templates = Jinja2Templates

# @app.get('/a')
# def main():
#     return StreamingResponse(detect("a"), media_type='multipart/x-mixed-replace; boundary=frame')
# @app.get('/b')
# def other():
#     return StreamingResponse(detect("b"), media_type='multipart/x-mixed-replace; boundary=frame')
# @app.get("/")
# async def main(request: Request):
#     result=1
#     return templates.TemplateResponse("templates/item.html",context={"request":request,"result":result})


@app.get('/a')
async def stream1():
    return StreamingResponse(run(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/b')
async def stream2():
    return StreamingResponse(detect("a"),  media_type='multipart/x-mixed-replace; boundary=frame')