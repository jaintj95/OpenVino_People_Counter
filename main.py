"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import queue
from collections import deque
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Predictions made when result is same in 4 consecutive frames
FRAME_SANITY = 4


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    return parser


def connect_mqtt():
    """
    Connects to the mqtt client and returns the client object
    """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def draw_boxes(frame, output, prob_t, width, height):
    """
    Draw bounding boxes around the detected objects in the frame
    
    Args:
        frame: frame from input feed
        output: inference output from the model
        prob_t: probability threshold
        
    Returns:
        frame: modified frame with bounding boxes
        current_count: number of people currently in the frame
    """
    
    current_count = 0
    
    for box in output[0][0]:
        conf = box[2]
        if conf >= prob_t:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            current_count += 1
            
    return frame, current_count
            

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    Args:
        args: Command line arguments parsed by `build_argparser()`
        client: MQTT client

    """
    # Initialise the inference engine
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the network model into the IE
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    input_shape = infer_network.get_input_shape()
    n, c, h, w = input_shape

    # Handle the input stream
    img_flag = False
    
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        img_flag = True
    
    capture = cv2.VideoCapture(args.input)
    capture.open(args.input)
    
    if img_flag:
        out = None
    else:
        out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (100,100))
    
    # log an error if incorrect input source is provided
    if not capture.isOpened():
        log.error("Given input source is unsupported")
        
    width_cap = capture.get(3)
    height_cap = capture.get(4)
        
    # Init variables
    frame_Counter = 0
    entry_time = 0
    current_count = 0
    prev_count = 0
    total_count = 0
    count_list = deque(maxlen=FRAME_SANITY) #doesn't grow beyond 4
        
    # Loop until stream is over
    while capture.isOpened():
        # Read the next frame
        flag, frame = capture.read()
        if not flag:
            break
        
        key_pressed = cv2.waitKey(60)
        
        # Pre-process the input image/frame
        proc_frame = cv2.resize(frame, (w, h))
        proc_frame = proc_frame.transpose((2, 0, 1))
        proc_frame = proc_frame.reshape(1, *proc_frame.shape) # add a batch dim
        
        # Initiate asynchronous inference for specified request
        infer_network.init_async_infer(proc_frame)
        start = time.time()
        frame_Counter += 1
        # Wait for the result of async request
        if infer_network.wait() == 0:
            end = time.time()
            time_diff = end - start
            
            # Get the results of the inference request
            result = infer_network.fetch_output()
            
            ### TODO: Extract any desired stats from the results ###
            frame, current_count = draw_boxes(frame, result, args.prob_threshold, width_cap, height_cap)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            inf_time_message = "Inference time: {:.3f}ms".format(time_diff * 1000)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 200, 10), 1)
            
#             if len(count_list) == FRAME_SANITY:
#                 count_list.pop()
            
            count_list.append(current_count)
            avg_count = sum(count_list)/4
            sane_count = int(np.ceil(avg_count))
            
            if frame_Counter % FRAME_SANITY == 0:
                if sane_count > prev_count:
                    entry_time = time.time()
                    total_count += (sane_count - prev_count)
                    client.publish("person", json.dumps({"total": total_count}))

                if sane_count < prev_count:
                    duration = int(time.time() - entry_time)
                    #prev_count = sane_count
                    client.publish("person/duration", json.dumps({"duration": duration}))
                
                client.publish("person", json.dumps({"count": sane_count}))
                prev_count = sane_count

            if key_pressed == 27:
                break

        # Send the frame to the FFMPEG server
        if not img_flag:
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()

        # Write an output image if `img_flag`
        if img_flag:
            cv2.imwrite('output_img.jpg', frame)
            
    
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    
    # Connect to the MQTT server
    client = connect_mqtt()
    
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()