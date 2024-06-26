# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import cv2 
from vlm import VLM 
from queue import Queue 
from api_server import FlaskServer 


response_dict = dict() 
overlay_response = ""
overlay_prompt = ""
alert = False 


def _draw_text(image, text, x, y, text_color, background_color):
    """Draw text on an image using OpenCV"""
    # Get text size
    font_scale = 1
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Create a filled rectangle as background for the text
    padding = 2
    cv2.rectangle(image, (x-padding, y), 
                  (x + text_width + padding, y + text_height + (baseline * 2)), background_color, -1)

    # Put text on the image
    cv2.putText(image, text, (x, y+text_height+baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def draw_lines(image, text, x, y, **kwargs):
    """Draw text on an image with word wrapping using OpenCV"""

    if text is None:
        return y

    text_color = kwargs.get("text_color", (255, 255, 255))  # Default white
    background_color = kwargs.get("background_color", (40,40,40))
    line_spacing = kwargs.get("line_spacing", 48)
    line_length = kwargs.get("line_length", 100)

    # Split text into words
    words = text.split()
    current_line = ""
    y_offset = y

    for word in words:
        if len(current_line) + len(word) <= line_length:
            current_line += word + " "
        else:
            _draw_text(image, current_line.strip(), x, y_offset, text_color, background_color)
            current_line = word + " "
            y_offset += line_spacing

    if current_line:
        _draw_text(image, current_line.strip(), x, y_offset, text_color, background_color)
        y_offset += line_spacing

    return y_offset


def vlm_callback(prompt, reply, **kwargs):
    global response_dict 
    global overlay_response
    global overlay_prompt 

    prompt_id = kwargs.get("prompt_id")
    overlay_prompt = prompt 
    overlay_response = reply
    response_dict[prompt_id] = reply 
     

def main(model_url, video_file, api_key, port, overlay=False, loop_video=False, hide_query=False):
    global response_dict 
    global overlay_response 
    global alert 

    cv2.namedWindow("Demo", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Demo", 1280, 720) 

    prompt_queue = Queue() 

    flask_server = FlaskServer(prompt_queue, response_dict, port=port)
    flask_server.start_flask()

    #open video file 
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    vlm = VLM(model_url, api_key, callback=vlm_callback)

    prompt = None 
    prompt_id = ""
    while True:
        #Get new frame
        ret, frame = cap.read() #Get new Frame 

        #Check frame return is valid 
        if not ret:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue 
            else:
                break 

        #Get new prompt 
        if not prompt_queue.empty():
            print("updating prompt")
            message = prompt_queue.get()
            prompt = message.data
            prompt_id = message.id 
            alert = True if message.type == "alert" else False 

        #VLM available, call on latest frame 
        if vlm.busy == False and prompt is not None:
            vlm(prompt, frame.copy(), prompt_id=prompt_id) #call on latest prompt 
            if not alert:
                prompt = None 
            
    
        #Output overlay if enabled
        if overlay:
            if not(hide_query and not alert):#if hide query is false then always overlay. If hide query is true then only overlay alerts. 
                y = 20
                y = draw_lines(frame, f"VLM Input: {overlay_prompt}", 20, y, text_color=(120,215,21), background_color=(40,40,40, 20))
                y = draw_lines(frame, f"VLM Response: {overlay_response}", 20, y, text_color=(255,255,255), background_color=(40,40,40, 20))
            cv2.imshow("Demo", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break 

    
    #clean up
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a model and a video file.")

    # Add the arguments
    parser.add_argument('--model_url',
                        type=str,
                        required=True,
                        help='The path to the model file')

    parser.add_argument('--video_file',
                        type=str,
                        required=True,
                        help='The path to the video file')
    
    parser.add_argument("--api_key", type=str, required=True, help="API Key")

    parser.add_argument("--port", type=int, required=False, default=5432, help="Flask Port")

    parser.add_argument("--overlay", action="store_true", help="Enable VLM overlay")

    parser.add_argument("--loop_video", action="store_true", help="Continuosly loop the video")

    parser.add_argument("--hide_query", action="store_true", help="only show alerts on the stream display.")

    # Execute the parse_args() method
    args = parser.parse_args()

    # Call the main function
    main(args.model_url, args.video_file, args.api_key, args.port, overlay=args.overlay, loop_video=args.loop_video, hide_query=args.hide_query)
