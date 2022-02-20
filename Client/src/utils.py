import time
import cv2
import requests
import json

print("cv2_version: {}".format(cv2.__version__))


class ImgSend:
    def __init__(self, host, port, debug_mode=False):
        self.host = host
        self.port = port
        self.url_recognize = host + ":" + str(port) + "/face-recognize"
        self.url_create = host + ":" + str(port) + "/create"
        self.dbg_mode = debug_mode
    
    def send_image_recognize(self, img):
        _, img_encoded = cv2.imencode(".png", img)

        data = img_encoded.tostring()
        headers = {"content-type": "image/png"}
        if self.dbg_mode:
            print("Sending request... ")
            #print(data)
        t1 = time.time()
        response = requests.post(self.url_recognize, data=data, headers=headers)
        t2 = time.time()
        dt = t2-t1
        if self.dbg_mode:
            print("Request processed: " + str(dt) + " sec")
        
        result = json.loads(response.text)
        return result

    def send_image_create(self, name, img):
        _, img_encoded = cv2.imencode(".png", img)
        headers = {"Content-Type": "application/json"}
        data = {
            "name": name,
            "image": img_encoded.tostring().decode('ISO-8859-1')
        }
        if self.dbg_mode:
            print("Sending request... ")
        t1 = time.time()
        response = requests.post(self.url_create, data=json.dumps(data), headers=headers)
        t2 = time.time()
        dt = t2 - t1
        if self.dbg_mode:
            print("Request processed: " + str(dt) + " sec")

        result = response.json()
        return result

