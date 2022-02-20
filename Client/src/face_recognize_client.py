import os
from utils import *
import time
import dlib
import numpy as np
import imutils
from imutils.video import VideoStream
import f_detector

# instancio detector
detector = f_detector.eye_blink_detector()
IMAGE_DIR = r"E:/20211/DATK/1.1Linhduong/face-recognize-with-flask-master/Client/images"

def _input_name():
    print('Enter person name(no space): ')
    name = str(input())
    # return name
    if name:
        return name
    else:
        print('Invalid Input')
        del name
        _input_name()


# Video Web Face Recognizer
class VideoWFR:
    def __init__(self, sender):
        self.sender = sender

    sum_recognition = 0
    sum_unrecognition = 0

    #add image in directory
    def process_many_image(self, image_url):
        d_name = 'AI face recognition'
        detection_num = 0
        img = cv2.imread("{}//{}".format(IMAGE_DIR, image_url))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rectangles = detector.detector_faces(gray, 0)
        boxes_face = f_detector.convert_rectangles2array(rectangles, img)
        f_count = len(boxes_face)
        detection_num += f_count

        if len(boxes_face) != 0:
            for face in boxes_face:
                list_point = face.tolist()
                print(face)
                crop_img = img[list_point[1]:list_point[3], list_point[0]:list_point[2]]  # frame[x:x1, y:y1]
                rects = []
                for rectangle in rectangles:
                    rects.append(rectangle)
                cv2.imshow(d_name, crop_img)
                cv2.waitKey(1)
                self._send_request(crop_img)
                cv2.destroyAllWindows()

    def _send_request(self, crop_img):
        image_url = "images/image_{}.png".format(len(os.listdir('images')) + 1)
        cv2.imwrite(image_url, crop_img)
        response = self.sender.send_image_recognize(crop_img)
        print(response["message"])
        if response["message"] == "RECOGNIZED":
            name = response["name"]
            percent = int(response["percent"])
            if save_path is not None:
                ps = name + "_" + ("%03d" % percent) + ".png"
                ps = os.path.join(save_path, ps)
                cv2.imwrite(ps, crop_img)
            print(response["name"] + ": " + response["percent"])
            self.sum_recognition += 1
            print(self.sum_recognition)

            #check acc
        else:
            self.sum_unrecognition += 1
            print(self.sum_unrecognition)

            #add name to database
            # name = _input_name()
            # print("name", name)
            # res = self.sender.send_image_create(name=name, img=crop_img)
            # print(res)

    #video stream, face anti spoofing
    def process_stream_video(self):
        vs = VideoStream(src=0).start()
        COUNTER = 0
        TOTAL = 0
        while True:
            frame = vs.read()
            frame = cv2.flip(frame, 1)
            star_time = time.time()
            frame = imutils.resize(frame, width=720)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('image', frame)
            cv2.waitKey(1)
            # detectar_rostro
            rectangles = detector.detector_faces(gray, 0)
            boxes_face = f_detector.convert_rectangles2array(rectangles, frame)
            if len(boxes_face) != 0:
                boxes_face = np.expand_dims(boxes_face[0], axis=0)
                list_point = boxes_face[0].tolist()
                crop_img = frame[list_point[1]:list_point[3], list_point[0]:list_point[2]]  # frame[x:x1, y:y1]
                rects = []
                for rectangle in rectangles:
                    rects.append(rectangle)

                if TOTAL >= 3:
                    self._send_request(crop_img)
                    TOTAL = 0

                # blinks_detector
                COUNTER, TOTAL = detector.eye_blink(gray, rects[0], COUNTER, TOTAL)
                # agregar bounding box
                img_post = f_detector.bounding_box(frame, boxes_face, ['blinks: {}'.format(TOTAL)])
            else:
                img_post = frame

            end_time = time.time() - star_time
            FPS = 1 / end_time
            cv2.putText(img_post, f"FPS: {round(FPS, 3)}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.imshow('face detection', img_post)
            cv2.waitKey(1)


if __name__ == "__main__":
    host = "http://127.0.0.1"
    port = 5000

    # Video Web recognition 
    save_path = r"rec"
    sender = ImgSend(host, port, True)
    vr = VideoWFR(sender)

    # selection image in directory
    path_img = "E:/20211/DATK/1.1Linhduong/face-recognize-with-flask-master/Client/images"
    files = os.listdir(path_img)
    for url_img in files:
        vr.process_many_image(url_img)

    #vr.process_stream_video()\
