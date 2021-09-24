import numpy as np
import cv2

class CameraModule():
    def __init__(self, mode='webcam', video_path=None, img_path=None):
        self.mode = mode
        self.video_path = video_path
        self.img_path = img_path

        self.img = np.zeros((224, 224, 3), dtype=np.uint8)
        self.cap = None
        self._select_cap()
    
    def _select_cap(self):
        if self.mode == 'webcam':
            self.cap = cv2.VideoCapture(0)
        elif self.mode == 'video':
            if self.video_path is None:
                self.mode = 'image'
            else:
                self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = None
    
    def get_image(self):
        try:
            if self.mode == 'webcam':
                ret, frame = self.cap.read()
                self.img = frame
            elif self.mode == 'video':
                ret, frame = self.cap.read()
            else:    
                frame = cv2.imread(self.img_path)
        except:
            frame = self.img
        
        return frame
    
    def set_video_path(self, video_path):
        self.video_path = video_path
    
    def set_img_path(self, img_path):
        self.img_path = img_path
    
    def close(self):
        if self.cap is not None:
            self.cap.release()