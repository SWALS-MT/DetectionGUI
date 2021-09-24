from tkinter import *
from tkinter.font import Font
import tkinter.ttk as ttk
import threading
import time

import cv2
import PIL
from PIL import ImageTk
import torch

from modules.camera import CameraModule
from modules.rcnn_mod import *
from modules.keypoint_mod import *

class DetectionGUI():
    def __init__(self):
        self.image_loop_flag = False
        self.model_flag = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # window settings
        self.window = Tk()
        self.window.title("Detection GUI")
        self.window.geometry("1200x600")
        self.window.configure(bg="#242e43")
        self.window.resizable(width=False, height=False)

        # canvas settings
        self.base_canvas = Canvas(self.window, width=1200, height=800, bg="#242e43")
        self.base_canvas.place(x=0, y=0)
        self.base_canvas.create_rectangle(700, 80, 1200, 600, fill="#1a212f")
        self.canvas1 = Canvas(self.window, width=640, height=480, bg="black")
        self.canvas1.place(x=30, y=100)

        # button settings
        self.button1 = Button(self.window, text="ON", command=self.image_loop_on)
        self.button1.place(x=950, y=100)
        button_f = Font(family='Segoe UI Black', size=24)
        self.base_canvas.create_text(730, 85, anchor=NW, font=button_f, text="Camera", fill="white")
        
        self.button2 = Button(self.window, text="OFF", command=self.image_loop_off)
        self.button2.place(x=1000, y=100)
        self.button2["state"] = DISABLED
        
        self.button3 = Button(self.window, text="ON", command=self.maskrcnn_on)
        self.button3.place(x=1050, y=150)
        self.button3["state"] = DISABLED
        self.base_canvas.create_text(730, 138, anchor=NW, font=button_f, text="Mask R-CNN", fill="white")
        
        self.button4 = Button(self.window, text="OFF", command=self.maskrcnn_off)
        self.button4.place(x=1100, y=150)
        self.button4["state"] = DISABLED

        self.button5 = Button(self.window, text="ON", command=self.fasterrcnn_on)
        self.button5.place(x=1050, y=200)
        self.button5["state"] = DISABLED
        self.base_canvas.create_text(730, 188, anchor=NW, font=button_f, text="Faster R-CNN", fill="white")
        
        self.button6 = Button(self.window, text="OFF", command=self.fasterrcnn_off)
        self.button6.place(x=1100, y=200)
        self.button6["state"] = DISABLED

        self.button7 = Button(self.window, text="ON", command=self.keypointrcnn_on)
        self.button7.place(x=1050, y=250)
        self.button7["state"] = DISABLED
        
        self.button8 = Button(self.window, text="OFF", command=self.keypointrcnn_off)
        self.button8.place(x=1100, y=250)
        self.button8["state"] = DISABLED
        self.base_canvas.create_text(730, 238, anchor=NW, font=button_f, text="Keypoint R-CNN", fill="white")

        # logo
        logo_img = PIL.Image.fromarray(cv2.cvtColor(cv2.imread("./images/mylogo_trans_white_square.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA))
        logo_img = ImageTk.PhotoImage(logo_img.resize((70, 70)).convert('RGBA'))
        self.base_canvas.create_image(60, 40, image=logo_img)
        f = Font(family='Segoe UI Black', size=32)
        self.base_canvas.create_text(250, 45, font=f, text="Detection GUI", fill="white")
        # logo_txt = Label(self.window, text="Detection GUI", font=f, fg="white", bg="#242e43")
        # logo_txt.place(x=120, y=15)

        # lines
        #4a608a
        #1a212f
        self.base_canvas.create_line(0, 80, 1200, 80, width=4, fill="#4a608a")
        self.base_canvas.create_line(700, 140, 1200, 140)

        self.window.mainloop()

    def image_loop(self):
        # 画像取得
        img = self.camera_mod.get_image()
        if img is None:
            return

        # Mask RCNN mode
        if self.model_flag:
            img, out = self.detection_mod(img)

        img_pil = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.resize((640, 480))
        self.img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas1.create_image(320, 240, image=self.img_tk)
        if self.image_loop_flag:
            self.window.after(10, self.image_loop)
            
    def image_loop_on(self):
        self.button1["state"] = DISABLED
        self.button2["state"] = NORMAL
        self.button3["state"] = NORMAL
        self.button4["state"] = DISABLED
        self.button5["state"] = NORMAL
        self.button6["state"] = DISABLED
        self.button7["state"] = NORMAL
        self.button8["state"] = DISABLED

        self.image_loop_flag = True
        # camera
        self.camera_mod = CameraModule(mode='webcam')
        self.image_loop()
    
    def image_loop_off(self):
        self.image_loop_flag = False
        time.sleep(1)
        self.camera_mod.close()

        self.button1["state"] = NORMAL
        self.button2["state"] = DISABLED
        self.button3["state"] = DISABLED
        self.button4["state"] = DISABLED
        self.button5["state"] = DISABLED
        self.button6["state"] = DISABLED
        self.button7["state"] = DISABLED
        self.button8["state"] = DISABLED

    def maskrcnn_on(self):
        self.button3["state"] = DISABLED
        self.button4["state"] = NORMAL
        self.button5["state"] = DISABLED
        self.button6["state"] = DISABLED
        self.button7["state"] = DISABLED
        self.button8["state"] = DISABLED

        self.detection_mod = MaskRCNNModule(device=self.device, finetune=False)
        self.model_flag = True
    
    def maskrcnn_off(self):
        self.model_flag = False

        self.button3["state"] = NORMAL
        self.button4["state"] = DISABLED
        self.button5["state"] = NORMAL
        self.button6["state"] = DISABLED
        self.button7["state"] = NORMAL
        self.button8["state"] = DISABLED
    
    def fasterrcnn_on(self):
        self.button3["state"] = DISABLED
        self.button4["state"] = DISABLED
        self.button5["state"] = DISABLED
        self.button6["state"] = NORMAL
        self.button7["state"] = DISABLED
        self.button8["state"] = DISABLED

        self.detection_mod = FasterRCNNModule(device=self.device, finetune=False)
        self.model_flag = True
    
    def fasterrcnn_off(self):
        self.model_flag = False

        self.button3["state"] = NORMAL
        self.button4["state"] = DISABLED
        self.button5["state"] = NORMAL
        self.button6["state"] = DISABLED
        self.button7["state"] = NORMAL
        self.button8["state"] = DISABLED

    def keypointrcnn_on(self):
        self.button3["state"] = DISABLED
        self.button4["state"] = DISABLED
        self.button5["state"] = DISABLED
        self.button6["state"] = DISABLED
        self.button7["state"] = DISABLED
        self.button8["state"] = NORMAL

        self.detection_mod = KeypointRCNNModule(device=self.device)
        self.model_flag = True
    
    def keypointrcnn_off(self):
        self.model_flag = False

        self.button3["state"] = NORMAL
        self.button4["state"] = DISABLED
        self.button5["state"] = NORMAL
        self.button6["state"] = DISABLED
        self.button7["state"] = NORMAL
        self.button8["state"] = DISABLED

if __name__ == '__main__':
    DetectionGUI()