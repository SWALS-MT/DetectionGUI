import torch
import torchvision
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn

import cv2
import numpy as np
import time

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]

keypoints_colors = [    # BGR
    (235, 212, 247),
    (240, 219, 233),
    (204, 221, 255),
    (204, 255, 255),
    (219, 240, 240),
    (214, 245, 245),
    (209, 236, 250),
    (252, 252, 207),
    (233, 240, 219),
    (250, 209, 223),
    (235, 212, 247),
    (252, 252, 207),
    (242, 242, 217),
    (223, 250, 209),
    (209, 250, 250),
    (209, 250, 209),
    (227, 237, 222)
]

def draw_texts(img, texts, offset_x=10, offset_y=0, font_scale=0.7, thickness=2, color=(0, 0, 255)):
    h, w, c = img.shape

    texts = [texts] if type(texts) == str else texts

    for i, text in enumerate(texts):
        cv2.putText(img, text, (offset_x, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

class KeypointRCNNModule():
    def __init__(self, device):
        torch.cuda.empty_cache()
        self.device = device
        # load a model pre-trained on COCO
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def __call__(self, img):
        img_tr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tr = np.transpose(img_tr, (2, 0, 1)).astype(np.float32) / 255.0

        t = torch.from_numpy(img_tr).to(self.device)
        t = t.unsqueeze(0)

        t1 = time.time()
        with torch.no_grad():
            out = self.model(t)
        t2 = time.time()
        fps = 1 / (t2 - t1)

        # draw the pose
        img = self.draw_keypoints(out, img)

        draw_texts(img, 'fps: '+str(round(fps, 4)), offset_x=10, offset_y=50, color=(0, 255, 0))

        return img, out
        
    def draw_keypoints(self, output, img, conf=0.9):
        pnum = 0
        for i, keypoints in enumerate(output[0]['keypoints']):
            if output[0]['scores'][i] < conf:
                continue
            keypoints = keypoints[:, :2].detach().cpu().numpy().astype(np.int16)

            # draw the keypoints
            ip_list = []
            for ip, p in enumerate(keypoints):
                if output[0]['keypoints_scores'][i, ip] < 0:
                    continue
                cv2.circle(img, (p[0], p[1]), 8, keypoints_colors[ip], thickness=-1, lineType=cv2.FILLED)
                ip_list.append(ip)
            # draw the lines
            for edge in edges:
                if edge[0] not in ip_list or edge[1] not in ip_list:
                    continue
                p1 = (keypoints[edge[0], 0], keypoints[edge[0], 1])
                p2 = (keypoints[edge[1], 0], keypoints[edge[1], 1])
                cv2.line(img, p1, p2, (255, 255, 0), thickness=2)
            pnum += 1
        draw_texts(img, 'people: ' + str(pnum), offset_x=10, offset_y=20, color=(0, 255, 0))

        return img

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = KeypointRCNNModule(device=device)

    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        if img is None:
            continue
        img, out = mod(img)
        cv2.imshow("keypoints", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()