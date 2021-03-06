import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import cv2
import numpy as np
import time


def draw_texts(img, texts, offset_x=10, offset_y=0, font_scale=0.7, thickness=2, color=(0, 0, 255)):
    h, w, c = img.shape

    texts = [texts] if type(texts) == str else texts

    for i, text in enumerate(texts):
        cv2.putText(img, text, (offset_x, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)


def draw_masks(img, mask, thresh=0.5):
    mask = mask[0]
    mask = ((mask > thresh) * 255).astype(np.uint8)
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(img, [contour], -1, color=(0, 255, 255), thickness=2)
    except cv2.error:
        return


class FasterRCNNModule():
    def __init__(self, device, finetune=False, num_classes=2):
        """
        :args
        device: torch.device, CUDA or CPU
        :option
        finetune: If you set it True, you can select the number of the output classes
        num_classes: the number of output classes.
        """
        torch.cuda.empty_cache()
        self.device = device
        # load a model pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        if finetune:
            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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
        
        boxes = out[0]["boxes"].data.cpu().numpy()
        scores = out[0]["scores"].data.cpu().numpy()
        labels = out[0]["labels"].data.cpu().tolist()

        category = {0: 'background', 1: 'person', 2: 'traffic light', 3: 'train', 4: 'traffic sign', 5: 'rider',
                    6: 'car', 7: 'bike', 8: 'motor', 9: 'truck', 10: 'bus'}

        boxes = boxes[scores >= 0.5].astype(np.int32)  # Confidence threshold
        pnum = 0
        for i, box in enumerate(boxes):
            if labels[i] == 1:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), thickness=2)
                draw_texts(img, 'person '+str(round(scores[i], 3)), offset_x=box[0], offset_y=box[1])
                pnum += 1
            else:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), thickness=1)
        draw_texts(img, 'people: '+str(pnum), offset_x=10, offset_y=20, color=(0, 255, 0))
        draw_texts(img, 'fps: '+str(round(fps, 4)), offset_x=10, offset_y=50, color=(0, 255, 0))

        return img, out


class MaskRCNNModule():
    def __init__(self, device, finetune=False, num_classes=2):
        """
        :args
        device: torch.device, CUDA or CPU
        :option
        finetune: If you set it True, you can select the number of the output classes
        num_classes: the number of output classes.
        """
        torch.cuda.empty_cache()
        self.device = device
        # load a model pre-trained pre-trained on COCO
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        if finetune:
            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)

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

        boxes = out[0]["boxes"].data.cpu().numpy()
        scores = out[0]["scores"].data.cpu().numpy()
        labels = out[0]["labels"].data.cpu().tolist()
        masks = out[0]["masks"].data.cpu().numpy()
        # print(masks.shape)

        category = {0: 'background', 1: 'person', 2: 'traffic light', 3: 'train', 4: 'traffic sign', 5: 'rider',
                    6: 'car', 7: 'bike', 8: 'motor', 9: 'truck', 10: 'bus'}

        boxes = boxes[scores >= 0.5].astype(np.int32)  # Confidence threshold
        pnum = 0
        for i, box in enumerate(boxes):
            if labels[i] == 1:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), thickness=2)
                draw_texts(img, 'person ' + str(round(scores[i], 3)), offset_x=box[0], offset_y=box[1])
                draw_masks(img, masks[i])
                pnum += 1
            else:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), thickness=1)
        draw_texts(img, 'people: ' + str(pnum), offset_x=10, offset_y=20, color=(0, 255, 0))
        draw_texts(img, 'fps: ' + str(round(fps, 4)), offset_x=10, offset_y=50, color=(0, 255, 0))

        return img, out


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mod = FasterRCNNModule(device=device, finetune=False)
    mod = MaskRCNNModule(device=device, finetune=False)

    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        img_disp, out = mod(img)

        cv2.imshow("result", img_disp)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
