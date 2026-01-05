import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from model import MyModel
from handtrackermodule import handDetector
import numpy as np
import cv2

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

EPOCHS = 15
LR = 0.01
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

hands = handDetector(maxHands=1)
model = MyModel()
model = model.to(device=DEVICE)

optimizer = torch.optim.Adam(model.parameters(), LR)

from pathlib import Path
from PIL import Image

classes = {
    "dislike" : 0,
    "like" : 1
}
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tarDir):
        self.paths = list(Path(tarDir).glob("*/*.jpg"))
    def load_img(self, idx):
        img_path = self.paths[idx]
        return Image.open(img_path)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = self.load_img(idx)
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (640,640))
        label_name = self.paths[idx].parent.name
        label = torch.tensor(classes[label_name])
        labels = [0,0]
        labels[label]=1
        showHands = hands.findHands(img_np)
        landmarks = hands.findPosition(img_np)
        lms = []
        for x,y,z in landmarks:
            x = float(x)
            y = float(y)
            x/=640
            y/=400
            lms.append(x)
            lms.append(y)
            lms.append(z)
        #I did this so that if the mediapipe model doesnt work on an image, it atleast gives an output of same size instead of empty list
        if(len(lms)==0):
            lms = [0]*63
        return torch.tensor(lms, dtype=torch.float), torch.tensor(labels, dtype=torch.float)
    
trainDataset = CustomDataset("dataset")
loss_fn = nn.BCELoss()
loss_arr = []
epoch_arr = []
for epoch in range(EPOCHS):
    counter = 0
    for landmarks, labels in tqdm(trainDataset):
        landmarks = landmarks.to(device = DEVICE)
        labels = labels.to(device=DEVICE)
        if(landmarks == [0]*63):
            continue
        counter += 1
        pred = model(landmarks)
        loss = loss_fn(pred, labels)
        epoch_arr.append(counter)
        loss_arr.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model, "models/model5.pth")
plt.plot(epoch_arr, loss_arr)
plt.show()