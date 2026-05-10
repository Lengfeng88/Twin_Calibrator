import torch
from torch.utils.data import Dataset
import numpy as np, cv2, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mock.sensor import MockSensor, PRESETS

class LightDataset(Dataset):
    LUM_MIN, LUM_RANGE = 10.0, 240.0
    CCT_MIN, CCT_RANGE = 2700.0, 5300.0

    def __init__(self, n_samples=8000, img_size=64):
        self.n = n_samples
        frames, labels = [], []
        per = n_samples // len(PRESETS)
        for scene in PRESETS:
            sensor = MockSensor(scene=scene,
                                size=(img_size, img_size),
                                drift=0.08)
            sensor.start()
            for _ in range(per):
                f = sensor.capture_array()
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                lum = float(np.mean(hsv[:,:,2]))
                cct = PRESETS[scene].cct_k + np.random.uniform(-200, 200)
                lum_n = np.clip((lum - self.LUM_MIN) / self.LUM_RANGE, 0, 1)
                cct_n = np.clip((cct - self.CCT_MIN) / self.CCT_RANGE, 0, 1)
                img = f.astype(np.float32) / 255.0
                frames.append(np.transpose(img, (2,0,1)))
                labels.append([float(lum_n), float(cct_n)])
            sensor.stop()
        idx = np.random.permutation(len(frames))
        self.X = np.array(frames, dtype=np.float32)[idx]
        self.Y = np.array(labels, dtype=np.float32)[idx]

    def __len__(self): return self.n
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])