import os
from torch.utils.data import Dataset
from PIL import Image

class LieTruthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label, class_name in enumerate(["truth", "lie"]):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for person in os.listdir(class_dir):
                person_dir = os.path.join(class_dir, person)
                for question in os.listdir(person_dir):
                    question_dir = os.path.join(person_dir, question)
                    for img_name in os.listdir(question_dir):
                        img_path = os.path.join(question_dir, img_name)
                        if img_path.lower().endswith(".png"):
                            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
