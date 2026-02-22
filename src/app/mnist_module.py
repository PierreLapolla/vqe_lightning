import torch.nn as nn
import torch.optim as optim

from app.base_module import BaseModule
from app.settings import AppSettings


class MNISTModule(BaseModule):
    def __init__(self, settings: AppSettings):
        super(MNISTModule, self).__init__(settings)

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def get_loss_func(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.settings.train.learning_rate,
            weight_decay=self.settings.train.weight_decay,
        )
        return optimizer

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
