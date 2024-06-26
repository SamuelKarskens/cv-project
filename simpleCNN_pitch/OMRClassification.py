from ImageClassificationBase import ImageClassificationBase
from torch import nn

class OMRClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            #
            nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            # nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            nn.MaxPool2d(2,2),

            # nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.MaxPool2d(2,2),
            #
            nn.Flatten(),
            nn.Linear(35200,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,25)
        )

    def forward(self, xb):
        return self.network(xb)