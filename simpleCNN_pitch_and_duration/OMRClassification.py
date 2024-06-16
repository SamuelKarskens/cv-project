from ImageClassificationBase import ImageClassificationPitchDuration
from torch import nn

class OMRClassification(ImageClassificationPitchDuration):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 5, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(20736, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.pitch_prediction = nn.Sequential(
            # head to predict pitch
            nn.Linear(512,25)
        )
        
        self.duration_prediction = nn.Sequential(
            # head to predict duration
            nn.Linear(512,5)
        )

    def forward(self, xb):
        output_backbone = self.network(xb)
        pitch_logits = self.pitch_prediction(output_backbone)
        duration_logits = self.duration_prediction(output_backbone)
        return pitch_logits, duration_logits