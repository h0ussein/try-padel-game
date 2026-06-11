"""core/trackers — vendored Deep-EIoU tracker (Step 3).

Faithful, dependency-light implementation of "Iterative Scale-Up ExpansionIoU and
Deep Features Association for Multi-Object Tracking in Sports" (Deep-EIoU,
hsiangwei0903/Deep-EIoU), adapted to run with our YOLOv11-Pose detections + OSNet
ReID. NOT ByteTrack — association is ExpansionIoU (EIoU) with iterative scale-up,
fused with appearance embeddings. One tracker instance is created per camera.
"""
from .deep_eiou import DeepEIoU, STrack  # noqa: F401
