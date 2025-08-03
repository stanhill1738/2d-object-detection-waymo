import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes=4):  # 3 foreground classes + 1 background
    # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone and FPN
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classification head with one for our dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
