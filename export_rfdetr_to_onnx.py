from rfdetr import RFDETRSmall

# load model
model = RFDETRSmall()

# export as onnx
model.export("models/rfdetr_small.onnx")