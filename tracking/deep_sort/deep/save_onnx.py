import torch
import torch.onnx
from model import Net  # Make sure this import works correctly

checkpoint = torch.load("./checkpoint/ckpt.t7", map_location=torch.device('cpu'))
net_dict = checkpoint['net_dict']

model = Net(reid=True)

model.load_state_dict(net_dict, strict=False)

model.eval()

dummy_input = torch.randn(1, 3, 128, 64)

torch.onnx.export(model,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  "reid_model.onnx",   # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print("Model has been converted to ONNX")