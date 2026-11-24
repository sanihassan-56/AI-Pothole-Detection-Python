
import onnxruntime as ort
import os

model_path = "best.onnx"

print("Loading ONNX model...")

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

print("\nMODEL INFORMATION")
print("----------------------------")

print("Model size:",
      round(os.path.getsize(model_path) / (1024 * 1024), 2),
      "MB")

print("\nINPUT DETAILS")

for inp in session.get_inputs():
    print("Name:", inp.name)
    print("Shape:", inp.shape)
    print("Type:", inp.type)

print("\nOUTPUT DETAILS")

for out in session.get_outputs():
    print("Name:", out.name)
    print("Shape:", out.shape)
    print("Type:", out.type)

print("\nONNX model inspection completed.")
