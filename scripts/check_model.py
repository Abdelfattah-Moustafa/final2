from ai_edge_litert.interpreter import Interpreter
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.tflite")

if os.path.exists(MODEL_PATH):
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Input Details:", interpreter.get_input_details())
    print("Output Details:", interpreter.get_output_details())
else:
    print(f"Model path not found: {MODEL_PATH}")
