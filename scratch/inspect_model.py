from ai_edge_litert.interpreter import Interpreter
import os

MODEL_PATH = r"c:\Users\abody\OneDrive\Desktop\sign-language-pipeline\models\model.tflite"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    exit(1)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input Details: {input_details}")
print(f"Output Details: {output_details}")
