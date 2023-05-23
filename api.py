from fastapi import FastAPI, File, UploadFile, responses
from PIL import Image
import io
import numpy as np
import onnxruntime as rt

app = FastAPI()

sess = rt.InferenceSession("model.onnx")

@app.get("/test")
async def test():
    return {"message": "API is working"}

@app.api_route("/dummy", methods=["POST"])
async def dummy_function(file: UploadFile = File(...)):
    print("Dummy function called")
    # Perform some dummy processing or return a dummy response
    return {"message": "Dummy function called"}

#@app.api_route("/process_image", methods=["POST"])
@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    input_data = await file.read()
    input_image = Image.open(io.BytesIO(input_data)).convert("RGB")

    # Resize and convert to numpy array
    input_array = np.array(input_image.resize((512, 512)))

    # Normalize to [-1, 1]
    input_array = (input_array.astype(np.float32) / 127.5) - 1
    input_array = np.expand_dims(input_array, axis=0)

    # Reorder dimensions to match model's expectations
    input_array = input_array.transpose((0, 3, 1, 2))  # Move the color dimension to the correct position

    inputs = {sess.get_inputs()[0].name: input_array}
    outputs = sess.run(None, inputs)

    # Denormalize from [-1, 1] to [0, 255]
    output_array = (outputs[0][0].transpose((1, 2, 0)) + 1) * 127.5
    output_array = np.clip(output_array, 0, 255)  # Clip values outside of valid range

    # Convert back to image
    output_image = Image.fromarray(output_array.astype(np.uint8), "RGB")

    byte_arr = io.BytesIO()
    output_image.save(byte_arr, format='PNG')
    byte_arr.seek(0)  # Move cursor to the beginning of the file

    return responses.StreamingResponse(byte_arr, media_type="image/png")
