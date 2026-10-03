# Image Prediction and Visualization

Start a model service using the [inference guide](../docs/INFERENCE.md), then install the client from the project root:

```bash
python -m pip install -r requirements.txt
python examples/predict.py --image your_image.jpg --phrase "the red car" --task bbox --output outputs/car
python examples/predict.py --image your_image.jpg --phrase "the center of the red car" --task point --output outputs/car_point
```

Use a new output directory for each prediction. The example saves `result.json` and `prediction.png`. The JSON includes labels, coordinates on a 0–999 grid, raw output, finish reason, token usage, and parsing status. A `None` response indicates that no matching object was found, so there is no box to draw. For malformed or truncated responses, the example preserves the raw output and exits without generating a visualization.

Replace `your_image.jpg` with your image path. Use `--base-url` and `--model` to select the service endpoint and model ID. To connect to a service that requires authentication, use the Python API's `api_key` parameter.
