# FaceVision

A Streamlit app that finds and counts faces in a photo using **MTCNN**, with confidence scores, facial landmarks, cropped faces, and a filter that removes false detections.

![FaceVision detecting four faces](facevision/assets/Screenshot2.png)

## Features
- Detects every face and draws a box with a confidence score and five landmarks (eyes, nose, mouth corners)
- Shows the face count, the average confidence and a gallery of cropped faces
- Adjustable minimum confidence, crop padding and landmark display
- **False-positive filter**: removes detections that aren't faces
- **Debug view**: lists every candidate MTCNN returned and why it was kept or dropped

![Debug view of raw detections](facevision/assets/Screenshot1.png)

## How the filtering works
MTCNN occasionally reports non-face objects with high confidence. In testing, a sandal scored 99.4% and a patterned dress 94.5%. FaceVision handles this in three steps:

1. **Confidence cutoff** (default 99.5%) removes weak candidates.
2. **Landmark check** requires the eyes to sit above the nose and the nose above the mouth. It is deliberately loose, so turned and tilted heads still pass. An earlier, stricter version rejected real side-on faces, which the debug view helped me spot.
3. **Overlap removal** drops a box that mostly overlaps a higher-scoring one.

Large photos are downscaled to 1280 px before detection, because MTCNN becomes very slow on multi-megapixel images.

## Run it
Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run app.py
```

The first run is slow while TensorFlow loads. Photos are processed in memory and never saved.

## Project structure
```
app.py                       Streamlit interface
src/facevision/detector.py   Detection, filtering, cropping and drawing
.streamlit/config.toml       Theme
```

## Limitations
- Very small, blurred or heavily covered faces may be missed. Lower the confidence slider to catch more, at the cost of more false positives.
- MTCNN runs on CPU here, so large group photos can take a few seconds.

## Tech
Python · MTCNN · TensorFlow · OpenCV · Streamlit · uv