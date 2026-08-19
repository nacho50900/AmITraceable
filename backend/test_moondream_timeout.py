import sys
import time
from PIL import Image
from app.vision.scene_analysis import analyze_image_content, _lazy_load

def main():
    print("Loading model...")
    _lazy_load()
    image = Image.open("../logo.jpg")
    print("Starting analysis...")
    start = time.time()
    res = analyze_image_content(image)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
