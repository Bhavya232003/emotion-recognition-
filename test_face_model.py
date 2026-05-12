import base64
import sys
from backend.facial_emotion import predict, warmup

def test_image(image_path):
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        print(f"\n--- Testing {image_path} ---")
        result = predict(img_b64)
        print(f"Dominant Emotion: {result.get('dominant_emotion')}")
        print(f"Scores: {result.get('scores')}")
        if 'error' in result:
            print(f"Error: {result['error']}")
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

if __name__ == "__main__":
    print("Warming up DeepFace...")
    warmup()
    if len(sys.argv) < 2:
        print("Please provide image paths as arguments.")
        sys.exit(1)
    
    for path in sys.argv[1:]:
        test_image(path)
