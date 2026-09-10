import os
from typing import Dict, Any
from model.predict import predict

class SignLanguagePipeline:
    """
    End-to-end pipeline orchestrating landmark extraction and ISL classification.
    """
    def __init__(self):
        # Predict module lazily loads model weights and class mapping on first call
        pass

    def run(self, video_path: str) -> Dict[str, Any]:
        """
        Runs the full inference pipeline on an input MP4 video file.
        
        Args:
            video_path (str): Path to the input video.
            
        Returns:
            dict: {
                "success": bool,
                "sign": str,
                "confidence": float,
                "error": str or None
            }
        """
        if not os.path.isfile(video_path):
            return {
                "success": False,
                "sign": "",
                "confidence": 0.0,
                "error": f"File not found: {video_path}"
            }

        try:
            result = predict(video_path)
            return {
                "success": True,
                "sign": result["sign"],
                "confidence": result["confidence"],
                "error": None
            }
        except Exception as err:
            return {
                "success": False,
                "sign": "",
                "confidence": 0.0,
                "error": str(err)
            }

# Global singleton pipeline instance
pipeline = SignLanguagePipeline()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
        print(f"Running pipeline on: {test_video}")
        output = pipeline.run(test_video)
        print("Pipeline output:", output)
    else:
        print("SignLanguagePipeline ready. Run with a video path argument to test.")