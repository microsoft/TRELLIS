import requests
from PIL import Image
import io
import os

class Gradio3DClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def send_image(self, image_path: str) -> dict:
        """Sends an image to the Gradio app and generates a 3D model."""
        with open(image_path, "rb") as f:
            response = requests.post(f"{self.base_url}/run/image_to_3d", files={"image": f})
        response.raise_for_status()
        return response.json()["data"][0]

    def extract_glb(self, state: dict, mesh_simplify: float = 0.95, texture_size: int = 1024) -> str:
        """Extracts a GLB file from the generated 3D model."""
        response = requests.post(f"{self.base_url}/run/extract_glb", json={"state": state, "mesh_simplify": mesh_simplify, "texture_size": texture_size})
        response.raise_for_status()
        return response.json()["data"][0]

    def generate_views(self, state: dict, num_views: int = 4) -> list:
        """Generates views of the 3D model from different angles."""
        images = []
        for angle in range(0, 360, 360 // num_views):
            response = requests.post(f"{self.base_url}/run/render_view", json={"state": state, "angle": angle})
            response.raise_for_status()
            images.append(response.content)
        return images

    def process_image(self, image_path: str) -> dict:
        """Full pipeline: Send image, generate model, extract GLB, and generate views."""
        model_info = self.send_image(image_path)
        glb_path = self.extract_glb(model_info)
        views = self.generate_views(model_info)

        return {
            "original_image": image_path,
            "glb_path": glb_path,
            "views": views
        }

if __name__ == "__main__":
    client = Gradio3DClient("http://localhost:7860")
    result = client.process_image("/home/charlie/Desktop/Holodeck/hippo/datasets/sacha_kitchen/segments/2/rgb/000.png")
    print("Process complete. Files:", result)
