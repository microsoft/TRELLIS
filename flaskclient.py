import requests
import os
import uuid

class Trellis3DClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        
    def generate_from_single_image(self, image_path, params=None):
        """
        Generate 3D model from a single image
        
        Args:
            image_path (str): Path to the input image
            params (dict): Optional parameters including:
                - seed (int): Random seed
                - ss_guidance_strength (float): Guidance strength for sparse structure generation
                - ss_sampling_steps (int): Sampling steps for sparse structure generation
                - slat_guidance_strength (float): Guidance strength for structured latent generation
                - slat_sampling_steps (int): Sampling steps for structured latent generation
                
        Returns:
            dict: Response containing session_id and download URLs
        """
        if params is None:
            params = {}
            
        url = f"{self.base_url}/generate_from_single_image"
        files = {'image': open(image_path, 'rb')}
        
        response = requests.post(url, files=files, data=params)
        return response.json()
        
    def generate_from_multiple_images(self, image_paths, params=None):
        """
        Generate 3D model from multiple images
        
        Args:
            image_paths (list): List of paths to input images
            params (dict): Optional parameters including:
                - seed (int): Random seed
                - ss_guidance_strength (float): Guidance strength for sparse structure generation
                - ss_sampling_steps (int): Sampling steps for sparse structure generation
                - slat_guidance_strength (float): Guidance strength for structured latent generation
                - slat_sampling_steps (int): Sampling steps for structured latent generation
                - multiimage_algo (str): Algorithm for multi-image generation ('stochastic' or 'multidiffusion')
                
        Returns:
            dict: Response containing session_id and download URLs
        """
        if params is None:
            params = {}
            
        url = f"{self.base_url}/generate_from_multiple_images"
        files = [('images', (os.path.basename(path), open(path, 'rb')) for path in image_paths]
        
        response = requests.post(url, files=files, data=params)
        return response.json()
        
    def extract_glb(self, session_id, params=None):
        """
        Extract GLB file from generated 3D model
        
        Args:
            session_id (str): Session ID from generation step
            params (dict): Optional parameters including:
                - mesh_simplify (float): Mesh simplification factor (0.9-0.98)
                - texture_size (int): Texture resolution (512, 1024, 1536, or 2048)
                
        Returns:
            dict: Response containing GLB download URL
        """
        if params is None:
            params = {}
            
        url = f"{self.base_url}/extract_glb"
        data = {'session_id': session_id, **params}
        
        response = requests.post(url, data=data)
        return response.json()
        
    def download_file(self, url, save_path=None):
        """
        Download a file from the server
        
        Args:
            url (str): Full URL to download (from previous responses)
            save_path (str): Optional path to save the file
            
        Returns:
            str: Path where file was saved
        """
        if save_path is None:
            save_path = os.path.basename(url)
            
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return save_path

# Example usage
if __name__ == '__main__':
    client = Trellis3DClient()
    
    # Example 1: Single image generation
    print("Generating from single image...")
    single_result = client.generate_from_single_image(
        'example_image.png',
        params={
            'seed': 42,
            'ss_guidance_strength': 7.5,
            'slat_guidance_strength': 3.0
        }
    )
    print(single_result)
    
    # Download preview
    client.download_file(
        f"http://localhost:5000{single_result['preview_url']}",
        'single_preview.mp4'
    )
    
    # Extract GLB
    glb_result = client.extract_glb(
        single_result['session_id'],
        params={'mesh_simplify': 0.95}
    )
    print(glb_result)
    
    # Download GLB
    client.download_file(
        f"http://localhost:5000{glb_result['glb_url']}",
        'single_model.glb'
    )
    
    # Example 2: Multiple image generation
    print("\nGenerating from multiple images...")
    multi_result = client.generate_from_multiple_images(
        ['view1.png', 'view2.png', 'view3.png'],
        params={
            'multiimage_algo': 'stochastic',
            'seed': 123
        }
    )
    print(multi_result)
    
    # Download preview
    client.download_file(
        f"http://localhost:5000{multi_result['preview_url']}",
        'multi_preview.mp4'
    )
    
    # Extract GLB
    glb_result = client.extract_glb(
        multi_result['session_id'],
        params={'texture_size': 2048}
    )
    print(glb_result)
    
    # Download GLB
    client.download_file(
        f"http://localhost:5000{glb_result['glb_url']}",
        'multi_model.glb'
    )