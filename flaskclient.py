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
            image_paths (str or list): Either:
                - A list of paths to input images, or
                - A directory path containing images (will load all image files from the directory)
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

        # If image_paths is a directory, find all image files in it
        if isinstance(image_paths, str) and os.path.isdir(image_paths):
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
            image_paths = [
                os.path.join(image_paths, f)
                for f in os.listdir(image_paths)
                if f.lower().endswith(image_extensions)
            ]
            if not image_paths:
                raise ValueError(f"No image files found in directory: {image_paths}")

        # Ensure image_paths is a list at this point
        if not isinstance(image_paths, list):
            raise ValueError("image_paths must be either a list of image paths or a directory path")

        url = f"{self.base_url}/generate_from_multiple_images"
        files = [('images', (os.path.basename(path), open(path, 'rb'))) for path in image_paths]

        try:
            response = requests.post(url, files=files, data=params)
            return response.json()
        finally:
            # Ensure all files are closed after the request
            for _, (_, file_obj) in files:
                file_obj.close()
        
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

    def generate_and_download_from_single_image(self, image_path, target_dir=None, params=None):
        """
        Generate 3D model from a single image and download all artifacts

        Args:
            image_path (str): Path to the input image
            target_dir (str): Optional target directory to save files (defaults to /tmp/random_uuid)
            params (dict): Optional generation parameters

        Returns:
            dict: Paths to downloaded files
        """
        if target_dir is None:
            target_dir = f"/tmp/{uuid.uuid4()}"
        os.makedirs(target_dir, exist_ok=True)

        # Generate the 3D model
        gen_result = self.generate_from_single_image(image_path, params)

        # Download preview
        preview_path = os.path.join(target_dir, 'preview.mp4')
        self.download_file(
            f"{self.base_url}{gen_result['preview_url']}",
            preview_path
        )

        # Extract and download GLB
        glb_result = self.extract_glb(gen_result['session_id'], params)
        glb_path = os.path.join(target_dir, 'model.glb')
        self.download_file(
            f"{self.base_url}{glb_result['glb_url']}",
            glb_path
        )

        return {
            'preview_path': preview_path,
            'glb_path': glb_path,
            'session_id': gen_result['session_id'],
            'target_dir': target_dir
        }

    def generate_and_download_from_multiple_images(self, image_paths, target_dir=None, params=None):
        """
        Generate 3D model from multiple images and download all artifacts

        Args:
            image_paths (list): List of paths to input images
            target_dir (str): Optional target directory to save files (defaults to /tmp/random_uuid)
            params (dict): Optional generation parameters

        Returns:
            dict: Paths to downloaded files
        """
        if target_dir is None:
            target_dir = f"/tmp/{uuid.uuid4()}"
        os.makedirs(target_dir, exist_ok=True)

        # Generate the 3D model
        gen_result = self.generate_from_multiple_images(image_paths, params)

        # Download preview
        preview_path = os.path.join(target_dir, 'preview.mp4')
        self.download_file(
            f"{self.base_url}{gen_result['preview_url']}",
            preview_path
        )

        # Extract and download GLB
        glb_result = self.extract_glb(gen_result['session_id'], params)
        glb_path = os.path.join(target_dir, 'model.glb')
        self.download_file(
            f"{self.base_url}{glb_result['glb_url']}",
            glb_path
        )

        return {
            'preview_path': preview_path,
            'glb_path': glb_path,
            'session_id': gen_result['session_id'],
            'target_dir': target_dir
        }

# Example usage
if __name__ == '__main__':
    client = Trellis3DClient()

    multi_result = client.generate_and_download_from_multiple_images(
        '/home/charlie/Desktop/Holodeck/hippo/datasets/sacha_kitchen/segments/6/rgb',
        target_dir="./blo",
        params={
            'multiimage_algo': 'stochastic',
            'seed': 123
        }
    )
    exit()

    single_result = client.generate_and_download_from_single_image(
        'test/000.png',
        target_dir="./bla",
        params={
            'seed': 42,
            'ss_guidance_strength': 7.5,
            'slat_guidance_strength': 3.0
        }
    )

    exit()
    
    # Example 1: Single image generation
    print("Generating from single image...")
    single_result = client.generate_from_single_image(
        'test/000.png',
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