import trimesh
import numpy as np

# Function to clean the thin base plane of a mesh
def clean_base_plane(mesh_path: str, output_path: str, threshold_ratio: float = 0.01):
    # Load the mesh
    mesh = trimesh.load(mesh_path)

    # If the mesh is a Scene, merge it into a single Trimesh
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Ensure it is a valid Trimesh
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded file is not a valid Trimesh object.")

    # Calculate the average Z position of each face
    face_z_heights = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]

    # Determine the Z threshold (bottom 1% by default)
    z_threshold = np.quantile(face_z_heights, threshold_ratio)

    # Identify faces below the threshold
    face_mask = face_z_heights > z_threshold

    # Keep only the faces above the threshold
    cleaned_faces = mesh.faces[face_mask]

    # Get the unique vertices used by these faces
    unique_vertices = np.unique(cleaned_faces)
    cleaned_vertices = mesh.vertices[unique_vertices]

    # Remap face indices to the cleaned vertices
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
    remapped_faces = np.vectorize(vertex_map.get)(cleaned_faces)

    # Create a new mesh using the cleaned faces and vertices
    cleaned_mesh = trimesh.Trimesh(vertices=cleaned_vertices, faces=remapped_faces)

    # Export the cleaned mesh
    cleaned_mesh.export(output_path)

    print(f"Cleaned mesh saved to {output_path}")


if __name__ == "__main__":
    clean_base_plane("./blo/model.glb", "./blo/model_clean.glb")