import trimesh
import numpy as np
from typing import Optional, Dict, List
import os


def find_and_remove_bottom_planes(
        scene: trimesh.Scene,
        z_threshold: Optional[float] = None,
        angle_threshold: float = 15.0,
        min_area_ratio: float = 0.05,
        naming_pattern: str = "world/geometry_"
) -> Dict[str, bool]:
    """
    Find and remove bottom plane artefacts from all meshes in a scene that match the naming pattern.

    Args:
        scene: The loaded trimesh Scene object
        z_threshold: Absolute Z threshold for bottom detection (auto-detected if None)
        angle_threshold: Maximum angle (degrees) from horizontal to consider as bottom plane
        min_area_ratio: Minimum area ratio (compared to bounding box) to consider as artefact plane
        naming_pattern: Pattern to identify geometry nodes to process

    Returns:
        Dictionary mapping geometry names to whether they were modified
    """
    results = {}

    # Iterate through all geometry nodes in the scene
    for node_name in scene.graph.nodes_geometry:
        if naming_pattern not in node_name:
            continue

        # Get the mesh instance
        geometry = scene.geometry[node_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue

        print(f"\nProcessing {node_name}...")
        mesh = geometry.copy()
        original_face_count = len(mesh.faces)

        # Calculate mesh properties
        bounds = mesh.bounds
        height = bounds[1][2] - bounds[0][2]

        # Auto-detect z-threshold if not provided
        if z_threshold is None:
            z_threshold = bounds[0][2] + height * 0.1  # Bottom 10% of mesh

        # Find faces in the bottom region
        face_z_coords = mesh.vertices[mesh.faces][:, :, 2]
        bottom_faces = np.all(face_z_coords < z_threshold, axis=1)

        if not np.any(bottom_faces):
            print(f"No bottom faces found in {node_name}")
            results[node_name] = False
            continue

        # Get normals of bottom faces
        bottom_normals = mesh.face_normals[bottom_faces]

        # Calculate angle from vertical (we want faces that are roughly horizontal)
        angles = np.degrees(np.arccos(np.abs(bottom_normals[:, 2])))
        horizontal_faces = angles < angle_threshold

        if not np.any(horizontal_faces):
            print(f"No horizontal faces found in bottom region of {node_name}")
            results[node_name] = False
            continue

        # Get all face indices that meet our criteria
        candidate_faces = np.where(bottom_faces)[0][horizontal_faces]

        # Calculate area of candidate faces
        candidate_area = mesh.area_faces[candidate_faces].sum()
        total_area = mesh.area

        if candidate_area / total_area < min_area_ratio:
            print(f"Bottom plane area ({candidate_area:.4f}) too small in {node_name}")
            results[node_name] = False
            continue

        # Create a submesh of just the candidate faces
        plane_mesh = mesh.submesh([candidate_faces], append=True)[0]

        # Split the plane mesh into connected components
        components = plane_mesh.split(only_watertight=False)

        if not components:
            print(f"No components found in candidate faces of {node_name}")
            results[node_name] = False
            continue

        # Find the largest component (likely our plane)
        largest_component = max(components, key=lambda x: x.area)

        # Verify this component is large enough
        if largest_component.area / total_area < min_area_ratio:
            print(f"Largest component in bottom region is too small in {node_name}")
            results[node_name] = False
            continue

        # Get the vertex indices of the largest component
        component_verts = set(largest_component.vertices.view(np.ndarray).flatten())

        # Find all faces in the original mesh that exclusively use these vertices
        original_faces_to_remove = []
        for face_idx in candidate_faces:
            face_verts = mesh.faces[face_idx]
            if all(v in component_verts for v in face_verts):
                original_faces_to_remove.append(face_idx)

        if not original_faces_to_remove:
            print(f"Could not map component back to original faces in {node_name}")
            results[node_name] = False
            continue

        # Remove the faces from the original mesh
        mask = np.ones(len(mesh.faces), dtype=bool)
        mask[original_faces_to_remove] = False
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()

        # Update the scene with the modified mesh
        scene.geometry[node_name] = mesh

        removed_faces = original_face_count - len(mesh.faces)
        print(f"Removed {removed_faces} faces from {node_name}")
        results[node_name] = True

    return results


def process_glb_files(
        input_path: str,
        output_path: str,
        z_threshold: Optional[float] = None,
        angle_threshold: float = 15.0,
        min_area_ratio: float = 0.05,
        naming_pattern: str = "world/geometry_"
):
    """
    Process a GLB file to remove bottom plane artefacts from specific geometry nodes.

    Args:
        input_path: Path to input GLB file
        output_path: Path to save cleaned GLB file
        z_threshold: Absolute Z threshold for bottom detection (auto-detected if None)
        angle_threshold: Maximum angle (degrees) from horizontal to consider as bottom plane
        min_area_ratio: Minimum area ratio (compared to bounding box) to consider as artefact plane
        naming_pattern: Pattern to identify geometry nodes to process
    """
    # Load the scene
    scene = trimesh.load(input_path)

    if not isinstance(scene, trimesh.Scene):
        print("Input file is not a scene, using single mesh handling")
        # Handle as single mesh (using previous approach)
        remove_bottom_plane(input_path, output_path, z_threshold, angle_threshold, min_area_ratio)
        return

    print(f"Loaded scene with {len(scene.geometry)} geometries")

    # Process all matching geometry nodes
    results = find_and_remove_bottom_planes(
        scene,
        z_threshold,
        angle_threshold,
        min_area_ratio,
        naming_pattern
    )

    # Count how many were modified
    modified_count = sum(results.values())
    print(f"\nModified {modified_count} out of {len(results)} matching geometries")

    if modified_count > 0:
        # Save the modified scene
        scene.export(output_path)
        print(f"Saved cleaned scene to {output_path}")
    else:
        print("No modifications made - output file not created")

# Example usage
if __name__ == "__main__":
    input_glb = "./blo/model.glb"
    output_glb = "./blo/output_cleaned.glb"

    process_glb_files(input_glb, output_glb)