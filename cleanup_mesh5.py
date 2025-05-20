import trimesh
import numpy as np

# Load the GLB file with error handling
def load_mesh(filename: str) -> trimesh.Trimesh:
    try:
        mesh = trimesh.load(filename)
        return mesh.geometry["geometry_0"]
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded object is not a single mesh.")
        return mesh
    except Exception as e:
        raise ValueError(f"Failed to load mesh: {e}")


import numpy as np


def remove_ground_plane(points, height_step=0.01, centroid_tolerance=0.1, max_iterations=100):
    """
    Remove ground plane from a point cloud using statistical analysis.

    Args:
        points: numpy array of shape (N, 3) containing the point cloud
        height_step: how much to increase the height of the removal rectangle each iteration
        centroid_tolerance: how close the centroid needs to be to the middle height range
        max_iterations: maximum number of iterations to perform

    Returns:
        numpy array of points with ground plane removed
    """
    original_points = points.copy()
    points = points.copy()

    # Calculate initial metrics
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    height_range = max_coords[2] - min_coords[2]

    for iteration in range(max_iterations):
        # 1. Calculate current centroid
        centroid = np.mean(points, axis=0)

        # 2. Find the 4 lowest points that are furthest from centroid in x-y plane
        # Get points in the bottom 10% height
        height_threshold = min_coords[2] + 0.1 * height_range
        bottom_points = points[points[:, 2] < height_threshold]

        if len(bottom_points) == 0:
            break

        # Calculate x-y distances from centroid
        xy_distances = np.linalg.norm(bottom_points[:, :2] - centroid[:2], axis=1)

        # Get indices of 4 furthest points in x-y plane
        furthest_indices = np.argpartition(xy_distances, -4)[-4:]
        ground_corners = bottom_points[furthest_indices]

        # 3. Create a bounding box from these corners
        ground_min = np.min(ground_corners, axis=0)
        ground_max = np.max(ground_corners, axis=0)

        # Expand the height of the bounding box
        ground_max[2] += height_step

        # 4. Remove points within this bounding box
        in_ground = np.all((points >= ground_min) & (points <= ground_max), axis=1)
        points = points[~in_ground]

        # Check termination conditions
        new_centroid = np.mean(points, axis=0)
        new_height_range = np.max(points[:, 2]) - np.min(points[:, 2])

        # Condition 1: Centroid is adequately centered vertically
        centroid_height_ratio = (new_centroid[2] - np.min(points[:, 2])) / new_height_range
        centroid_centered = abs(centroid_height_ratio - 0.5) < centroid_tolerance

        # Condition 2: The furthest corners are now closer to centroid (ground removed)
        new_bottom_points = points[points[:, 2] < (np.min(points[:, 2]) + 0.1 * new_height_range)]
        if len(new_bottom_points) > 0:
            new_xy_distances = np.linalg.norm(new_bottom_points[:, :2] - new_centroid[:2], axis=1)
            avg_distance_reduced = np.mean(new_xy_distances) < 0.5 * np.mean(xy_distances)
        else:
            avg_distance_reduced = True

        if centroid_centered and avg_distance_reduced:
            break

        # Update for next iteration
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        height_range = max_coords[2] - min_coords[2]

    return points

# Identify and remove the bottom plane artifact
def remove_bottom_plane(mesh: trimesh.Trimesh, height_threshold: float = 0.5) -> trimesh.Trimesh:
    # Make a copy of the original mesh
    mesh = mesh.copy()

    # Calculate mesh characteristics
    vertices = mesh.vertices
    min_z = np.min(vertices[:, 2])
    z_range = np.max(vertices[:, 2]) - min_z

    # Adaptive threshold based on mesh size
    adaptive_threshold = min_z + height_threshold * z_range

    # Find all faces that are entirely within the bottom plane region
    face_z_values = vertices[mesh.faces][:, :, 2]  # Z-coordinates of all face vertices
    max_face_z = np.max(face_z_values, axis=1)  # Max z for each face

    # Faces where all vertices are below the threshold
    faces_to_remove = max_face_z <= adaptive_threshold

    # Remove these faces
    mesh.update_faces(~faces_to_remove)

    # Remove unreferenced vertices to clean up
    mesh.remove_unreferenced_vertices()

    return mesh


def remove_ground_plane_from_mesh(mesh: trimesh.Trimesh,
                                  height_step=0.01,
                                  centroid_tolerance=0.1,
                                  max_iterations=100) -> trimesh.Trimesh:
    """
    Remove ground plane from a mesh using statistical analysis of its vertices.

    Args:
        mesh: Input mesh to process
        height_step: How much to increase the height of the removal volume each iteration
        centroid_tolerance: How close the centroid needs to be to the middle height range
        max_iterations: Maximum number of iterations to perform

    Returns:
        Processed mesh with ground plane removed
    """
    # Make a copy of the original mesh
    mesh = mesh.copy()
    vertices = mesh.vertices

    # Calculate initial metrics
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    height_range = max_coords[2] - min_coords[2]

    # Initialize removal volume
    removal_min = None
    removal_max = None

    for iteration in range(max_iterations):
        # 1. Calculate current centroid
        centroid = np.mean(vertices, axis=0)

        # 2. Find the 4 lowest points that are furthest from centroid in x-y plane
        # Get points in the bottom 10% height
        height_threshold = min_coords[2] + 0.1 * height_range
        bottom_points = vertices[vertices[:, 2] < height_threshold]

        if len(bottom_points) == 0:
            break

        # Calculate x-y distances from centroid
        xy_distances = np.linalg.norm(bottom_points[:, :2] - centroid[:2], axis=1)

        # Get indices of 4 furthest points in x-y plane
        furthest_indices = np.argpartition(xy_distances, -4)[-4:]
        ground_corners = bottom_points[furthest_indices]

        # 3. Create a bounding box from these corners
        ground_min = np.min(ground_corners, axis=0)
        ground_max = np.max(ground_corners, axis=0)

        # Expand the height of the bounding box
        ground_max[2] += height_step

        # 4. Find faces that are entirely within this bounding box
        # Get all vertices of each face
        face_vertices = vertices[mesh.faces]

        # Check if all 3 vertices of each face are within the bounding box
        in_ground = np.all(
            np.all((face_vertices >= ground_min) & (face_vertices <= ground_max), axis=2),
            axis=1)

        # Remove these faces
        mesh.update_faces(~in_ground)

        # Check termination conditions
        vertices = mesh.vertices  # Get updated vertices after face removal
        if len(vertices) == 0:
            break

        new_centroid = np.mean(vertices, axis=0)
        new_height_range = np.max(vertices[:, 2]) - np.min(vertices[:, 2])

        # Condition 1: Centroid is adequately centered vertically
        centroid_height_ratio = (new_centroid[2] - np.min(vertices[:, 2])) / new_height_range
        centroid_centered = abs(centroid_height_ratio - 0.5) < centroid_tolerance

        # Condition 2: The furthest corners are now closer to centroid (ground removed)
        new_bottom_points = vertices[vertices[:, 2] < (np.min(vertices[:, 2]) + 0.1 * new_height_range)]
        if len(new_bottom_points) > 0:
            new_xy_distances = np.linalg.norm(new_bottom_points[:, :2] - new_centroid[:2], axis=1)
            avg_distance_reduced = np.mean(new_xy_distances) < 0.5 * np.mean(xy_distances)
        else:
            avg_distance_reduced = True

        if centroid_centered and avg_distance_reduced:
            break

        # Update for next iteration
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        height_range = max_coords[2] - min_coords[2]

    # Clean up any unreferenced vertices
    mesh.remove_unreferenced_vertices()

    return mesh

# Save the cleaned mesh
def save_mesh(mesh: trimesh.Trimesh, filename: str, file_format: str = "glb"):
    mesh.export(filename, file_type=file_format)

if __name__ == "__main__":
    input_file = "./bla/model.glb"
    output_file = "./bla/model2.glb"

    mesh = load_mesh(input_file)
    cleaned_mesh = remove_ground_plane_from_mesh(mesh)
    save_mesh(cleaned_mesh, output_file)

    #print(f"Cleaned mesh saved to {args.output_file}.")