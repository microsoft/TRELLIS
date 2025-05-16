import numpy as np
import trimesh
from pygltflib import GLTF2
from pygltflib.utils import gltf2glb


def remove_bottom_plane(glb_path, output_path, height_threshold=0.1, plane_normal_tolerance=0.9):
    """
    Remove unwanted planar artifacts at the bottom of a GLB mesh.

    Args:
        glb_path (str): Path to input GLB file
        output_path (str): Path to save cleaned GLB file
        height_threshold (float): Height threshold to consider vertices as "bottom"
        plane_normal_tolerance (float): Tolerance for detecting planar surfaces (0.9 = mostly aligned with XY plane)
    """
    # Load the GLB file
    gltf = GLTF2().load(glb_path)

    # Process each mesh in the GLB
    for mesh_index, mesh in enumerate(gltf.meshes):
        for primitive in mesh.primitives:
            # Get the position accessor and buffer view
            pos_accessor = gltf.accessors[primitive.attributes.POSITION]
            pos_buffer_view = gltf.bufferViews[pos_accessor.bufferView]
            pos_buffer = gltf.buffers[pos_buffer_view.buffer]

            # Get the position data as numpy array
            pos_data = gltf.get_data_from_buffer_uri(pos_buffer.uri)
            pos_array = np.frombuffer(pos_data, dtype=np.float32,
                                      count=pos_accessor.count * 3,
                                      offset=pos_buffer_view.byteOffset)
            pos_array = pos_array.reshape(-1, 3)

            # Get indices if they exist
            if hasattr(primitive, 'indices'):
                indices_accessor = gltf.accessors[primitive.indices]
                indices_buffer_view = gltf.bufferViews[indices_accessor.bufferView]
                indices_buffer = gltf.buffers[indices_buffer_view.buffer]
                indices_data = gltf.get_data_from_buffer_uri(indices_buffer.uri)
                indices_array = np.frombuffer(indices_data,
                                              dtype=np.uint16 if indices_accessor.componentType == 5123 else np.uint32,
                                              count=indices_accessor.count,
                                              offset=indices_buffer_view.byteOffset)
            else:
                indices_array = np.arange(len(pos_array))

            # Find bottom vertices
            min_z = np.min(pos_array[:, 2])
            bottom_vertices = pos_array[:, 2] < (min_z + height_threshold)

            if np.sum(bottom_vertices) < 3:  # Not enough vertices to form a plane
                continue

            # Check if these vertices form a plane (normal mostly in Z direction)
            bottom_pos = pos_array[bottom_vertices]
            centroid = np.mean(bottom_pos, axis=0)
            cov = np.cov((bottom_pos - centroid).T)
            _, eig_vecs = np.linalg.eig(cov)
            normal = eig_vecs[:, np.argmin(np.abs(eig_vecs))]
            normal = normal / np.linalg.norm(normal)

            # If the normal is mostly vertical (aligned with Z axis)
            if abs(normal[2]) > plane_normal_tolerance:
                print(f"Found bottom plane in mesh {mesh_index} with normal {normal}")

                # Create a mask of faces that contain bottom vertices
                if len(indices_array) % 3 == 0:  # Assuming triangles
                    faces = indices_array.reshape(-1, 3)
                    bottom_faces_mask = np.any(bottom_vertices[faces], axis=1)

                    # Keep only non-bottom faces
                    if hasattr(primitive, 'indices'):
                        # For indexed geometry
                        valid_faces = faces[~bottom_faces_mask]
                        new_indices = valid_faces.flatten()

                        # Update the indices buffer
                        indices_buffer.data = new_indices.tobytes()
                        indices_accessor.count = len(new_indices)
                    else:
                        # For non-indexed geometry
                        valid_vertices_mask = ~bottom_vertices
                        new_pos_array = pos_array[valid_vertices_mask]

                        # Update the position buffer
                        pos_buffer.data = new_pos_array.tobytes()
                        pos_accessor.count = len(new_pos_array)

    # Save the cleaned GLB
    gltf.save(output_path)


# Example usage
input_glb = "./blo/model.glb"
output_glb = "./blo/OUT.glb"
remove_bottom_plane(input_glb, output_glb)