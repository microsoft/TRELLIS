import os
import numpy as np
import trimesh
from pygltflib import GLTF2
from tqdm import tqdm


def is_bottom_plane(vertices, faces, z_threshold=0.1, plane_area_threshold=0.5):
    """
    Identify if there's a large planar surface at the bottom of the mesh.

    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        z_threshold: Height threshold to consider as "bottom"
        plane_area_threshold: Minimum area to consider as significant plane

    Returns:
        Tuple of (bool indicating if plane found, face indices of the plane)
    """
    # Find vertices near the bottom
    min_z = np.min(vertices[:, 2])
    bottom_vertices = vertices[:, 2] < min_z + z_threshold

    # Get faces that use these bottom vertices
    bottom_faces_mask = np.any(bottom_vertices[faces], axis=1)
    bottom_faces = faces[bottom_faces_mask]

    if len(bottom_faces) == 0:
        return False, np.array([])

    # Calculate normals of bottom faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    face_normals = mesh.face_normals[bottom_faces_mask]

    # Find faces that are roughly horizontal (normal points mostly up/down)
    vertical_normals = np.abs(face_normals[:, 2]) > 0.9  # 0.9 means ~25 degree tolerance
    horizontal_faces = bottom_faces[vertical_normals]

    if len(horizontal_faces) == 0:
        return False, np.array([])

    # Calculate area of horizontal faces
    horizontal_face_indices = np.where(bottom_faces_mask)[0][vertical_normals]
    areas = mesh.area_faces[horizontal_face_indices]
    total_area = np.sum(areas)

    if total_area < plane_area_threshold:
        return False, np.array([])

    return True, horizontal_face_indices


def remove_plane_from_glb(glb_path, output_path=None):
    """
    Load a GLB file, detect and remove bottom plane, then save it back.

    Args:
        glb_path: Path to input GLB file
        output_path: Path to save cleaned GLB (None to overwrite)
    """
    if output_path is None:
        output_path = glb_path

    # Load GLB file
    gltf = GLTF2().load(glb_path)

    # Process each mesh in the GLB
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            # Get accessor indices
            pos_accessor = gltf.accessors[primitive.attributes.POSITION]
            normal_accessor = gltf.accessors[primitive.attributes.NORMAL]
            indices_accessor = gltf.accessors[primitive.indices]

            # Get buffer views
            pos_view = gltf.bufferViews[pos_accessor.bufferView]
            normal_view = gltf.bufferViews[normal_accessor.bufferView]
            indices_view = gltf.bufferViews[indices_accessor.bufferView]

            # Get actual data
            buffer = gltf.buffers[pos_view.buffer]
            pos_data = gltf.get_data_from_buffer_uri(buffer.uri)
            pos_array = np.frombuffer(pos_data, dtype=np.float32,
                                      count=pos_accessor.count * 3,
                                      offset=pos_view.byteOffset)
            pos_array = pos_array.reshape(-1, 3)

            indices_data = gltf.get_data_from_buffer_uri(buffer.uri)
            indices_array = np.frombuffer(indices_data,
                                          dtype=np.uint16 if indices_accessor.componentType == 5123 else np.uint32,
                                          count=indices_accessor.count,
                                          offset=indices_view.byteOffset)
            indices_array = indices_array.reshape(-1, 3)

            # Check for bottom plane
            has_plane, plane_face_indices = is_bottom_plane(pos_array, indices_array)

            if has_plane:
                print(f"Found bottom plane with {len(plane_face_indices)} faces in {glb_path}")

                # Remove the plane faces
                mask = np.ones(len(indices_array), dtype=bool)
                mask[plane_face_indices] = False
                new_indices = indices_array[mask]

                # Update the indices in the GLB structure
                indices_accessor.count = len(new_indices) * 3
                indices_data = new_indices.tobytes()

                # Update buffer view byte length
                indices_view.byteLength = len(indices_data)

                # Update the buffer
                buffer_data = gltf.get_data_from_buffer_uri(buffer.uri)
                buffer_data = buffer_data[:indices_view.byteOffset] + indices_data + buffer_data[
                                                                                     indices_view.byteOffset + len(
                                                                                         indices_data):]
                gltf.set_data_from_buffer_uri(buffer.uri, buffer_data)

    # Save the cleaned GLB
    gltf.save(output_path)


def process_glb_directory(root_dir, output_dir=None):
    """
    Process all GLB files in a directory and its subdirectories.

    Args:
        root_dir: Root directory containing GLB files
        output_dir: Output directory (None to overwrite original files)
    """
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all GLB files
    glb_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.glb'):
                glb_files.append(os.path.join(root, file))

    # Process each file
    for glb_path in tqdm(glb_files, desc="Processing GLB files"):
        if output_dir is not None:
            rel_path = os.path.relpath(glb_path, root_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = None

        try:
            remove_plane_from_glb(glb_path, output_path)
        except Exception as e:
            print(f"Error processing {glb_path}: {str(e)}")


if __name__ == "__main__":
    import argparse

    #parser = argparse.ArgumentParser(description="Remove bottom planes from GLB meshes")
    #parser.add_argument("input_dir", help="Input directory containing GLB files")
    #parser.add_argument("--output_dir", help="Output directory (optional, overwrites by default)")

    #args = parser.parse_args()

    process_glb_directory("./blo/model.glb", "./blo/modelCLEANED.glb")