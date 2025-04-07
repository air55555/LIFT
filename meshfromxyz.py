# Required libraries for mesh processing and visualization
import numpy as np
from plyfile import PlyData, PlyElement
import plotly.graph_objects as go
import os

def load_point_cloud(file_path):
    """
    Load point cloud data from a text file.
    
    Args:
        file_path (str): Path to the text file containing point cloud data
        
    Returns:
        numpy.ndarray: Array of points with shape (n_points, 3) where each point has (x,y,z) coordinates
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    points = []
    for line in lines[1:]:  # Skip header line
        if line.strip():
            x, y, z = map(float, line.split(','))
            points.append([x, y, z])
    return np.array(points)

def compute_mesh_statistics(vertices, faces):
    """
    Calculate various statistics for a 3D mesh.
    
    Args:
        vertices (numpy.ndarray): Array of vertex coordinates with shape (n_vertices, 3)
        faces (numpy.ndarray): Array of face indices with shape (n_faces, 3)
        
    Returns:
        dict: Dictionary containing mesh statistics including:
            - num_vertices: Number of vertices
            - num_faces: Number of faces
            - bounding_box: Min and max coordinates
            - centroid: Center point of the mesh
            - surface_area: Total surface area of the mesh
    """
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    bounding_box = np.vstack([min_bounds, max_bounds])
    centroid = vertices.mean(axis=0)

    def compute_face_area(face):
        """Calculate the area of a triangular face using Heron's formula."""
        vertices_face = vertices[face]
        a = np.linalg.norm(vertices_face[0] - vertices_face[1])
        b = np.linalg.norm(vertices_face[1] - vertices_face[2])
        c = np.linalg.norm(vertices_face[2] - vertices_face[0])
        s = (a + b + c) / 2
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    surface_area = np.sum([compute_face_area(face) for face in faces])

    return {
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "bounding_box": bounding_box,
        "centroid": centroid,
        "surface_area": surface_area
    }

def display_mesh_statistics(file_path):
    """
    Display statistics for a PLY mesh file.
    
    Args:
        file_path (str): Path to the PLY file
    """
    vertices, faces = read_ply(file_path)
    stats = compute_mesh_statistics(vertices, faces)

    print(f"Statistics for {file_path}:")
    print(f"Number of vertices: {stats['num_vertices']}")
    print(f"Number of faces: {stats['num_faces']}")
    print(f"Bounding box: \n{stats['bounding_box']}")
    print(f"Centroid: {stats['centroid']}")
    print(f"Surface area: {stats['surface_area']:.2f}")

def read_ply(file_path):
    """
    Read a PLY file and extract vertices and faces.
    
    Args:
        file_path (str): Path to the PLY file
        
    Returns:
        tuple: (vertices, faces) where:
            - vertices: numpy array of vertex coordinates
            - faces: numpy array of face indices
    """
    plydata = PlyData.read(file_path)
    vertices = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    faces = np.vstack(plydata['face']['vertex_indices'])
    return vertices, faces

def print_all_ply_files(directory_path):
    """
    Process and display statistics for all PLY files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing PLY files
    """
    ply_files = [f for f in os.listdir(directory_path) if f.endswith('.ply')]
    ply_files = sorted(ply_files)
    for i, file_name in enumerate(ply_files):
        file_path = os.path.join(directory_path, file_name)
        print(file_name)
        display_mesh_statistics(file_path)

def visualize_with_plotly(file_path):
    """
    Create an interactive 3D visualization of a mesh using Plotly.
    
    Args:
        file_path (str): Path to the PLY file to visualize
    """
    vertices, faces = read_ply(file_path)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    lines = []
    for face in faces:
        for idx in range(3):
            start, end = face[idx], face[(idx + 1) % 3]
            lines.append(go.Scatter3d(
                x=[x[start], x[end]],
                y=[y[start], y[end]],
                z=[z[start], z[end]],
                mode='lines',
                line=dict(color='black', width=2)
            ))

    layout = go.Layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            bgcolor='white'
        ),
        showlegend=False,
        paper_bgcolor='white'
    )

    fig = go.Figure(data=lines, layout=layout)
    fig.write_html("pl.html")  # Save visualization to HTML file
    fig.show()  # Display interactive visualization

def visualize_all_ply_files(directory_path, start_index, num_files):
    """
    Process and visualize multiple PLY files from a directory.
    
    Args:
        directory_path (str): Path to the directory containing PLY files
        start_index (int): Index of the first file to process
        num_files (int): Number of files to process
    """
    ply_files = [f for f in os.listdir(directory_path) if f.endswith('.ply')]
    ply_files = sorted(ply_files)
    for i, file_name in enumerate(ply_files[start_index:start_index+num_files]):
        if file_name.endswith('.ply'):
            file_path = os.path.join(directory_path, file_name)
            print(f"Visualizing {file_name}...")
            display_mesh_statistics(file_path)
            visualize_with_plotly(file_path)

# Set the directory path and process files
directory_path=r'S:\Users\LRS\PycharmProjects\LIFT'
print_all_ply_files(directory_path)  # Display statistics for all PLY files
visualize_all_ply_files(directory_path,0,1)  # Visualize the first PLY file