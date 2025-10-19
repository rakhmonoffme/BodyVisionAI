"""
3D Mesh Viewer for OBJ Files

INSTALLATION:
    pip install trimesh pyglet matplotlib

USAGE:
    python mesh_viewer.py body_mesh_male.obj
    
    or in code:
    from mesh_viewer import view_mesh
    view_mesh('body_mesh_male.obj')
    
MODES:
    matplotlib  - Show in matplotlib (default, best for Mac)
    interactive - Open interactive viewer (requires display)
"""

import sys
import trimesh
import numpy as np
from pathlib import Path


def view_mesh_interactive(mesh_path: str):
    """Open interactive 3D viewer (requires display)."""
    mesh = trimesh.load(mesh_path)
    
    print(f"\nMesh Information:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume * 1000:.2f} liters")
    print(f"  Surface Area: {mesh.area:.2f} m²")
    print(f"\nControls:")
    print("  - Left click + drag: Rotate")
    print("  - Right click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'w': Toggle wireframe")
    print("  - Press 'z': Reset view")
    print("  - Press 'q': Quit\n")
    
    mesh.show()


def view_mesh_matplotlib(mesh_path: str, save_image: str = None):
    """View mesh using matplotlib (works on Mac without display issues)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    mesh = trimesh.load(mesh_path)
    
    fig = plt.figure(figsize=(12, 8))
    
    # Create 4 subplots for different views
    views = [
        (221, (30, 45), "Front-Right View"),
        (222, (30, 135), "Front-Left View"),
        (223, (30, -135), "Back-Left View"),
        (224, (30, -45), "Back-Right View")
    ]
    
    for subplot, (elev, azim), title in views:
        ax = fig.add_subplot(subplot, projection='3d')
        
        # Subsample faces for performance
        step = max(1, len(mesh.faces) // 5000)
        faces = mesh.faces[::step]
        vertices = mesh.vertices
        
        poly = Poly3DCollection(
            vertices[faces],
            alpha=0.7,
            facecolor='cyan',
            edgecolor='black',
            linewidths=0.1
        )
        ax.add_collection3d(poly)
        
        # Set equal aspect ratio
        max_range = np.array([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ]).max() / 2.0
        
        mid = np.array([
            vertices[:, 0].mean(),
            vertices[:, 1].mean(),
            vertices[:, 2].mean()
        ])
        
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
    
    info_text = (
        f"Mesh: {Path(mesh_path).name}\n"
        f"Vertices: {len(mesh.vertices):,}\n"
        f"Faces: {len(mesh.faces):,}\n"
        f"Volume: {mesh.volume * 1000:.2f} L\n"
        f"Surface Area: {mesh.area:.4f} m²\n"
        f"Watertight: {mesh.is_watertight}"
    )
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_image:
        plt.savefig(save_image, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_image}")
    
    plt.show()


def print_mesh_info(mesh_path: str):
    """Print detailed mesh information."""
    mesh = trimesh.load(mesh_path)
    
    print(f"\n{'='*60}")
    print(f"MESH INFORMATION: {Path(mesh_path).name}")
    print(f"{'='*60}")
    print(f"Vertices: {len(mesh.vertices):,}")
    print(f"Faces: {len(mesh.faces):,}")
    print(f"Edges: {len(mesh.edges):,}")
    print(f"\nGeometry:")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume * 1000:.4f} liters ({mesh.volume:.6f} m³)")
    print(f"  Surface Area: {mesh.area:.4f} m²")
    print(f"  Center of Mass: {mesh.center_mass}")
    print(f"\nBounding Box:")
    print(f"  Min: {mesh.bounds[0]}")
    print(f"  Max: {mesh.bounds[1]}")
    print(f"  Size: {mesh.bounds[1] - mesh.bounds[0]}")
    
    if mesh.is_watertight:
        print(f"\n✓ Mesh is watertight - volume calculation is accurate")
    else:
        print(f"\n✗ Mesh is NOT watertight - volume may be inaccurate")
    
    print(f"{'='*60}\n")


def view_mesh(mesh_path: str, mode: str = 'matplotlib', save_image: str = None):
    """
    View 3D mesh file.
    
    Args:
        mesh_path: Path to OBJ file
        mode: 'matplotlib' (default) or 'interactive'
        save_image: If using matplotlib, save to this file
    """
    if not Path(mesh_path).exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    print_mesh_info(mesh_path)
    
    if mode == 'matplotlib':
        view_mesh_matplotlib(mesh_path, save_image)
    elif mode == 'interactive':
        view_mesh_interactive(mesh_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'matplotlib' or 'interactive'")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python mesh_viewer.py <mesh_file.obj> [mode]")
        print("\nModes:")
        print("  matplotlib  - Show in matplotlib (default, best for Mac)")
        print("  interactive - Open interactive viewer (requires display)")
        print("\nExample:")
        print("  python mesh_viewer.py body_mesh_male.obj")
        print("  python mesh_viewer.py body_mesh_male.obj interactive")
        sys.exit(1)
    
    mesh_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'matplotlib'
    
    view_mesh(mesh_file, mode=mode)