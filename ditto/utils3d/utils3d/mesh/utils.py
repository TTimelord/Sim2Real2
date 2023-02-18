"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-01-12
 @desc Mesh utils
"""
import trimesh


def as_mesh(scene_or_mesh):
    """Convert a possible scene to mesh

    Args:
        scene_or_mesh (trimesh.Scene or trimesh.Trimesh): input scene or mesh.

    Returns:
        trimesh.Trimesh: Converted mesh with only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces, visual=g.visual)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh
