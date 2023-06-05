from typing import List

import numpy as np
import open3d as o3d
import sapien.core as sapien
import trimesh


def get_actor_meshes(actor: sapien.ActorBase):
    """in actor frame"""
    meshes = []
    for col_shape in actor.get_collision_shapes():
        geom = col_shape.geometry
        if isinstance(geom, sapien.BoxGeometry):
            mesh = trimesh.creation.box(extents=2 * geom.half_lengths)
        elif isinstance(geom, sapien.CapsuleGeometry):
            mesh = trimesh.creation.capsule(
                height=2 * geom.half_length, radius=geom.radius
            )
        elif isinstance(geom, sapien.SphereGeometry):
            mesh = trimesh.creation.icosphere(radius=geom.radius)
        elif isinstance(geom, sapien.PlaneGeometry):
            continue
        elif isinstance(
            geom, (sapien.ConvexMeshGeometry, sapien.NonconvexMeshGeometry)
        ):
            vertices = geom.vertices  # [n, 3]
            faces = geom.indices  # [m * 3]
            faces = [faces[i : i + 3] for i in range(0, len(faces), 3)]
            vertices = vertices * geom.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise TypeError(type(geom))
        mesh.apply_transform(col_shape.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_articulation_meshes(articulation: sapien.ArticulationBase):
    """in world frame"""
    meshes = []
    for link in articulation.get_links():
        link_meshes = get_actor_meshes(link)
        link_mesh = merge_meshes(link_meshes)
        if not link_mesh:
            continue
        link_mesh.apply_transform(link.get_pose().to_transformation_matrix())
        meshes.append(link_mesh)

    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def to_trimesh(x):
    if is_trimesh(x):
        return x
    elif isinstance(x, np.ndarray):
        assert x.ndim == 2 and x.shape[-1] == 3
        return trimesh.points.PointCloud(x)
    elif isinstance(x, o3d.geometry.TriangleMesh):
        vertices = np.asarray(x.vertices)
        faces = np.asarray(x.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    elif isinstance(x, o3d.geometry.PointCloud):
        points = np.asarray(x.points)
        return trimesh.points.PointCloud(vertices=points)
    else:
        print(type(x))
        raise NotImplementedError()


def is_trimesh(x):
    return isinstance(x, (trimesh.Trimesh, trimesh.points.PointCloud))
