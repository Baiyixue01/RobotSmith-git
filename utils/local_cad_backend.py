import numpy as np
import trimesh


def _select_primitive(prompt: str):
    text = (prompt or "").lower()
    if any(k in text for k in ("sphere", "ball", "orb")):
        return "ball"
    if any(k in text for k in ("cylinder", "pillar", "tube", "rod")):
        return "cylinder"
    return "cube"


def _estimate_scale(prompt: str):
    text = (prompt or "").lower()
    if "tiny" in text:
        return 0.04
    if "small" in text:
        return 0.07
    if "large" in text:
        return 0.16
    if "huge" in text:
        return 0.22
    return 0.1


def text_to_mesh(prompt) -> trimesh.Trimesh:
    """
    Local text-to-mesh adapter.

    Args:
        prompt (str): Natural language prompt.

    Returns:
        trimesh.Trimesh: A watertight mesh in the project coordinate space.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("text_to_mesh() requires a non-empty prompt string.")

    primitive = _select_primitive(prompt)
    base = _estimate_scale(prompt)

    if primitive == "ball":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=base)
    elif primitive == "cylinder":
        mesh = trimesh.creation.cylinder(radius=base * 0.6, height=base * 2.2, sections=64)
    else:
        extents = np.array([1.2, 1.0, 0.8], dtype=float) * base * 2.0
        mesh = trimesh.creation.box(extents=extents)

    if mesh is None:
        raise RuntimeError("Local backend failed to produce a mesh object.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("Local backend produced an empty mesh.")

    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh
