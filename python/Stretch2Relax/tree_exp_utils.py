from __future__ import annotations

import gzip
import pickle
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

Node = Dict[str, Any]


def _copy_value(x: np.ndarray) -> np.ndarray:
    """Return a defensive copy of a numpy array."""
    return np.array(x, copy=True)


def build_perfect_binary_tree(
    x0: np.ndarray,
    funcA: Callable[[np.ndarray], np.ndarray],
    funcB: Callable[[np.ndarray], np.ndarray],
    height: int,
    *,
    copy_values: bool = True,
    left_label: str = "A",
    right_label: str = "B",
    eval_x = None,
) -> Node:
    """
    Build a perfect binary tree of intermediate iterates.

    The tree uses a map-based representation:
        {
            'val': <numpy array>,
            'depth': <int>,
            'path': <str>,
            'op_name': <str | None>,
            'left': {...},   # optional
            'right': {...},  # optional
        }

    Parameters
    ----------
    x0
        Root value.
    funcA, funcB
        Update functions. For a node value x:
            left child  = funcA(x)
            right child = funcB(x)
    height
        Number of edges from the root to each leaf.
        height=0 returns a tree containing only the root.
    copy_values
        Whether to store defensive copies of the arrays.
    left_label, right_label
        Labels appended to the path when descending left/right.

    Returns
    -------
    dict
        Nested dict representing the perfect binary tree.
    """
    if not isinstance(x0, np.ndarray):
        raise TypeError("x0 must be a numpy.ndarray")
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D numpy array")
    if height < 0:
        raise ValueError("height must be nonnegative")

    def store(x: np.ndarray) -> np.ndarray:
        return _copy_value(x) if copy_values else x

    def _build(x: np.ndarray, depth: int, path: str, op_name: Optional[str]) -> Node:
        node: Node = {
            "val": store(x),
            "depth": depth,
            "path": path,
            "op_name": op_name,
            "stats": None if eval_x is None else eval_x(_copy_value(x) if copy_values else x),
        }
        if depth == height:
            return node

        x_left = funcA(_copy_value(x) if copy_values else x)
        x_right = funcB(_copy_value(x) if copy_values else x)

        if not isinstance(x_left, np.ndarray) or x_left.ndim != 1:
            raise TypeError("funcA must return a 1D numpy.ndarray")
        if not isinstance(x_right, np.ndarray) or x_right.ndim != 1:
            raise TypeError("funcB must return a 1D numpy.ndarray")

        node["left"] = _build(x_left, depth + 1, path + left_label, left_label)
        node["right"] = _build(x_right, depth + 1, path + right_label, right_label)
        return node

    return _build(x0, depth=0, path="", op_name=None)


def is_leaf(node: Node) -> bool:
    return "left" not in node and "right" not in node


def get_node_by_path(
    tree: Node,
    path: str,
    *,
    left_label: str = "A",
    right_label: str = "B",
) -> Node:
    """Return the node reached by following a path like '', 'A', 'BAB'."""
    node = tree
    for step in path:
        if step == left_label:
            if "left" not in node:
                raise KeyError(f"Path '{path}' does not exist")
            node = node["left"]
        elif step == right_label:
            if "right" not in node:
                raise KeyError(f"Path '{path}' does not exist")
            node = node["right"]
        else:
            raise ValueError(
                f"Invalid path step '{step}'. Expected '{left_label}' or '{right_label}'."
            )
    return node


def get_value_by_path(
    tree: Node,
    path: str,
    *,
    field: str = "val",
    left_label: str = "A",
    right_label: str = "B",
) -> Any:
    """Return a requested field at the node specified by path."""
    node = get_node_by_path(
        tree, path, left_label=left_label, right_label=right_label
    )
    if field not in node:
        raise KeyError(f"Field '{field}' not found at path '{path}'")
    return node[field]


def iter_nodes_preorder(tree: Node) -> Iterator[Node]:
    """Yield all nodes in preorder traversal."""
    yield tree
    if "left" in tree:
        yield from iter_nodes_preorder(tree["left"])
    if "right" in tree:
        yield from iter_nodes_preorder(tree["right"])


def iter_leaves(tree: Node) -> Iterator[Node]:
    """Yield leaf nodes from left to right."""
    if is_leaf(tree):
        yield tree
        return
    if "left" in tree:
        yield from iter_leaves(tree["left"])
    if "right" in tree:
        yield from iter_leaves(tree["right"])


def collect_leaf_values(tree: Node, *, field: str = "val") -> Dict[str, Any]:
    """Return a dict mapping each leaf path to the requested node field."""
    out: Dict[str, Any] = {}
    for node in iter_leaves(tree):
        path = node["path"]
        if field not in node:
            raise KeyError(f"Field '{field}' not found at path '{path}'")
        out[path] = node[field]
    return out


def collect_all_values(tree: Node, *, field: str = "val") -> Dict[str, Any]:
    """Return a dict mapping each node path to the requested node field."""
    out: Dict[str, Any] = {}
    for node in iter_nodes_preorder(tree):
        path = node["path"]
        if field not in node:
            raise KeyError(f"Field '{field}' not found at path '{path}'")
        out[path] = node[field]
    return out


def evaluate_path(
    x0: np.ndarray,
    path: str,
    funcA: Callable[[np.ndarray], np.ndarray],
    funcB: Callable[[np.ndarray], np.ndarray],
    *,
    left_label: str = "A",
    right_label: str = "B",
) -> np.ndarray:
    """Apply the sequence encoded by path directly, without building the full tree."""
    x = _copy_value(x0)
    for step in path:
        if step == left_label:
            x = funcA(x)
        elif step == right_label:
            x = funcB(x)
        else:
            raise ValueError(
                f"Invalid path step '{step}'. Expected '{left_label}' or '{right_label}'."
            )
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise TypeError("Each function application must return a 1D numpy.ndarray")
    return x


def tree_to_records(tree: Node) -> List[Dict[str, Any]]:
    """
    Flatten the tree into a list of row-like dicts.

    Each record contains metadata plus the stored vector itself.
    If a node has a ``stats`` dict, each stats key is added as a scalar column.
        {
            'path': 'ABA',
            'depth': 3,
            'op_name': 'A',
            'is_leaf': True,
            'parent_path': 'AB',
            'branch': 'A',
            'energy': 2.0,      # from node['stats']
            'grad_norm': 1.35,  # from node['stats']
            'val': np.ndarray(...),
        }
    """
    records: List[Dict[str, Any]] = []
    for node in iter_nodes_preorder(tree):
        path = node["path"]
        row = {
            "path": path,
            "depth": node["depth"],
            "op_name": node["op_name"],
            "is_leaf": is_leaf(node),
            "parent_path": path[:-1] if path else None,
            "branch": path[-1] if path else None,
            "val": node["val"],
        }

        stats = node.get("stats")
        if stats is not None:
            if not isinstance(stats, dict):
                raise TypeError("node['stats'] must be a dict[str, float] when present")
            for key, value in stats.items():
                if not isinstance(key, str):
                    raise TypeError("node['stats'] keys must be strings")
                row[key] = float(value)

        records.append(row)
    return records


def tree_to_flat_records(
    tree: Node,
) -> List[Dict[str, Any]]:
    """
    Flatten the tree into table-ready records while keeping full vectors.

    Unlike coordinate-expansion layouts, each output row stores the full
    1D vector in a single ``val`` column as a Python list, which is
    directly compatible with polars list columns. Stats columns from
    ``tree_to_records`` are preserved in each row.

    Example output row:
        {
            'path': 'AB',
            'depth': 2,
            'is_leaf': False,
            'val': [...],
            'energy': 2.0049,
            'grad_norm': 1.3528,
        }

    This is convenient for pandas/polars/DataFrame workflows.
    """
    out_rows: List[Dict[str, Any]] = []
    for row in tree_to_records(tree):
        out = dict(row)
        val = out.get("val")
        if isinstance(val, np.ndarray):
            out["val"] = val.tolist()
        out_rows.append(out)
    return out_rows


def pretty_format_tree(tree: Node, *, max_array_chars: int = 60) -> str:
    """Return a readable multi-line string representation of the tree."""

    def fmt_arr(arr: np.ndarray) -> str:
        s = np.array2string(arr, precision=4, separator=", ")
        return s if len(s) <= max_array_chars else s[: max_array_chars - 3] + "..."

    lines: List[str] = []

    def _walk(node: Node, indent: str) -> None:
        label = node["path"] if node["path"] else "<root>"
        lines.append(f"{indent}{label}: {fmt_arr(node['val'])}")
        if "left" in node:
            _walk(node["left"], indent + "  ")
        if "right" in node:
            _walk(node["right"], indent + "  ")

    _walk(tree, "")
    return "\n".join(lines)


def save_tree_pickle(
    tree: Node,
    file_path: str,
    *,
    use_gzip: Optional[bool] = None,
) -> None:
    """
    Save a tree to disk using Python pickle.

    Parameters
    ----------
    tree
        Nested-dict tree to serialize.
    file_path
        Output file path. If it ends with ".gz" and use_gzip is None,
        gzip compression is enabled automatically.
    use_gzip
        Whether to gzip-compress the pickle payload.
        - True: force gzip
        - False: plain pickle
        - None: infer from file extension (".gz")
    """
    should_gzip = file_path.endswith(".gz") if use_gzip is None else use_gzip
    open_fn = gzip.open if should_gzip else open
    with open_fn(file_path, "wb") as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tree_pickle(file_path: str, *, use_gzip: Optional[bool] = None) -> Node:
    """
    Load a tree from a pickle file and validate the root shape.

    Parameters
    ----------
    file_path
        Input file path. If it ends with ".gz" and use_gzip is None,
        gzip decompression is enabled automatically.
    use_gzip
        Whether to expect gzip-compressed pickle content.
        - True: force gzip
        - False: plain pickle
        - None: infer from file extension (".gz")
    """
    should_gzip = file_path.endswith(".gz") if use_gzip is None else use_gzip
    open_fn = gzip.open if should_gzip else open
    with open_fn(file_path, "rb") as f:
        tree = pickle.load(f)

    if not isinstance(tree, dict):
        raise TypeError("Loaded object is not a tree dict")
    if "val" not in tree or "depth" not in tree or "path" not in tree:
        raise ValueError("Loaded dict does not match expected tree node schema")
    return tree


__all__ = [
    "Node",
    "build_perfect_binary_tree",
    "is_leaf",
    "get_node_by_path",
    "get_value_by_path",
    "iter_nodes_preorder",
    "iter_leaves",
    "collect_leaf_values",
    "collect_all_values",
    "evaluate_path",
    "tree_to_records",
    "tree_to_flat_records",
    "pretty_format_tree",
    "save_tree_pickle",
    "load_tree_pickle",
]
