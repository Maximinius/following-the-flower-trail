def precompute_and_save_solutions(
        input_path: str = "pre_generated_mazes.npy",
        output_path: str | None = None,
):
    """
    Load every maze in ``input_path``, solve it once with BFS, and
    save a *list of dicts*  [{'maze': <2-D np.ndarray>, 'solution': [(x,y), …]}, …]
    back to disk.  If *output_path* is omitted the original file is
    overwritten **in-place** (safe – numpy uses a temp file first).

    Example
    -------
    >>> from maze_module import precompute_and_save_solutions
    >>> precompute_and_save_solutions()               # overwrites the file
    >>> precompute_and_save_solutions(
    ...     "pre_generated_mazes.npy",
    ...     "pre_generated_mazes_with_solutions.npy") # keeps the original
    """
    import os
    import numpy as np
    from maze_module_experimental import solve_maze   # <-- your existing BFS

    out = output_path or input_path
    mazes = np.load(input_path, allow_pickle=True)
    processed = []

    for idx, m in enumerate(mazes, 1):
        if not isinstance(m, np.ndarray) or m.ndim != 2:
            raise TypeError(
                f"Entry {idx} is not a 2-D maze grid – "
                f"got type {type(m)} with shape {getattr(m, 'shape', None)}"
            )
        sol = solve_maze(m)
        if not sol:
            raise ValueError(f"Maze #{idx} has no path from top to bottom.")
        processed.append({"maze": m, "solution": sol})

    np.save(out, np.array(processed, dtype=object), allow_pickle=True)
    print(
        f"✓   Saved {len(processed)} mazes -with-solutions -> "
        f"{os.path.abspath(out)}"
    )
precompute_and_save_solutions()               # overwrites the file
