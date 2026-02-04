import numpy as np

def get_coord_transformations(positions: np.ndarray):
    """Get original2normalized and normalized2original coordinate transformations for this scene.

    Args:
        positions (np.ndarray): (T, N, 3) gaussian means through time

    Returns:
    Tuple[Callable, Callable]:
        - point_original2normalized: Callable(original_positions: np.ndarray) -> np.ndarray: normalized_positions
        - point_normalized2original: Callable(normalized_positions: np.ndarray) -> np.ndarray: original_positions
        - distance_original2normalized: Callable(original_distance: float) -> normalized_distance: float
        - distance_normalized2original: Callable(normalized_distance: float) -> original_distance: float
    """
    assert positions.shape[-1] == 3, "last dim must be 3"
    # flat_pos = positions.reshape(-1, 3)
    # mean_pos = flat_pos.mean(axis=0)
    # std_pos = flat_pos.std(axis=0) + 1e-8

    # def point_original2normalized(x):
    #     return (x - mean_pos) / std_pos

    # def point_normalized2original(x):
    #     return x * std_pos + mean_pos

    # def distance_original2normalized(x):
    #     return x / std_pos

    # def distance_normalized2original(x):
    #     return x * std_pos

    # return point_original2normalized, point_normalized2original, distance_original2normalized, distance_normalized2original

    # da3 is rougly metric and origin is in center so we just go from m to cm
    def point_original2normalized(x):
        return x * 100

    def point_normalized2original(x):
        return x / 100

    def distance_original2normalized(x):
        return x * 100

    def distance_normalized2original(x):
        return x / 100

    return point_original2normalized, point_normalized2original, distance_original2normalized, distance_normalized2original