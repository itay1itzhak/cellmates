import torch


class Sample:
    """
    A set of cells within a radius r from a responding cell.
    """

    def __init__(
        self,
        cell_types: torch.Tensor,
        distances: torch.Tensor,
        responder_cell_type: int,
        is_dividing: bool,
    ) -> None:
        """Initialize a sample

        Args:
            cell_types (list[int]): types of all cells in the sample (including the responding cell)
            distances (np.ndarray): n cells x n cells matrix of distances
            responder_cell_type (int): type of the responding cell
            is_dividing (bool): is the central cell dividing
        """
        self.cell_types = cell_types
        self.distances = distances
        self.responder_cell_type = responder_cell_type
        self.is_dividing = is_dividing

        self.L = len(cell_types)

    def __repr__(self) -> str:
        return f"""
        responder_cell_type: {self.responder_cell_type}
        is_dividing: {self.is_dividing}
        cell_types: {self.cell_types}
        distances: \n{self.distances}
        """
    
    def __eq__(self, other) -> bool:
        return (
            torch.allclose(self.cell_types,other.cell_types) and \
            (self.is_dividing == other.is_dividing) and \
            (self.responder_cell_type == other.responder_cell_type) and \
            torch.allclose(self.distances, other.distances)
        )
        
