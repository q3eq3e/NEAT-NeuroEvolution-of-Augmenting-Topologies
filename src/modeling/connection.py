from src.modeling.node import Node
from src.modeling.node import NodeTypes


class Connection:
    def __init__(
        self,
        from_node: Node,
        to_node: Node,
        weight: float,
        innovation_number: int,
        enabled: bool = True,
    ):
        if to_node.type == NodeTypes.INPUT:
            raise ValueError("Cannot create connection to input node.")
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.innovation_number = innovation_number
        self.enabled = enabled

    def __str__(self):
        return (
            f"Connection(from: {self.from_node.index}, to: {self.to_node.index}, weight: {self.weight}, innovation_number: {self.innovation_number}, "
            + "Enabled"
            if self.enabled
            else "DISABLED" + ")"
        )

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def get_source_node(self):
        return self.from_node

    def get_target_node(self):
        return self.to_node

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True
