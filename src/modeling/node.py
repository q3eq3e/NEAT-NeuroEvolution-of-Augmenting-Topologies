from enum import Enum
from typing import Callable


class NodeTypes(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Node:
    def __init__(
        self,
        index: int,
        type: NodeTypes,
        bias: float,
        act: Callable[[float], float],
        layer: int,
        out: float = 0.0,
    ) -> None:
        self.index = index
        self.type = type
        self.bias = bias  # different behaviour than in the paper
        self.act = act
        self.layer = layer
        self.connections = []
        self.out = out

        if self.type == NodeTypes.INPUT:
            self.layer = 0

        if self.type == NodeTypes.OUTPUT:
            self.layer = 1

    def get_inputs(self) -> list["Node"]:
        return [conn.get_source_node() for conn in self.connections if conn.enabled]

    def get_active_connections(self) -> list["Connection"]:
        return [conn for conn in self.connections if conn.enabled]

    def add_input(
        self, input_node: "Node", input_weight: float, innovation_number: int
    ) -> "Connection":
        if input_node not in self.get_inputs():
            new_connection = Connection(
                input_node, self, input_weight, innovation_number
            )
            self.connections.append(new_connection)
            return new_connection
        raise ValueError("Input node already connected.")

    def calculate_output(self) -> float:
        if self.type == NodeTypes.INPUT:
            raise ValueError("Input nodes do not calculate output.")

        sum = 0.0
        for conn in self.get_active_connections():
            sum += conn.get_source_node().out * conn.get_weight()

        sum += self.bias
        return self.act(sum)

    def set_output(self, output: float) -> None:
        self.out = output

    def is_input_node(self) -> bool:
        return self.type == NodeTypes.INPUT

    def is_output_node(self) -> bool:
        return self.type == NodeTypes.OUTPUT


class Connection:
    def __init__(
        self,
        from_node: Node,
        to_node: Node,
        weight: float,
        innovation_number: int,
        enabled: bool = True,
    ) -> None:
        if to_node.type == NodeTypes.INPUT:
            raise ValueError("Cannot create connection to input node.")
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        if not isinstance(innovation_number, int) or innovation_number < 0:
            raise ValueError(f"Invalid innovation number: {innovation_number} .")
        self.innovation_number = innovation_number
        self.enabled = enabled

    def __str__(self) -> str:
        return (
            f"Connection(from: {self.from_node.index}, to: {self.to_node.index},"
            + f" weight: {self.weight}, innovation_number: {self.innovation_number}, "
            + ("Enabled" if self.enabled else "DISABLED")
            + ")"
        )

    def get_weight(self) -> float:
        return self.weight

    def set_weight(self, weight: float) -> None:
        self.weight = weight

    def get_source_node(self) -> Node:
        return self.from_node

    def get_target_node(self) -> Node:
        return self.to_node

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def __eq__(self, other: "Connection") -> bool:
        if not isinstance(other, Connection):
            return False
        return self.from_node == other.from_node and self.to_node == other.to_node

    def __hash__(self) -> int:
        return hash((self.from_node, self.to_node))
