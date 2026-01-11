from enum import Enum


class NodeTypes(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Node:
    def __init__(self, index, type, bias, act, layer, out=0.0):
        self.index = index
        self.type = type
        self.bias = bias  # different behaviour than in the paper
        self.act = act
        self.layer = layer
        self.connections = []

        # self.weights = []
        # self.inputs = []
        # self.outputs = []
        self.out = out
        if self.type == NodeTypes.INPUT:
            self.layer = 0

        if self.type == NodeTypes.OUTPUT:
            self.layer = 1

    def get_inputs(self):
        return [conn.get_source_node() for conn in self.connections if conn.enabled]

    def get_active_connections(self):
        return [conn for conn in self.connections if conn.enabled]

    def add_input(self, input_node, input_weight, innovation_number):
        if input_node not in self.get_inputs():
            new_connection = Connection(
                input_node, self, input_weight, innovation_number
            )
            self.connections.append(new_connection)
            return new_connection
        raise ValueError("Input node already connected.")
        # self.weights.append(input_weight)
        # self.inputs.append(input_node)

        # for node in self.inputs:
        #     if node.layer >= self.layer:
        #         self.layer = node.layer + 1

    # def rm_input(self, input_node):
    #     for conn in self.connections:
    #         if conn.get_source_node() == input_node:
    #             conn.disable()
    #             return

    def calculate_output(self):
        if self.type == NodeTypes.INPUT:
            raise ValueError("Input nodes do not calculate output.")

        sum = 0.0
        for conn in self.get_active_connections():
            sum += conn.get_source_node().out * conn.get_weight()

        sum += self.bias
        return self.act(sum)

    def set_output(self, output):
        self.out = output

    def is_input_node(self):
        return self.type == NodeTypes.INPUT

    def is_output_node(self):
        return self.type == NodeTypes.OUTPUT


class Connection:
    def __init__(
        self,
        from_node,
        to_node,
        weight: float,
        innovation_number: int,
        enabled: bool = True,
    ):
        if to_node.type == NodeTypes.INPUT:
            raise ValueError("Cannot create connection to input node.")
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        if not isinstance(innovation_number, int) or innovation_number < 0:
            raise ValueError(f"Invalid innovation number: {innovation_number} .")
        self.innovation_number = innovation_number
        self.enabled = enabled

    def __str__(self):
        return (
            f"Connection(from: {self.from_node.index}, to: {self.to_node.index}, weight: {self.weight}, innovation_number: {self.innovation_number}, "
            + ("Enabled" if self.enabled else "DISABLED")
            + ")"
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

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def __eq__(self, value):
        if not isinstance(value, Connection):
            return False
        return self.from_node == value.from_node and self.to_node == value.to_node

    def __hash__(self):
        return hash((self.from_node, self.to_node))
