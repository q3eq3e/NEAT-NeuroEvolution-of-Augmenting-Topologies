from enum import Enum
from src.modeling.connection import Connection


class NodeTypes(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Node:
    def __init__(self, index, type, bias, act, layer):
        self.index = index
        self.type = type
        self.bias = bias
        self.act = act
        self.layer = layer

        # self.weights = []
        # self.inputs = []
        self.connections = []
        # self.outputs = []
        self.out = 0.0
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
        return ValueError("Input node already connected.")
        # self.weights.append(input_weight)
        # self.inputs.append(input_node)

        # for node in self.inputs:
        #     if node.layer >= self.layer:
        #         self.layer = node.layer + 1

    def rm_input(self, input_node):
        for conn in self.connections:
            if conn.get_source_node() == input_node:
                self.connections.disable()
                return

    def calculate_output(self):
        if self.type == NodeTypes.INPUT:
            return ValueError("Input nodes do not calculate output.")

        sum = 0.0
        for conn in self.get_active_connections():
            sum += conn.get_source_node().out * conn.get_weight()

        sum += self.bias
        return self.act(sum)

    def set_output(self, output):
        self.out = output
