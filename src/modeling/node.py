from enum import Enum


class NodeTypes(Enum):
    HIDDEN = 1
    INPUT = 2
    OUTPUT = 3


class Node:
    def __init__(self, index, type, bias, act, layer):
        self.index = index
        self.type = type
        self.bias = bias
        self.act = act
        self.layer = layer

        self.weights = []
        self.inputs = []
        # self.outputs = []
        self.out = 0.0
        if self.type == NodeTypes.INPUT:
            self.layer = 0
            self.weights = [1.0]

        if self.type == NodeTypes.OUTPUT:
            self.layer = 1

    def add_input(self, input_node, input_weight):
        if input_node not in self.inputs:
            self.weights.append(input_weight)
            self.inputs.append(input_node)
            # for node in self.inputs:
            #     if node.layer >= self.layer:
            #         self.layer = node.layer + 1

    def rm_input(self, input_node):
        index = self.inputs.index(input_node)
        self.weights.pop(index)
        self.inputs.pop(index)

    def calculate_output(self):
        if self.type == NodeTypes.INPUT:
            return self.out

        sum = 0.0
        for i in range(len(self.inputs)):
            sum += self.inputs[i].out * self.weights[i]

        sum += self.bias
        return self.act(sum)

    def set_output(self, output):
        self.out = output
