from src.modeling.node import Node
from src.modeling.node import NodeTypes
from src.modeling.activation import sigmoid


class NN:
    def __init__(self, input_size, output_size, act=sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = []
        for i in range(input_size):
            self.nodes.append(Node(i, NodeTypes.INPUT, 1.0, act, 0))
        for i in range(output_size):
            self.nodes.append(Node(input_size + i, NodeTypes.OUTPUT, 0.0, act, 1))
        self.act = act
        for i in range(output_size):
            self.add_connection(self.nodes[0], self.nodes[input_size + i])

    def add_connection(self, from_node, to_node, weight=1.0):
        to_node.add_input(from_node, weight)

    def remove_connection(self, from_node, to_node):
        to_node.rm_input(from_node)

    def add_node(self, connection_from, connection_to, act=None, bias=0.0):
        if act is None:
            act = self.act

        new_node_layer = self.adjust_layers(connection_from, connection_to)
        new_node = Node(len(self.nodes), NodeTypes.HIDDEN, bias, act, new_node_layer)
        removed_weight = connection_to.weights[
            connection_to.inputs.index(connection_from)
        ]
        self.add_connection(connection_from, new_node, weight=1.0)
        self.add_connection(new_node, connection_to, weight=removed_weight)
        connection_to.rm_input(connection_from)
        self.nodes.append(new_node)
        self.nodes.sort(key=lambda node: node.layer)

    def adjust_layers(self, connection_from, connection_to):
        layer_diff = connection_to.layer - connection_from.layer
        if layer_diff > 1:
            new_node_layer = connection_from.layer + 1
        elif layer_diff == 1:
            # Need to insert a new layer
            for node in self.nodes:
                if node.layer >= connection_to.layer:
                    node.layer += 1
            new_node_layer = connection_from.layer + 1
        elif layer_diff == 0:
            if connection_from.type == NodeTypes.OUTPUT:
                for node in self.nodes:
                    if node.type == NodeTypes.OUTPUT:
                        node.layer += 1
                new_node_layer = connection_from.layer - 1
            else:
                new_node_layer = connection_from.layer
        elif layer_diff == -1:
            # Need to insert a new layer
            for node in self.nodes:
                if node.layer >= connection_from.layer:
                    node.layer += 1
            new_node_layer = connection_to.layer + 1
        else:  # layer_diff < -1
            new_node_layer = connection_to.layer + 1
        return new_node_layer

    def calculate_output(self, inputs):
        i = 0
        while i < len(self.nodes) and self.nodes[i].type == NodeTypes.INPUT:
            self.nodes[i].set_output(inputs[i])
            i += 1
        for layer in range(1, self.nodes[-1].layer + 1):
            j = i
            outputs = []
            while i < len(self.nodes) and self.nodes[i].layer == layer:
                outputs.append(self.nodes[i].calculate_output())
                i += 1
            for node in self.nodes[j:i]:
                node.set_output(outputs.pop(0))
        output = []
        i -= 1
        while self.nodes[i].type == NodeTypes.OUTPUT:
            output = [self.nodes[i].out] + output
            i -= 1
        return output
