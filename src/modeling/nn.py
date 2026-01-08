from src.modeling.node import Node
from src.modeling.node import NodeTypes


class NN:
    def __init__(self, input_size, output_size, act=sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = []
        self.act = act

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
