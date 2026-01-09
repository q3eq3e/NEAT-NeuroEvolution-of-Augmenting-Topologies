from src.modeling.node import Node
from src.modeling.node import NodeTypes
from src.modeling.connection import Connection
from src.modeling.activation import sigmoid, identity


class NN:
    def __init__(self, input_size, output_size, act=sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.connections = []
        self.nodes = []
        for i in range(input_size):
            self.nodes.append(
                Node(i, NodeTypes.INPUT, 1.0, identity, 0)
            )  # dlaczego bias 1.0?
        for i in range(output_size):
            self.nodes.append(Node(input_size + i, NodeTypes.OUTPUT, 0.0, act, 1))
        self.act = act
        for i in range(output_size):
            # self.add_connection(self.nodes[0], self.nodes[input_size + i])
            for j in range(input_size):
                self.add_connection(self.nodes[j], self.nodes[input_size + i])

    def add_connection(self, from_node, to_node, innovation_number, weight=1.0):
        new_connection = to_node.add_input(from_node, weight, innovation_number)
        self.connections.append(new_connection)

    def remove_connection(self, from_node, to_node):
        to_node.rm_input(from_node)

    def remove_connection_obj(self, connection):
        connection.get_target_node().rm_input(connection.get_source_node())

    def active_connections(self):
        return [conn for conn in self.connections if conn.enabled]

    def add_node(self, connection, innovation, act=None, bias=0.0):
        if act is None:
            act = self.act

        new_node_layer = self.adjust_layers(
            connection.get_source_node(), connection.get_target_node()
        )
        new_node = Node(len(self.nodes), NodeTypes.HIDDEN, bias, act, new_node_layer)
        removed_weight = connection.get_weight()
        self.add_connection(
            connection.get_source_node(), new_node, innovation, weight=1.0
        )
        self.add_connection(
            new_node,
            connection.get_target_node(),
            innovation + 1,
            weight=removed_weight,
        )
        self.remove_connection_obj(connection)
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
