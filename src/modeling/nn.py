from src.modeling.node import Node, Connection
from src.modeling.node import NodeTypes
from src.modeling.activation import sigmoid, identity
from copy import deepcopy
from typing import Callable, Optional
import random


class NN:
    def __init__(
        self, input_size: int, output_size: int, act: Callable[[float], float] = sigmoid
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.act = act
        self.connections = []
        self.nodes = []
        for i in range(input_size):
            self.nodes.append(Node(i, NodeTypes.INPUT, 0.0, identity, 0))
        for i in range(output_size):
            self.nodes.append(Node(input_size + i, NodeTypes.OUTPUT, 0.0, sigmoid, 1))
        for i in range(output_size):
            for j in range(input_size):
                self.add_connection(
                    self.nodes[j],
                    self.nodes[input_size + i],
                    i * input_size + j,
                    weight=random.uniform(-1.0, 1.0),
                )

    @staticmethod
    def create_from_parent(parent, infos: dict) -> "NN":
        nn = deepcopy(parent.get_nn())
        nn.connections.sort(key=lambda conn: conn.innovation_number)
        infos.sort(key=lambda info: info["innovation_number"])
        for conn, info in zip(nn.connections, infos):
            if conn.innovation_number == info["innovation_number"]:
                conn.set_weight(info["weight"])
                conn.enabled = info["enabled"]
            else:
                raise ValueError("you should not be here")
        return nn

    def get_nodes_indices(self) -> list[int]:
        return [node.index for node in self.nodes]

    def get_connection(self, from_node: Node, to_node: Node) -> Connection:
        for conn in self.connections:
            if (
                conn.get_source_node() == from_node
                and conn.get_target_node() == to_node
            ):
                return conn
        return None

    def conn_exists(self, from_node: Node, to_node: Node) -> bool:
        return self.get_connection(from_node, to_node) is not None

    def add_connection(
        self,
        from_node: Node,
        to_node: Node,
        innovation_number: int,
        weight: float = 1.0,
    ) -> None:
        new_connection = to_node.add_input(from_node, weight, innovation_number)
        self.connections.append(new_connection)

    def _remove_connection(self, connection: Connection) -> None:
        connection.disable()

    def active_connections(self) -> list[Connection]:
        return [conn for conn in self.connections if conn.enabled]

    def add_node(
        self, connection: Connection, innovation: int, act=None, bias=0.0, out=0.0
    ):
        if act is None:
            act = self.act

        new_node_layer = self._adjust_layers(
            connection.get_source_node(), connection.get_target_node()
        )
        new_node = Node(
            len(self.nodes), NodeTypes.HIDDEN, bias, act, new_node_layer, out
        )
        removed_weight = connection.get_weight()
        self.nodes.append(new_node)
        self.add_connection(
            connection.get_source_node(), new_node, innovation, weight=1.0
        )
        self.add_connection(
            new_node,
            connection.get_target_node(),
            innovation + 1,
            weight=removed_weight,
        )
        self._remove_connection(connection)
        self.nodes.sort(key=lambda node: node.layer)

    def _adjust_layers(self, connection_from, connection_to):
        layer_diff = connection_to.layer - connection_from.layer
        if layer_diff > 1:
            new_node_layer = connection_from.layer + 1
        elif layer_diff == 1:
            # Need to insert a new layer
            dst_layer = connection_to.layer
            for node in self.nodes:
                if node.layer >= dst_layer:
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
            src_layer = connection_from.layer
            for node in self.nodes:
                if node.layer >= src_layer:
                    node.layer += 1
            new_node_layer = connection_to.layer + 1
        else:  # layer_diff < -1
            new_node_layer = connection_to.layer + 1
        return new_node_layer

    def calculate_output(self, inputs):
        i = 0
        outputs = []
        while i < len(self.nodes) and self.nodes[i].type == NodeTypes.INPUT:
            self.nodes[i].set_output(inputs[i])
            i += 1
        for layer in range(1, self.nodes[-1].layer + 1):
            j = i
            outputs = []
            while i < len(self.nodes) and self.nodes[i].layer == layer:
                outputs.append(self.nodes[i].calculate_output())
                i += 1
            for k, node in enumerate(self.nodes[j:i]):
                node.set_output(outputs[k])

        return outputs

    def __str__(self):
        neurons_in_layer = [
            [n for n in self.nodes if n.layer == i]
            for i in range(self.nodes[-1].layer + 1)
        ]
        pos_neurons = [
            {
                "y": int(n.layer * 20),
                "x": int(
                    50
                    * (neurons_in_layer[n.layer].index(n) + 1)
                    / (len(neurons_in_layer[n.layer]) + 1)
                ),
            }
            for n in self.nodes
        ]
        print("------ Neural Network Visualization ------")
        print(pos_neurons)

        max_x = max(p["x"] for p in pos_neurons)
        max_y = max(p["y"] for p in pos_neurons)

        W = int(max_x + 3)
        H = int(max_y + 3)

        canvas = [[" " for _ in range(W)] for _ in range(H)]

        # edge drawing
        for conn in self.active_connections():
            src = conn.get_source_node()
            dst = conn.get_target_node()
            x1, y1 = (
                pos_neurons[self.nodes.index(src)]["x"],
                pos_neurons[self.nodes.index(src)]["y"],
            )
            x2, y2 = (
                pos_neurons[self.nodes.index(dst)]["x"],
                pos_neurons[self.nodes.index(dst)]["y"],
            )
            print(f"Conn from ({x1},{y1}) to ({x2},{y2})")
            dx = x2 - x1
            dy = y2 - y1

            steps = max(abs(dx), abs(dy))
            if steps == 0:
                canvas[y1 + 1][x1] = "/"
                canvas[y1 + 1][x1 - 1] = "-"
                canvas[y1 + 1][x1 - 2] = "\\"
                canvas[y1][x1 - 2] = "|"
                canvas[y1 - 1][x1 - 2] = "/"
                canvas[y1 - 1][x1 - 1] = "-"
                canvas[y1 - 1][x1] = "\\"
            else:
                for i in range(1, steps):
                    x = x1 + dx * i // steps
                    y = y1 + dy * i // steps
                    x = round(x)
                    y = round(y)

                    if dx == 0:
                        canvas[y][x] = "|"
                    elif dy == 0:
                        canvas[y][x] = "-"
                    elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                        canvas[y][x] = "\\"
                    else:
                        canvas[y][x] = "/"

                x2 = x1 + dx * (steps - 1) // steps
                y2 = y1 + dy * (steps - 1) // steps

            x2 = round(x2)
            y2 = round(y2)
            # arrow direction
            canvas[y2][x2] = "●"

        for i, n in enumerate(pos_neurons):
            canvas[n["y"]][n["x"]] = str(i)

        return "\n".join("".join(row) for row in canvas)
