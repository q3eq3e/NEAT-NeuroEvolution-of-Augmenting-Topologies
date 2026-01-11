from src.modeling.node import Node
from src.modeling.node import NodeTypes

# from src.modeling.node import Connection
from src.modeling.activation import sigmoid, identity
from copy import copy, deepcopy


class NN:
    def __init__(self, input_size, output_size, act=sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.connections = []
        self.nodes = []
        for i in range(input_size):
            self.nodes.append(Node(i, NodeTypes.INPUT, 0.0, identity, 0))
        for i in range(output_size):
            self.nodes.append(Node(input_size + i, NodeTypes.OUTPUT, 0.0, act, 1))
        self.act = act
        for i in range(output_size):
            # self.add_connection(self.nodes[0], self.nodes[input_size + i], i)
            for j in range(input_size):
                self.add_connection(
                    self.nodes[j], self.nodes[input_size + i], i * input_size + j
                )

    def create_from_genome(genes, act=sigmoid):
        unique_nodes = set()
        genes = copy(genes)
        for conn in genes:
            # unique_nodes.add(conn.get_source_node())
            # unique_nodes.add(conn.get_target_node())
            node = conn.get_source_node()
            add_node = True
            if node.type == NodeTypes.INPUT or node.type == NodeTypes.OUTPUT:
                for u in unique_nodes:
                    if u.index == node.index:
                        add_node = False
                        conn.source_node = u
                        break
            if add_node:
                conn.source_node = copy(conn.get_source_node())
                unique_nodes.add(conn.source_node)

            node = conn.get_target_node()
            add_node = True
            if node.type == NodeTypes.OUTPUT:
                for u in unique_nodes:
                    if u.index == node.index:
                        add_node = False
                        conn.target_node = u
                        break
            if add_node:
                conn.target_node = copy(conn.get_target_node())
                unique_nodes.add(conn.target_node)

        unique_nodes = list(unique_nodes)
        input_size = len([n for n in unique_nodes if n.type == NodeTypes.INPUT])
        output_size = len([n for n in unique_nodes if n.type == NodeTypes.OUTPUT])
        nn = NN(input_size, output_size, act)
        nn.nodes = unique_nodes
        nn.connections = genes
        # print(len([n for n in nn.nodes if n.type == NodeTypes.OUTPUT]))

        nn.nodes.sort(key=lambda node: node.layer)
        return nn

    def get_nodes_indices(self):
        return [node.index for node in self.nodes]

    def get_connection(self, from_node, to_node):
        for conn in self.connections:
            if (
                conn.get_source_node() == from_node
                and conn.get_target_node() == to_node
            ):
                return conn
        return None

    def conn_exists(self, from_node, to_node):
        return self.get_connection(from_node, to_node) is not None

    def add_connection(self, from_node, to_node, innovation_number, weight=1.0):
        new_connection = to_node.add_input(from_node, weight, innovation_number)
        self.connections.append(new_connection)

    # def remove_connection(self, from_node, to_node):
    #     to_node.rm_input(from_node)

    def _remove_connection(self, connection):
        connection.disable()
        # connection.get_target_node().rm_input(connection.get_source_node())

    def active_connections(self):
        return [conn for conn in self.connections if conn.enabled]

    def add_node(self, connection, innovation, act=None, bias=0.0, out=0.0):
        if act is None:
            act = self.act

        new_node_layer = self._adjust_layers(
            connection.get_source_node(), connection.get_target_node()
        )
        new_node = Node(
            len(self.nodes), NodeTypes.HIDDEN, bias, act, new_node_layer, out
        )
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
        self._remove_connection(connection)
        self.nodes.append(new_node)
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
        for layer in range(0, self.nodes[-1].layer + 1):
            j = i
            outputs = []
            while i < len(self.nodes) and self.nodes[i].layer == layer:
                # if layer == self.nodes[-1].layer:
                #     print(self.nodes[i])
                #     print(self.nodes[i].calculate_output())
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

        # rysowanie krawędzi
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
                # rysowanie grotu strzałki
                x2 = x1 + dx * (steps - 1) // steps
                y2 = y1 + dy * (steps - 1) // steps

            x2 = round(x2)
            y2 = round(y2)
            # grot strzałki
            canvas[y2][x2] = "●"

        # rysowanie neuronów
        for i, n in enumerate(pos_neurons):
            canvas[n["y"]][n["x"]] = str(i)

        return "\n".join("".join(row) for row in canvas)
