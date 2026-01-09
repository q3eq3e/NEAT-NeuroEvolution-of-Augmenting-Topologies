from src.modeling.nn import NN
from src.modeling.genome import Genome
from src.modeling.node import Connection
from src.modeling.node import Node
from src.modeling.node import NodeTypes

if __name__ == "__main__":
    # nn = NN(3, 2)
    # print(nn.calculate_output([1.0, 1.0, 1.0]))
    # nn.add_connection(nn.nodes[3], nn.nodes[3], 4, 0.5)
    # nn.add_connection(nn.nodes[3], nn.nodes[4], 5, 0.5)
    # nn.active_connections()[0].disable()
    # print(nn.calculate_output([1.0, 1.0, 1.0]))
    # nn.active_connections()[0].set_weight(-0.5)
    # print(nn.calculate_output([1.0, 1.0, 1.0]))
    # nn.add_node(nn.active_connections()[0], 7)
    # print(nn.calculate_output([1.0, 1.0, 1.0]))
    # nn.add_node(nn.active_connections()[1], 8)
    # nn.add_node(nn.active_connections()[1], 9)
    # nn.add_node(nn.active_connections()[-1], 9)
    # nn.add_node(nn.active_connections()[4], 9)

    # nn.add_connection(nn.nodes[-3], nn.nodes[-2], 9, -1.5)
    # nn.add_node(nn.active_connections()[-1], 10)
    # # nn.add_connection(nn.nodes[1], nn.nodes[0], 1.5)
    # nn.add_connection(nn.nodes[5], nn.nodes[3], 10, 0.3)
    # nn.add_node(nn.active_connections()[-1], 11)
    # nn.add_connection(nn.nodes[-1], nn.nodes[4], 11, -1.5)
    # nn.add_node(nn.active_connections()[-1], 11)
    # nn.add_node(nn.active_connections()[3], 11)
    # print(nn)
    # print(nn.calculate_output([1.0, 1.0, 1.0]))
    # genome = Genome.create_from_nn(nn)
    # for gene in genome.get_genes():
    #     print(gene)
    # nn2 = genome.get_nn()
    # print(nn2)

    genes = []
    A = Node(1, NodeTypes.INPUT, 0.0, None, 0)
    B = Node(2, NodeTypes.HIDDEN, 0.0, None, 1)
    C = Node(3, NodeTypes.HIDDEN, 0.0, None, 2)
    D = Node(4, NodeTypes.INPUT, 0.0, None, 0)
    E = Node(5, NodeTypes.OUTPUT, 0.0, None, 3)
    genes.append(Connection(A, B, 1.0, 0))
    genes.append(Connection(B, C, 1.0, 0))
    genes.append(Connection(C, C, 1.0, 0))
    genes.append(Connection(D, B, 1.0, 0))
    genes.append(Connection(C, E, 1.0, 0))
    genes.append(Connection(E, B, 1.0, 0))
    genome = Genome(genes)
    print(genome)
    nn = genome.get_nn()
    print(nn)
