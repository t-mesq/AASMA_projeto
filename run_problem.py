# from graph import *
import random  # to generate random distances while there is no connection to the API
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import pylab as pl
from agent import *


def read_adresses_input(filename):
    nodes = []
    nodes_adresses = []
    read_nodes = 0

    doc = open(filename, "r").readlines()

    for line in doc:
        read_parameter, value = line.split(' ', 1)
        if read_parameter == 'capacity':
            capacity = int(value)
        elif read_parameter == 'schools':
            school_nodes = list(map(int, value.split(' ')))
        elif read_parameter == 'nodes':
            read_nodes = int(value)
        elif read_nodes > 0:
            nodes.append(list(map(int, line.split())))
            read_nodes -= 1
        else:
            nodes_adresses.append(eval(line))

    return capacity, school_nodes, nodes, nodes_adresses


path = (2, 3, 4, 1, 5, 8, 7, 1, 9, 6, 2, 0, 5, 1, 2, 1)

def get_pos_from_coordinates(coordinates):
    pos = {}
    min_x = min(coordinates, key=lambda pair: pair[0])[0]
    min_y = min(coordinates, key=lambda pair: pair[1])[1]
    max_x = max(coordinates, key=lambda pair: pair[0])[0]
    max_y = max(coordinates, key=lambda pair: pair[1])[1]

    offset_x, offset_y = max_x - min_x, max_y - min_y

    for i, (x, y) in enumerate(coordinates):
        pos[i] = ((x - min_x) / offset_x, (y - min_y) / offset_y)


    return pos


def print_path_graph(capacity, school_nodes, nodes, nodes_adresses, path):
    print(capacity, school_nodes, nodes, nodes_adresses, sep="\n")

    number_nodes = len(nodes)

    # create edges list considering a path
    sub_paths = []
    sub_path = []
    for i in range(len(path) - 1):
        sub_path.append((path[i], path[i + 1]))
        if path[i + 1] in school_nodes:
            sub_paths.append(sub_path)
            sub_path = []

    print("Sup-paths:", sub_paths)

    G = nx.MultiDiGraph()
    pos = get_pos_from_coordinates(nodes_adresses)
    print(pos)
    plt.figure(figsize=(5, 5))

    for i, sub_path in enumerate(sub_paths):
        keys = G.add_edges_from(sub_path, path=i)
        collection = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(x, y, i) for x, y in sub_path],
            connectionstyle="arc3,rad=0.1",
            edge_color=[i]*len(sub_path),
            edge_cmap=plt.cm.Set1,
            edge_vmin=float(0),
            edge_vmax=float(len(sub_paths)),
            alpha=0.8,
            label=str(i))
        for patch in collection:
            patch.set_linestyle('dashed')
    colors = ['red' if n not in school_nodes else 'orange' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=colors, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)

        # nx.draw_networkx(graph, pos=nx.spring_layout(), node_color=colors)
    # plt.show()
    print("Graph:", G.graph, G.edges, G.nodes, sep='\n')
    plt.show()
    return sub_paths


def read_input(filename):
    '''Read list of nodes from input file and returns the list of nodes on the proposed format
        Considering the format:
        n_nodes
        node1_coord_x node1_coord_y "school_flag" "school_id"
        node2_coord_x node2_coord_y  "associated_school_id"
        (...)
        **Al schools are given first**'''

    # Read lines from file
    doc = open(filename, "r").readlines()

    # Save the nodes' positions and schools id's
    nodes = []
    schools = {}
    node_id = 0
    read_nodes = 0
    read_schools = 0

    for line in doc:
        read_parameter = line.split(' ')[0]
        if read_parameter == 'capacity':
            capacity = int(line.split(' ')[1])
        elif read_parameter == 'iterations':
            max_iterations = int(line.split(' ')[1])
        elif read_parameter == 'nodes':
            read_nodes = int(line.split(' ')[1])
        elif read_nodes > 0:
            line = line.split(' ')
            nodes.append([node_id, (float(line[0]), float(line[1]))])
            node_id += 1
            read_nodes -= 1
        elif read_parameter == 'schools':
            read_schools = int(line.split(' ')[1])
        elif read_schools > 0:
            line = line.split(' ')
            school_id = int(line[0])
            schools[school_id] = []
            for i in line[1:]:
                schools[school_id].append(int(i))
            read_schools -= 1
    return capacity, max_iterations, nodes, schools



def main(arg: list = []) -> None:
    print_path_graph(*read_adresses_input("generated_map.txt"), path)
    if len(arg) < 2:
        print("The correct way to run is: python run_problem.py <file>")
        continue_program = str(input("Do you want to continue using the file sample.txt? [y/n]"))
        if continue_program == "n":
            sys.exit()
        else:
            filename = 'sample.txt'
    else:
        filename = arg[1]

    # read list of nodes from file
    capacity, max_iterations, nodes_list, schools = read_input(filename)

    number_nodes = len(nodes_list)

    # create edges list considering a complete graph
    edges_list = []
    # print("nodes_list", nodes_list)
    for i, pos in nodes_list:
        for j, pos in nodes_list:
            if i == j:
                continue
            # get distance between u and v using the Google API (for now it is random)
            dist = random.randint(1, 100)  # distance from u to v might be different from v to u (there might be one way streets, for example)
            edges_list.append([[i, j], dist])

    graph = nx.DiGraph()
    # print("schools_ids", schools.keys())
    colors = ['blue' if n not in schools.keys() else 'orange' for n in range(number_nodes)]

    # print("schools")
    for i, pos in nodes_list:
        graph.add_node(i, pos = pos, node_id = i)
        # if i not in schools_ids:
        #     if school_id in schools.keys():
        #         schools[school_id].append(i)
        #     else:
        #         schools[school_id] = [i]
    # print("graph.nodes", graph.nodes)
    print("schools", schools)

    for e, w in edges_list:
        graph.add_edge(e[0], e[1], weight=w)

    pos_list = nx.get_node_attributes(graph, 'pos')

    nx.draw(graph, pos_list, node_color=colors)
    # plt.show()

    agent = Agent(nodes_list, schools, graph, capacity)
    agent.run(max_iterations)
    #agent.get_solution()


if __name__ == "__main__":
    import sys
    main(sys.argv)
