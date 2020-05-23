# from graph import *
import random  # to generate random distances while there is no connection to the API
import networkx as nx
import matplotlib.pyplot as plt
from agent import *


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