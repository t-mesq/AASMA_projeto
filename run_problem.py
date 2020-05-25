import random  # to generate random distances while there is no connection to the API
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import enum
import pylab as pl
from agent import *
import numpy as np


def choose_grid(nr):
    return nr // 4 + 1, 4

def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox=True, shadow=True, borderaxespad=0)

def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, step: int, percentage=False, reverse=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_labels = list(map(str, xvalues))
    ax.bar(xvalues, yvalues, width=0.6*step)
    ax.set_xticks(xvalues)
    ax.set_xticklabels(x_labels, rotation=90, fontsize='small')



def read_addresses_input(filename):
    nodes = []
    nodes_addresses = []
    read_nodes = 0
    schools = {}

    doc = open(filename, "r").readlines()

    for line in doc:
        read_parameter, value = line.split(' ', 1)
        if read_parameter == 'capacity':
            capacity = int(value)
        elif read_parameter == 'iterations':
            iterations = int(value)
        elif read_parameter == 'schools':
            school_nodes = list(map(int, value.split(' ')))
            read_schools = len(school_nodes)
        elif read_schools > 0:
            schools[school_nodes[len(school_nodes)-read_schools]] = list(map(int, line.split(' ')))
            read_schools -= 1
        elif read_parameter == 'nodes':
            read_nodes = int(value)
        elif read_nodes > 0:
            nodes.append(list(map(int, line.split())))
            read_nodes -= 1
        else:
            nodes_addresses.append(eval(line))

    return capacity, iterations, school_nodes, schools, nodes, nodes_addresses

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


def print_path_graph(school_nodes, nodes, nodes_addresses, path=[], verbose=False):

    print( school_nodes, nodes, nodes_addresses, sep="\n")

    number_nodes = len(nodes)

    # create edges list considering a path
    sub_paths = []
    sub_path = []
    for i in range(len(path) - 1):
        sub_path.append((path[i], path[i + 1]))
        if path[i + 1] in school_nodes:
            sub_paths.append(sub_path)
            sub_path = []


    G = nx.MultiDiGraph()
    pos = get_pos_from_coordinates(nodes_addresses)

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

    if verbose:
        print("Sup-paths:", sub_paths)
        print("Positions:", pos)
        print("Graph:", G.graph, G.edges, G.nodes, sep='\n')
    # plt.show()
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

    if len(arg) < 2:
        print("The correct way to run is: python run_problem.py <file>")
        continue_program = str(input("Do you want to continue using the file generated_map.txt? [y/n]"))
        if continue_program == "n":
            sys.exit()
        else:
            filename = 'generated_map.txt'
    else:
        filename = arg[1]


    # read list of nodes from file
    capacity, max_iterations, school_ids, schools_list, adj_matrix, addresses = read_addresses_input(filename)

    print("schools", schools_list)

    Mode = enum.Enum("Mode", "Single Threaded")
    mode = Mode.Single
    # mode = Mode.Threaded


    if mode == Mode.Single:
        learning_rate = 0.9
        agent = Agent(schools_list, adj_matrix, capacity, learning_rate = learning_rate, max_iterations=max_iterations)
        sequence, times, restart_counts = agent.run()

        # number of restarts plot
        step = int(max_iterations/len(restart_counts))
        x_axis = [x for x in range(step, max_iterations+step, step)]
        bar_chart(plt.gca(),  x_axis,restart_counts, "Number of restarts by " + str(step) + " iterations", "Iterations", "Number of restarts", step)
        plt.savefig("number_of_restarts" + filename + ".png", bbox_inches='tight', dpi=300)
        plt.clf()

        # last path graph plot
        print_path_graph(school_ids, adj_matrix, addresses, sequence)
        plt.savefig("graph_"+filename+".png")
        plt.clf()

        # travel times plot
        multiple_line_chart(plt.gca(), list(range(len(times))), {"learning rate " + str(learning_rate): times}, "Greedy Path duration per 10 restarts", "Route Restarts (10)", "Greedy path duration")
        plt.savefig("figure_"+filename+".png")
        plt.clf()

    else:
        lock = defaultdict(lambda: threading.Lock())
        Q = defaultdict(int)

        for i in range(5):
            agent = ThreadingAgent(lock, Q, schools_list, adj_matrix, capacity, max_iterations=max_iterations)
            agent.start()


if __name__ == "__main__":
    import sys
    main(sys.argv)
