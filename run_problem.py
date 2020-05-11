from graph import *
import random # to generate random distances while there is no connection to the API
import networkx as nx
import matplotlib.pyplot as plt



def read_input(filename):
    '''Read list of nodes from input file and returns the list of nodes on the proposed format 
        Considering the format:
        n_nodes
        node1_coord_x node1_coord_y 
        node2_coord_x node2_coord_y "school"
        (...) '''

    
    # Read lines from file
    doc = open(filename, "r").readlines()
    
    # Save the nodes' list
    nodes_list = []  
    node_id = 0
    # The first line was already saved (n_nodes, n_edges)
    for line in doc:

        line = line.split(" ")
        is_school = len(line) > 2 # if there are more than 2 elements at a certain line, then it is a school
        nodes_list.append(cNode(node_id, float(line[0]), float(line[1]), is_school))
        node_id += 1

    return nodes_list




def main(arg: list = []) -> None:
   
    if len(arg) < 2:
        print("The correct way to run is: python run_problem.py <file>")
        return 
    else:
        filename = arg[1]
        
    # read list of nodes from file
    nodes_list = read_input(filename)

    # create edges list considering a complete graph
    edges_list = []
    alt_edges_list = []
    for u in nodes_list:
        for v in nodes_list:
            if u == v:
                continue
            # get distance between u and v using the Google API (for now it is random)
            dist = random.randint(1,100) # distance from u to v might be different from v to u (there might be one way streets, for example)
            edges_list.append(cEdge(u, v, dist))
            alt_edges_list.append([u.id, v.id])


    graph = cGraph(nodes_list, edges_list)

    [print(node.id, node.is_school) for node in graph.nodes] 
    [print(edge.tail.id, edge.head.id, edge.weight) for edge in graph.edges]


    # while we're not using Goolge's API
    color_choice = lambda x : 'orange' if x.is_school else 'blue' 
    alt_graph = nx.DiGraph()
    colors = []
    for node in nodes_list:
        alt_graph.add_node(node.id, pos=(node.coord_x, node.coord_y), color =  color_choice(node))

    alt_graph.add_edges_from(alt_edges_list)

    pos=nx.get_node_attributes(alt_graph,'pos')
    colors = nx.get_node_attributes(alt_graph, 'color')


    nx.draw(alt_graph, pos, node_color = colors.values())
    plt.show()


if __name__ == "__main__":
    import sys
    main(sys.argv)