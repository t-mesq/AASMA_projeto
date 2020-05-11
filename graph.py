class cNode():

	def __init__(self, id, coord_x = 0, coord_y = 0, is_school = False):
		self.id = id
		self.coord_x = coord_x
		self.coord_y = coord_y
		self.is_school = is_school

class cEdge():

	def __init__(self, tail, head, weight = 0, id = 0):
		self.id = id
		self.tail = tail
		self.head = head
		self.weight = weight

class cGraph():

	def __init__(self, nodes_list = [], edges_list = []):
		self.n_nodes = len(nodes_list)
		self.n_edges = len(edges_list)
		self.edges = edges_list
		self.nodes = nodes_list



		
		


