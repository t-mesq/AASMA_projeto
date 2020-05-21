import random as rd
import copy
import numpy as np


# class cSchool():

# 	def __init__(self, school_id, id_graph, pos, children):
# 		self.id = school_id
# 		self.id_graph = id_graph # corresponding id on graph -> to which node does it correspond
# 		self.pos = pos
# 		self.children = children
# 		self.child_left = children
from collections import defaultdict
from enum import Enum


class cChildren():

	def __init__(self, child_id, home_pos, school_id):
		self.id = child_id
		self.home = home_pos
		self.school = school_id


class School:
	def __init__(self, id, students):
		self.id = id
		self.students = students


class cTrip():

	def __init__(self, agent, schools, capacity):
		self.agent = agent
		self.schools = schools
		self.capacity = capacity
		self.first_school = -1

	def choose_next_place(self, current_cargo, current_place):

		possible_successors = []
		for node in self.agent.graph.successors(current_place['node_id']):
			if node in current_cargo:
				possible_successors.append(node)
			elif self.capacity - len(current_cargo) > 0 and (node in self.schools.keys() and self.schools[node] != []):
				possible_successors.append(node)

		print("possible_successors", possible_successors)
		# choosing randomly for now... 
		return rd.choice(possible_successors)

	def randomly_choose_children(self, school_id, number_places):

		if len(self.schools[school_id]) < number_places:
			return self.schools[school_id]
		else:
			return rd.sample(self.schools[school_id], number_places)

	def leave_children_at_home(self, current_cargo, place):

		return [child for child in current_cargo if self.agent.children[child].id != place]

	def update_children_left_at_school(self, children_at_school, current_cargo):
			
		return [child for child in children_at_school if child not in current_cargo]


	def solve(self):

		# choose a school
		random_school_id = rd.randint(1,len(self.schools))
		self.first_school = random_school_id
		print("first_school", random_school_id)
		# choose children from that school
		current_cargo = self.randomly_choose_children(random_school_id, self.capacity)
		print("current_cargo", current_cargo)
		# update children left at that school
		self.schools[random_school_id] = self.update_children_left_at_school(self.schools[random_school_id], current_cargo)
		print("current_school_children", self.schools[random_school_id])
		# set current place
		current_place = self.agent.graph.nodes()[random_school_id]
		print("current_place", current_place)
		# condition to stop -> school has not kids
		school_has_kids = not all([not self.schools[key] for key in self.schools.keys()])
		print("schools", self.schools)

		path = [current_place]

		travel_time = 0

		while school_has_kids:

			next_place_id = self.choose_next_place(current_cargo, current_place)

			if next_place_id in self.schools.keys():
				print("next_place_school", self.schools[next_place_id], current_cargo)
				# get kids from that school
				current_cargo += self.randomly_choose_children(next_place_id, self.capacity-len(current_cargo))
				# update children left at school
				self.schools[next_place_id] = self.update_children_left_at_school(self.schools[next_place_id], current_cargo)
			else:
				print("current_cargo", current_cargo)
				current_cargo = self.leave_children_at_home(current_cargo, next_place_id)
				print("current_cargo after leaving children at home", current_cargo)

			school_has_kids = not all([not self.schools[key] for key in self.schools.keys()])

			travel_time += self.agent.graph.edges[current_place['node_id'], next_place_id]['weight']
			current_place = self.agent.graph.nodes[next_place_id]
			path.append(current_place)

		return path, travel_time



class cAgent():

	def __init__(self, children, schools, graph, capacity):

		self.children = {}
		for child_id, pos, school_id in children:
			self.children[child_id] = cChildren(child_id, pos, school_id)
		self.schools = schools
		self.graph = graph
		self.capacity = capacity


	# eventually add here the q-learning function
	def solve(self, max_iterations):

		it = 0
		while it < max_iterations:

			print("ITER", it)

			# what can we optimize?
			# how children are chosen to take each bus -> how many at each time. Suppose the number of children % capacity != 0. How to divide them?
			# which school should we choose first?
			# group home adresses based on bus capacity?
			# what should the agent learn?

			# each state is a solution OR each state is an unvisited node (see references)

			trip = cTrip(self, copy.deepcopy(self.schools), copy.deepcopy(self.capacity))

			# add some information from the previous trip(s) to guide the next trip
			path, travel_time = trip.solve()

			print("path", path, "travel_time", travel_time)

			# update q function etc and restart
			it += 1


""""
** ** ** ** ** ** ** ** ** ** ** **
** ** A: Q - learning 	** ** ** **
** ** ** ** ** ** ** ** ** ** ** ** 
"""

ActionSelection = Enum("ActionSelection", "eGreedy softMax")
LearningApproach = Enum("LearningApproach", "QLearning SARSA")

actionSelection = ActionSelection.eGreedy;
learningApproach = LearningApproach.QLearning;

it = 0
total = 100000
discount = 0.9
learningRate = 0.8
epsilon = 0.7
randfactor = 0.05

q = defaultdict(lambda: np.zeros(10))	#actions
actions = {}

def getState(pos, bus_content, schools_state):
	return 0 #n_pos * schools_states * bus_state

#Creates the initial Q - value function structure: (x y action) < - 0
def initQfunction():
	return



