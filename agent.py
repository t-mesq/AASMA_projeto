import random as rd
import copy
from collections import defaultdict
from functools import partial
from itertools import repeat
import math


def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()


# class cSchool():

# 	def __init__(self, school_id, id_graph, pos, children):
# 		self.id = school_id
# 		self.id_graph = id_graph # corresponding id on graph -> to which node does it correspond
# 		self.pos = pos
# 		self.children = children
# 		self.child_left = children

class Address():

    def __init__(self, address_id, position, is_school):
        self.id = address_id
        self.position = position
        self.is_school = is_school


class Trip():
    ''' An episode of the q-learning algorithm '''

    def __init__(self, agent, schools):
        self.agent = agent
        self.schools = schools
        self.bus_capacity = copy.deepcopy(agent.capacity)
        self.initial_state_key = ()
        self.final_state_key = ()
        self.initial_school_id = -1

    def get_reward(self, current_state, action, weight=0):
        if action == 'pick' or action == 'drop':
            return 100
        elif weight > 0:
            return -weight
        elif self.compute_state_key(current_state) == self.initial_state_key:
            return -10000
        elif self.compute_state_key(current_state) == self.final_state_key:
            return 10000000

    def get_possible_actions(self, current_state):

        position = current_state['pos']
        capacity = self.agent.capacity - sum(current_state['bus'].values())
        # at a school with capacity to pick kids and there are kids at that school
        if position in current_state['schools'].keys() and capacity > 0 and sum(current_state['schools'][position]) > 0:
            action = 'pick'
        # at a home address and there are kids to drop at that address
        elif position in current_state['bus'].keys() and current_state['bus'][position] > 0:
            action = 'drop'
        else:
            action = 'travel'

        return self.compute_possible_action_results(current_state, action), action

    def choose_and_execute_next_action(self, current_state, possible_actions_results):

        prob = rd.uniform(0, 1)
        if prob > self.agent.epsilon:
            return rd.choice(possible_actions_results)
        else:
            return self.get_max_action(current_state, possible_actions_results)

    def compute_possible_action_results(self, current_state, action):

        if action == 'drop':
            next_state = copy.deepcopy(current_state)
            next_state['bus'][current_state['pos']] = 0
            return [next_state]
        elif action == 'pick':
            return self.pick_children_from_school(current_state)
        elif action == 'travel':
            return self.possible_states_to_travel_to(current_state)

    def pick_children_from_school(self, current_state):

        possible_choices = []  # initialize possible choices
        school_id = current_state['pos']
        current_school = current_state['schools'][school_id]  # get info on current school
        current_addresses = [key for key in current_state['bus'].keys() if current_state['bus'][key] > 0]  # get info on current cargo to choose children that share the same address
        if current_addresses:
            possible_choices = [address for address in current_addresses if current_school[address] > 0]
        # it there are no current addresses on the bus compatible with children at school
        if not possible_choices:
            possible_choices = [i for i in range(len(current_school)) if current_school[i] > 0]

        return_states = []
        for address in possible_choices:
            next_state = copy.deepcopy(current_state)
            # update bus cargo
            if address in next_state['bus'].keys():
                next_state['bus'][address] += 1
            else:
                next_state['bus'][address] = 1
            # update school's state
            next_state['schools'][school_id][address] -= 1
            return_states.append(next_state)

        # print("pick_children_from_school", return_states)
        return return_states

    def possible_states_to_travel_to(self, current_state):

        possible_successors = []
        for node_id in self.agent.graph.successors(current_state['pos']):
            if node_id in current_state['bus'].keys() and current_state['bus'][node_id] > 0:
                possible_successors.append(node_id)
            elif self.agent.capacity - sum(current_state['bus'].values()) > 0 and (node_id in self.schools.keys() and sum(current_state['schools'][node_id]) > 0):
                possible_successors.append(node_id)
            elif sum(current_state['bus'].values()) == 0 and all([sum(current_state['schools'][key]) == 0 for key in current_state['schools'].keys()]):
                possible_successors.append(self.initial_school_id)
                break

        return_states = []
        for suc in possible_successors:
            next_state = copy.deepcopy(current_state)
            next_state['pos'] = suc
            return_states.append(next_state)

        # print("possible_states_to_travel_to", return_states)
        return return_states

    def get_max_action(self, current_state, possible_actions):
        ''' get action that maximizes the q value, given the current state '''
        max_value = -math.inf
        maximizer = possible_actions[0]
        for action in possible_actions:
            key = (self.compute_state_key(current_state), self.compute_state_key(action))
            q_value = self.agent.get_q_value(key)
            if q_value > max_value:
                max_value = q_value
                maximizer = action
        return maximizer

    def compute_state_key(self, state):
        '''compute key given the state '''
        key = [state['pos']]
        for s_id in state['schools']:
            school_list = [s_id]
            school_list.append(tuple(state['schools'][s_id]))
            key.append(tuple(school_list))
        cargo_list = []
        for cargo_id in state['bus']:
            cargo_list.append((cargo_id, state['bus'][cargo_id]))
        key.append(tuple(cargo_list))
        return tuple(key)

    def get_state_from_key(self, key):
        ''' return state given its key'''
        state = {'pos': key[0]}
        state['schools'] = {}
        for element in key[1:len(key) - 1]:
            state['schools'][element[0]] = list(element[1])
        state['bus'] = {}
        for element in key[len(key) - 1]:
            state['bus'][element[0]] = element[1]
        return state

    def run(self, first_school_id):

        self.initial_school_id = first_school_id
        sequence = [first_school_id]
        travel_time = 0

        # initialize current state and final state
        current_state = {'pos': first_school_id}  # first node id
        final_state = {'pos': first_school_id}
        current_state['schools'] = self.schools
        final_state['schools'] = {}
        for school_id in self.schools.keys():
            final_state['schools'][school_id] = [0] * len(self.agent.addresses)  # no child left at school for each address
        # empty bus
        current_state['bus'] = {}
        final_state['bus'] = {}
        for address in self.agent.addresses:
            current_state['bus'][address] = 0
            final_state['bus'][address] = 0

        # print(final_state)
        # get keys
        current_state_key = self.compute_state_key(current_state)
        final_state_key = self.compute_state_key(final_state)

        self.initial_state_key = current_state_key
        self.final_state_key = final_state_key
        previous_q_value = 0
        count = 0
        while current_state_key != final_state_key and count < 1000:

            # percept (current_state)
            # decide
            next_possible_actions, action = self.get_possible_actions(current_state)
            # execute
            next_state = self.choose_and_execute_next_action(current_state, next_possible_actions)
            # update
            if current_state['pos'] != next_state['pos']:
                sequence.append(next_state['pos'])

            if action == 'travel':
                if current_state['pos'] == next_state['pos']:
                    weight = 0
                else:
                    weight = self.agent.graph.edges[current_state['pos'], next_state['pos']]['weight']
            else:
                weight = 0

            reward = self.get_reward(next_state, action, weight)
            travel_time += weight

            key = (self.compute_state_key(current_state), self.compute_state_key(self.get_max_action(current_state, next_possible_actions)))
            prediction_error = reward + self.agent.discount * self.agent.get_q_value(key) - previous_q_value

            self.agent.update_q_value(key, previous_q_value + self.agent.learning_rate * prediction_error)

            previous_q_value = self.agent.get_q_value(key)

            current_state = next_state

            current_state_key = self.compute_state_key(current_state)

            # current_action = choose_next_action(current_state)
            # current_action_key = current_state + [current_action]
            count += 1

        return sequence, travel_time


class Agent():

    def __init__(self, addresses, schools, graph, capacity):

        self.addresses = {}
        for address_id, pos in addresses:
            self.addresses[address_id] = Address(address_id, pos, address_id in schools.keys())
        self.schools = schools
        self.graph = graph
        self.capacity = capacity
        self.actions = {'pick', 'travel', 'drop'}
        # self.states = nested_defaultdict(int, 1+len(schools)+1) # 1 for position + one for each school + 1 for the bus
        # self.q_values = nested_defaultdict(int, 1+len(schools)+1+1) # same as states + action
        self.states = {}
        self.q_values = defaultdict(int)
        self.epsilon = 0.2
        self.learning_rate = 0.8
        self.discount = 0.9

    def get_q_value(self, key):
        return self.q_values[key]

    def update_q_value(self, key, value):
        self.q_values[key] = value

    # eventually add here the q-learning function
    def run(self, max_iterations):

        it = 0
        while it < max_iterations:

            # what can we optimize?
            # how children are chosen to take each bus -> how many at each time. Suppose the number of children % capacity != 0. How to divide them?
            # which school should we choose first?
            # group home adresses based on bus capacity?
            # what should the agent learn?

            # each state is a solution OR each state is an unvisited node (see references)
            trip = Trip(self, copy.deepcopy(self.schools))

            # add some information from the previous trip(s) to guide the next trip
            # choose a school
            random_school_id = rd.choice(list(self.schools.keys()))
            # print("ITER", it, random_school_id)
            sequence, travel_time = trip.run(random_school_id)
            # print(self.q_values.values())

            if it % 100 == 0:
                print(travel_time, sequence)

            # update q function etc and restart
            it += 1



