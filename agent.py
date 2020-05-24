import random as rd
import copy
import numpy as np
from collections import defaultdict
from functools import partial
from itertools import repeat
import math
import threading


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
from collections import defaultdict
from enum import Enum

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

    def get_reward(self, current_state, action, travel_time):
        # if action == 'pick' or action == 'drop':
        #     return 100
        # elif weight > 0:
        #     return -weight
        # elif self.compute_state_key(current_state) == self.initial_state_key:
        #     return -10000
        if self.compute_state_key(current_state) == self.final_state_key:
            return 1/travel_time
        else:
            return 0

    def get_possible_actions(self, current_state):

        possible_actions = []
        position = current_state['pos']
        capacity = self.agent.capacity - sum(current_state['bus'].values())
        # at a home address and there are kids to drop at that address
        if position not in current_state['schools'].keys() and position in current_state['bus'].keys() and current_state['bus'][position] > 0:
            possible_actions.append('drop')
        # at a school with capacity to pick kids and there are kids at that school
        else:
            possible_actions.append('travel')
            if position in current_state['schools'].keys() and capacity > 0 and sum(current_state['schools'][position]) > 0:
                possible_actions.append('pick')

        # print("get_possible_actions", possible_actions)
        return self.compute_possible_action_results(current_state, possible_actions), possible_actions

    def choose_and_execute_next_action(self, current_state, possible_actions_results, greedy=False):

        if not greedy:
            prob = rd.uniform(0, 1)
            greedy = prob > self.agent.epsilon

        if greedy:
            return self.get_max_action(current_state, possible_actions_results)
        else:
            return rd.choice(possible_actions_results)
            
    
    def compute_possible_action_results(self, current_state, actions):
        '''compute the possible next states given the current state and a set of possible actions'''

        possible_actions_results = []
        for action in actions:
            if action == 'drop':
                next_state = copy.deepcopy(current_state)
                next_state['bus'][current_state['pos']] = 0
                next_state['action'] = 'drop'
                possible_actions_results.append(next_state)
            if action == 'pick':
                possible_actions_results += self.pick_children_from_school(current_state)
            if action == 'travel':
                possible_actions_results += self.possible_states_to_travel_to(current_state)

        # if not possible_actions_results:
        #     print(actions)
        # print("compute_possible_action_results", possible_actions_results)
        return possible_actions_results

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
            next_state['action'] = 'pick'
            return_states.append(next_state)

        # print("pick_children_from_school", return_states)
        return return_states

    def possible_states_to_travel_to(self, current_state):

        possible_successors = []
        # every node is a possible successor
        for node_id in range(len(self.agent.graph)):
            if node_id == current_state['pos']:
                continue
            elif node_id not in current_state['schools'].keys() and node_id in current_state['bus'].keys() and current_state['bus'][node_id] > 0:
                possible_successors.append(node_id)
            #elif self.agent.capacity - sum(current_state['bus'].values()) > 0 and (node_id in self.schools.keys() and sum(current_state['schools'][node_id]) > 0):
            elif node_id in self.schools.keys() and (self.agent.capacity - sum(current_state['bus'].values()) > 0) and sum(current_state['schools'][node_id]) > 0:
                possible_successors.append(node_id)
        #in the end the agent may return to the final schooç
        if sum(current_state['bus'].values()) == 0 and not any(sum(current_state['schools'][school_id]) > 0 for school_id in self.schools.keys()):
            possible_successors.append(self.final_state_key[0])
        return_states = []
        for suc in possible_successors:
            next_state = copy.deepcopy(current_state)
            next_state['pos'] = suc
            next_state['action'] = 'travel'
            return_states.append(next_state)

        # print("possible_states_to_travel_to", return_states)
        return return_states

    def get_max_action(self, current_state, possible_actions):
        ''' get action that maximizes the q value, given the current state '''
        max_value = -math.inf
        maximizer = possible_actions[0]
        q_values = [[self.agent.get_q_value((self.compute_state_key(current_state), self.compute_state_key(action))), action] for action in possible_actions]
        if all(elem[0] == q_values[0][0] for elem in q_values):
            maximizer = rd.choice(possible_actions)
        else:
            maximizer = max(q_values, key=lambda elem: elem[0])[1]
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
        # print("key", key)
        state = {'pos': key[0]}
        state['schools'] = {}
        for element in key[1:len(key) - 1]:
            state['schools'][element[0]] = list(element[1])
        state['bus'] = {}
        for element in key[len(key) - 1]:
            state['bus'][element[0]] = element[1]
        state['action'] = ''
        return state

    def recover_greedy_path(self):

        current_state_key = self.initial_state_key
        final_state_key = self.final_state_key
        current_state = self.get_state_from_key(current_state_key)
        sequence = [current_state['pos']]
        travel_time = 0
        iterations = 0
        while current_state_key != final_state_key and iterations < 200:
            # percept (current_state)
            # decide
            next_possible_actions, actions = self.get_possible_actions(current_state)
            # execute
            next_state = self.choose_and_execute_next_action(current_state, next_possible_actions, greedy=True)
            # update
            # if current_state['pos'] != next_state['pos']:
            #     sequence.append(next_state['pos'])

            sequence.append(next_state["pos"])

            if next_state['action'] == 'travel':
                if current_state['pos'] == next_state['pos']:
                    weight = 0
                else:
                    weight = self.agent.graph[current_state['pos']][next_state['pos']]
            else:
                weight = 0
            travel_time += weight
            # reward = self.get_reward(next_state, next_state['action'], travel_time)


            # key = (self.compute_state_key(current_state), self.compute_state_key(self.get_max_action(current_state, next_possible_actions)))
            # prediction_error = reward + self.agent.discount * self.agent.get_q_value(key) - previous_q_value

            # self.agent.update_q_value(key, previous_q_value + self.agent.learning_rate * prediction_error)

            # previous_q_value = self.agent.get_q_value(key)

            current_state = next_state

            current_state_key = self.compute_state_key(current_state)
            iterations += 1
            # print(current_state)

        return sequence, travel_time


    def run(self, max_iterations):

        first_scool_pos = rd.choice(list(self.schools.keys()))
        self.initial_school_id = first_scool_pos

        # initialize current state and final state
        current_state = {'pos': first_scool_pos}  # first node id
        final_state = {'pos': first_scool_pos}
        current_state['schools'] = self.schools
        final_state['schools'] = {}
        for school_id in self.schools.keys():
            final_state['schools'][school_id] = [0] * len(self.agent.graph)  # no child left at school for each address
        # empty bus
        current_state['bus'] = {}
        final_state['bus'] = {}
        for i in range(len(self.agent.graph)):
            current_state['bus'][i] = 0
            final_state['bus'][i] = 0
        current_state['action'] = ''
        final_state['action'] = ''

        # print(final_state)
        # get keys
        current_state_key = self.compute_state_key(current_state)
        final_state_key = self.compute_state_key(final_state)

        self.initial_state_key = current_state_key
        self.final_state_key = final_state_key
        previous_q_value = 0
        
        it = 0

        travel_times = []
        sequence = [current_state]
        sequence_nodes = [current_state['pos']]
        travel_time = 0
        # print(current_state)
        # print(final_state)
        count_restart = 0
        last_restart = 0
        while it < max_iterations:

            # percept (current_state)
            # decide
            next_possible_actions, actions = self.get_possible_actions(current_state)
            # execute
            next_state = self.choose_and_execute_next_action(current_state, next_possible_actions)
            # update
            sequence.append(next_state)
            sequence_nodes.append(next_state['pos'])

            if next_state['action'] == 'travel':
                if current_state['pos'] == next_state['pos']:
                    weight = 0
                else:
                    weight = self.agent.graph[current_state['pos']][next_state['pos']]
            else:
                weight = 0
            travel_time += weight
            reward = self.get_reward(next_state, next_state['action'], travel_time)


            key = (self.compute_state_key(current_state), self.compute_state_key(self.get_max_action(current_state, next_possible_actions)))
            prediction_error = reward + self.agent.discount * self.agent.get_q_value(key) - previous_q_value

            self.agent.update_q_value(key, previous_q_value + self.agent.learning_rate * prediction_error)

            previous_q_value = self.agent.get_q_value(key)

            current_state = next_state

            current_state_key = self.compute_state_key(current_state)

            self.agent.update_epsilon()


            if current_state_key == final_state_key:
                current_state = self.get_state_from_key(self.initial_state_key)
                current_state['pos'] = rd.choice(list(self.schools.keys()))
                final_state['pos'] = current_state['pos']
                final_state_key = self.compute_state_key(final_state)
                self.initial_state_key = self.compute_state_key(current_state)
                self.initial_school_id = final_state['pos']
                current_state['action'] = 'restart'
                # print(travel_time, sequence, sep='\n')
                travel_times.append(travel_time)
                travel_time = 0
                # print("final_state",final_state, final_state_key, "current_state",current_state, self.initial_state_key, sep='\n')
                # print("ITER", it)
                count_restart += 1
                last_restart = it

            if it % 100000 == 0:
                print(str(it) + " " + str(self.agent.epsilon), self.recover_greedy_path(), sep='\n')
                

            it += 1

        # print(sequence, sep='\n')
        print(count_restart)
        return sequence, travel_times


class Agent():

    def __init__(self, schools, graph, capacity, epsilon=0.8, learning_rate=0.8, discount=0.9, max_iterations=10000, rand_factor=0.01, q_values=None):
        self.schools = schools
        self.addresses = {}
        self.graph = graph
        self.capacity = capacity
        self.actions = {'pick', 'travel', 'drop'}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.max_iterations = max_iterations
        self.rand_factor = rand_factor
        self.dec = (epsilon-rand_factor)/max_iterations
        if q_values is None:
            self.q_values = defaultdict(int)
        else:
            self.q_values = q_values

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.dec, self.rand_factor);

    def get_q_value(self, key):
        return self.q_values[key]

    def update_q_value(self, key, value):
        self.q_values[key] = value

    # eventually add here the q-learning function
    def run(self):
        trip = Trip(self, self.schools)
        trip.run(self.max_iterations)


class ThreadingAgent(Agent, threading.Thread):

    def __init__(self, lock, q_values, schools, graph, capacity, epsilon=0.8, learning_rate=0.8, discount=0.9, max_iterations=10000, rand_factor=0.01):
        threading.Thread.__init__(self)
        Agent.__init__(self, schools, graph, capacity, epsilon=epsilon, learning_rate=learning_rate, discount=discount, max_iterations=max_iterations, rand_factor=rand_factor, q_values=q_values)
        self.lock = lock

    def update_q_value(self, key, value):
        self.lock[key].acquire()
        self.q_values[key] = value
        self.lock[key].release()

    def get_q_value(self, key):
        self.lock[key].acquire()
        try:
            return self.q_values[key]
        finally:
            self.lock[key].release()