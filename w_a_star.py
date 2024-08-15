import heapq
import random
from problem_env.games.base_game.game import Game


def get_path(path_list, goal_state):
    output_path = []
    curr = goal_state
    while curr:
        output_path.insert(0, curr)
        curr = path_list[curr]
    return output_path


def get_node_from_open(open_set, threshold):
    for node in list(open_set.keys()):
        if open_set[node] <= threshold:
            return node


def replace_value(min_heap, index):
    parent = (index - 1) // 2
    if index > 0 and min_heap[index[0]] < min_heap[parent[0]]:
        min_heap[index], min_heap[parent] = min_heap[parent], min_heap[index]
        replace_value(min_heap, parent)
def return_goal(min_heap):
    for current in min_heap:
        state, w_state, state_expansions = current[1]
        if state.position == state.goal:
            return current
    return None

def weighted_A_star_const(start, game, max_number_of_expansions, w):
    expansions = 0
    cost_function = game.move_from_state_to_state_cost_function
    heuristic_function = game.heuristic_function
    threshold = heuristic_function(start) * w
    min_heap = [(heuristic_function(start) * w, (start, w, max_number_of_expansions), 0)]
    open_set = dict()
    open_set[start] = (heuristic_function(start) * w, (start, w, max_number_of_expansions), 0)
    close_list = set()
    close_list.add(start)
    path = {(start, w, max_number_of_expansions): None}
    while expansions < max_number_of_expansions:
        if not min_heap:
            break
        if return_goal(min_heap) is not None:
            current=return_goal(min_heap)
            state, w_state, state_expansions = current[1]
            return get_path(path, (state, w_state, state_expansions))
        current = heapq.heappop(min_heap)
        expansions = expansions + 1
        state, w_state, state_expansions = current[1]
        open_set.pop(state)
        g_score = current[2]
        threshold = max(threshold, g_score + w * heuristic_function(state))
        if state.position == state.goal:
            return get_path(path, (state, w_state, state_expansions))
            # return get_path(path, (state, w_state)), expansions
        for successor in game.get_neighbors(state):
            g_successor = g_score + cost_function(state, successor)
            f_successor = g_successor + heuristic_function(successor) * w
            if successor in open_set and open_set[successor][2] > g_successor:
                old = open_set.get(successor)
                successor = old[1][0]
                curr = (f_successor, (successor, w,max_number_of_expansions-expansions), g_successor)
                open_set[successor] = curr
                min_heap = list(min_heap)
                min_heap.remove(old)
                open_set[successor] = curr
                heapq.heapify(min_heap)
                heapq.heappush(min_heap, curr)
                path[(successor, w,max_number_of_expansions-expansions)] = (state, w_state, state_expansions)
            elif successor not in close_list:
                close_list.add(successor)
                curr = (f_successor, (successor, w, max_number_of_expansions-expansions), g_successor)
                open_set[successor] = curr
                heapq.heappush(min_heap, curr)
                path[(successor, w, max_number_of_expansions-expansions)] = (state, w_state, state_expansions)

    return get_path(path, (state, w_state,state_expansions))
