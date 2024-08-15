from enum import Enum
from typing import Union, List, Tuple

from problem_env.games.base_game.state import State


# @Israel - priority 1
# TODO: Add interface for environment

# TODO Add a game environment
class Game:
    def __init__(self, initial_state=None, goal_state=None, *args, **kwargs):
        self.initial_state = initial_state.copy() if initial_state else None
        self._current_state = self.initial_state.copy() or self.reset()
        self.goal_state = goal_state or self.generate_random_states(1)[0]

    class Actions(Enum):
        UP = 1
        DOWN = 2
        LEFT = 3
        RIGHT = 4

        def __str__(self):
            return f'{self.name} ({self.value})'

    def reset(self):
        if self.initial_state:
            self._current_state = self.generate_random_states(1)[0]
        else:
            self._current_state = self.initial_state
        return self._current_state

    def generate_random_states(self, number_of_states=10) -> List[State]:
        """
        This method generates random states
        :return: a list of random states
        """
        random_states = [State.generate_random_state().get_state_as_list() for _ in range(number_of_states)]
        return random_states

    def get_neighbors(self, state: State=None) -> List[State]:
        """
        :param state:
        :return: list containing neighboring states
        """
        if state is None:
            state = self._current_state
        return state.get_neighbors()

    def value_function(self, state=None) -> float:
        """
        This method calculates the value of a state
        :param state: the state to calculate the value for
        :return: the value of the state
        """
        if state is None:
            state = self._current_state
        pass
        raise NotImplementedError

    def heuristic_function(self, state=None):
        """
        This method calculates the heuristic value of a state
        :param state: the state to calculate the heuristic value for
        :return: the heuristic value of the state
        """
        if state is None:
            state = self._current_state
        return self.value_function(state)

    def get_step_action(self, state, successor):
        """
        This method calculates which action is needed to be taken to move from state to successor
        :param state: original state
        :param successor: new state to move to
        :return: the action that is needed to be taken to move from state to successor
        """
        raise NotImplementedError

    def cost_function(self, state, action: Actions) -> float:
        """
        This method calculates the cost of taking an action from a state
        :param state: the state to take the action from
        :param action: the action to take
        :return: the cost of taking the action from the state
        """
        raise NotImplementedError

    def move_from_state_to_state_cost_function(self, state, successor) -> float:
        """
        This method calculates the cost of moving from state to successor
        :param state:
        :param successor:
        :return:
        """
        action = self.get_step_action(state, successor)
        action_cost = self.cost_function(state, action)
        return action_cost

    def edge_function(self, state, successor) -> float:
        """
        An alias for move_from_state_to_state_cost_function
        """
        return self.move_from_state_to_state_cost_function(state, successor)

    def get_state_as_list(self, state) -> list:
        """
        This method returns the state as a list (vector)
        :return: the state as a list
        """
        pass

    def is_goal_state(self, state=None) -> bool:
        """
        This method checks if the state is a goal state
        :param state: the state to check
        :return: True if the state is a goal state, False otherwise
        """
        if state is None:
            state = self._current_state
        return state == self.goal_state

    def is_goal(self, state):
        """
        An alias for is_goal_state
        """
        return self.is_goal_state(state)

    def display_state(self, state=None):
        """
        This method displays a given state
        :param state: the state to display
        :return: None
        """
        if state is None:
            state = self._current_state
        state.display()

    def display_current_state(self):
        """
        This method displays the current state
        """
        self.display_state(self._current_state)

    def get_goal_state(self):
        return self.goal_state

    def get_state_for_forward_pass(self, start):
        pass

    def get_input_channels_num(self):
        pass

    def get_h_values(self, param):
        pass
