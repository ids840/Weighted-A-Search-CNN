from enum import Enum
from typing import Union, List, Tuple


class State:
    def __init__(self, state=None, *args, **kwargs):
        if state is None:
            self.state = State.generate_random_state()
        if isinstance(state, State):
            self.state = state.get_state_as_list()
        else:
            self.state = state

    def get_state_as_list(self):
        return self.state

    def get_neighbors(self) -> List['State']:
        """
        This method returns the neighboring states of the current state
        :return: a list of neighboring states
        """
        pass

    @staticmethod
    def generate_random_state() -> 'State':
        """
        This method generates a random state
        :return: a random state
        """
        pass

    def display(self):
        pass

    def copy(self):
        return State(self.state)

    def __copy__(self):
        return State(self.state)

    def __lt__(self, other):
        return self.state < other.state

    def __hash__(self):
        return hash(tuple(self.state))

    def __eq__(self, other):
        return self.state == other.state and self.k == other.k

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.__class__.__name__}()'
