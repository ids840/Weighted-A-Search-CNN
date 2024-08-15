from problem_env.games.base_game.state import State


class Heuristic:
    def get_h_values(self, states):
        if not states:
            return []
        if isinstance(states[0], State):
            states_as_list = [state.get_state_as_list() for state in states]
        else:
            states_as_list = states
        gaps = []

        for state_as_list in states_as_list:
            gap = 0
            if state_as_list[0] != 1:
                gap = 1

            for i in range(len(state_as_list) - 1):
                if abs(state_as_list[i] - state_as_list[i + 1]) != 1:
                    gap += 1

            gaps.append(gap)
        return gaps
