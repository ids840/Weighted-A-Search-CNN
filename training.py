import w_a_star


class LabelGenerator:
    def __init__(self, game):
        self.game = game

    # find the first (i.e min) w that give solution, label him as 1 and the other as 0.
    def bootstrap_training(self, w_values,  W_models,epochs, T_initial=2, max_iterations=5):
        labels = []
        w_inputs = []
        t_inputs = []
        for i in range(len(w_values)):
            w_inputs.append([])
        for i in range(len(w_values)):
            t_inputs.append([])
        for i in range(len(w_values)):
            labels.append([])
        states = self.game.generate_random_states(number_of_states=50)
        for state in states:
            found = False
            T = T_initial
            for iteration in range(max_iterations):
                if not found:
                    for w in w_values:
                        path = w_a_star.weighted_A_star_const(state, self.game, T, w)
                        last_state_is_goal = path[-1][0].position == path[-1][0].goal
                        path_length = len(path)
                        for i in range(path_length):
                            curr_state_tensor = self.game.get_state_for_forward_pass(path[i][0])
                            w_inputs[w_values.index(w)].append(curr_state_tensor)
                            t_inputs[w_values.index(w)].append(path[i][2])
                        if (last_state_is_goal and not found) or (
                                not found and iteration == max_iterations-1 and w == w_values[-1]):
                            found = True
                            for i in range(path_length):
                                labels[w_values.index(w)].append(1)
                        else:
                            for i in range(path_length):
                                labels[w_values.index(w)].append(0)
                T *= 2
        for i in range(len(w_values)):
            W_models[i].train_model(w_inputs[i], t_inputs[i], epochs, labels[i])

    # try each state for each w
    def bootstrap_training_alternative(self, W_models, epochs, T_initial=2, max_iterations=5):

        w_values = [1, 2, 5, 7, 10]
        original_states = self.game.generate_random_states(number_of_states=50)
        for w in w_values:
            inputs = []
            labels = []
            w_inputs = []
            t_inputs = []
            T = T_initial
            states = [item for item in original_states]
            for iteration in range(max_iterations):
                for state in states:
                    path, expansions = w_a_star.weighted_A_star_const(state, self.game, T, w)
                    last_state_is_goal = path[-1][0].position == path[-1][0].goal
                    path_length = len(path)
                    for i in range(path_length):
                        curr_state_tensor = self.game.get_state_for_forward_pass(path[i][0])
                        inputs.append(curr_state_tensor)
                        w_inputs.append(path[i][1])
                        t_inputs.append(path[i][2])
                    if last_state_is_goal:
                        for i in range(path_length):
                            labels.append(1)
                        states.remove(state)
                    else:
                        for i in range(path_length):
                            labels.append(0)
                T *= 2
            W_models[w_values.index(w)].train_model(inputs, w_inputs, t_inputs, epochs, labels)



