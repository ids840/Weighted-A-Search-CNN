import matplotlib.pyplot as plt
from model import GridStateCNN
# from model import ModelManager
from problem_env.games.grid_game.game import GridGame
from training import LabelGenerator
from w_a_star import weighted_A_star_const
import time
import os

def display_path_with_delay_and_clear(path, delay_time):
    def clear_console():
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

    for item in path:
        # If `item` is a tuple, extract the state from the tuple
        if isinstance(item, tuple):
            state = item[0]
        else:
            state = item
            print('state goal:', state.goal)
        clear_console()
        state.display_state()  # Call display_state() on the state object
        time.sleep(delay_time)


def w_for_state(w_models, state, w_values, T):
    w =  w_values[-1]
    found = False
    for i in range(len(w_models)):
        has_solution = w_models[i].forward(state, T).item()
        if has_solution > 0.5 and not found:
            found=True
            w = w_values[i]
    return w
def main():
    grid_size = (20, 20)
    game = GridGame(grid_size=grid_size)

    # get_model_from_version_number = 0
    # problem_name = f"grid_maze_train_iterations_{game.grid_size[0]}_{game.grid_size[1]}_{get_model_from_version_number}"
    # new_version_saved_problem_name = problem_name

    # model_manager = ModelManager()
    max_iterations = 4    # 5
    max_expansions = 150         # 100
    initial_t = 18 # grid_size[0] // 10 + 4    # 2
    number_of_samples = 1000
    epochs = 1
    training_iterations = 7
    to_train = True
    w_values = [1, 2, 5, 7, 10]

    models = []
    for i in range(len(w_values)):
        models.append(GridStateCNN((20,20),5))
    # train
    if to_train:
        # model = model_manager.get_or_create_model(problem_name=problem_name,
        #                                           input_channels=game.get_input_channels_num(),
        #                                           num_classes=1,
        #                                           grid_size=game.grid_size)
        label_generator = LabelGenerator(game=game)
        print("Training model...")
        for i in range(training_iterations):
            print(f"training iteration {i + 1}/{training_iterations}")
            label_generator.bootstrap_training(w_values,models, epochs,initial_t,
                                                    max_iterations)
        print("Training complete.")
        # print("Saving model...")
        # last_char = new_version_saved_problem_name[-1]
        # if last_char.isdigit():
        #     version_number = int(last_char) + 1
        #     new_version_saved_problem_name = new_version_saved_problem_name[:-1] + str(version_number)
        # else:
        #     version_number = 1
        #     new_version_saved_problem_name += "_1"
        # model_manager.save_model(model, problem_name=new_version_saved_problem_name)
        # print("Model saved.")


    # # load
    # else:
    #     model = model_manager.load_model(problem_name=problem_name, grid_size=game.grid_size)
    #     start = game.generate_random_states(1)[0]
    #     path, expansions = weighted_A_star(start=start, w_model=model, game=game, w_values = [1,2,5,7,10],
    #                                        max_number_of_expansions=1000000, train=0)
    #     if path:
    #         display_path_with_delay_and_clear(path,0.3)
    #     print(expansions)

    array_of_not_solutions = [0] * len(w_values)
    array_of_path_lengths = [0] *  len(w_values)
    array_of_path_lengths_for_random = [0] *  len(w_values)
    random_w_not_solutions = 0
    samples = number_of_samples

    print("Running experiments...")
    for i in range(samples):
        start = game.generate_random_states(1)[0]
        success = [0] * len(w_values)
        for w in range(len(w_values)):
            path = weighted_A_star_const(start,game,max_expansions,w_values[w])
            last_state_is_goal = path[-1][0].position == path[-1][0].goal
            if not last_state_is_goal:
                array_of_not_solutions[w] += 1
            else:
                success[w] = 1
                array_of_path_lengths[w] += len(path)
        w_for_random = w_for_state(models,game.get_state_for_forward_pass(start),w_values,max_expansions)
        path = weighted_A_star_const(start, game, max_expansions, w_for_random)
        last_state_is_goal = path[-1][0].position == path[-1][0].goal
        if not last_state_is_goal:
            random_w_not_solutions += 1
        else:
            for j in range(len(w_values)):
                if success[j] == 1:
                    array_of_path_lengths_for_random[j] += len(path)

    all_not_solutions = array_of_not_solutions + [random_w_not_solutions]
    print(all_not_solutions)
    labels = [f'w={w}' for w in w_values] + ['Random w']
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(all_not_solutions)), all_not_solutions, alpha=0.6, color='blue')
    plt.xticks(range(len(all_not_solutions)), labels)
    plt.xlabel('w Values')
    plt.ylabel('Number of Non-Solutions')
    plt.title('Comparison of Non-Solutions for Each w (Including Random w)')
    plt.show()

    # Calculate averages
    avg_path_lengths = [array_of_path_lengths[i] / (samples-array_of_not_solutions[i]) for i in range(len(w_values))]
    avg_path_lengths_for_random = [array_of_path_lengths_for_random[i] / (samples-array_of_not_solutions[i]) for i in range(len(w_values))]

    # Plot for each w value
    for i in range(len(w_values)):
        plt.figure(figsize=(10, 5))
        plt.bar(['w', 'Random'], [avg_path_lengths[i], avg_path_lengths_for_random[i]], color=['blue', 'green'],
                alpha=0.7)

        plt.xlabel('w Type')
        plt.ylabel('Average Path Length')
        plt.title(f'Average Path Length Comparison for w={w_values[i]}')
        plt.show()


if __name__ == '__main__':
    main()