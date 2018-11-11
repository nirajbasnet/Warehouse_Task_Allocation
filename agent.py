import threading
import Gridworld
import numpy as np
import copy

# np.random.normal(mu,sigma,length of weights array)
task_list = np.array([4, 1, 1, 2])


def update_tasklist(index):
    task_list[index] -= 1


def is_empty_tasklist():
    if task_list.sum() != 0:
        return False
    return True


class Agent:
    def __init__(self, grid_world, pos_idx):
        self.grid_world = grid_world
        self.pos = self.grid_world.agents_pos[pos_idx]
        self.input_space = 16  # (pos_x,pos_y,item_x*4,item_y*4,bin_x,bin_y,task*4(of tasklist)
        self.output_space = 4  # len(grid_world.len(items)
        self.hidden_space = 16
        self.population_size = 2
        self.weights_ih = []  # weights from input to hidden layer
        self.weights_ho = []  # weights from hidden to output layer
        self.k = 2  # no of policies per agent in population
        self.scores = [0 for i in range(self.k)]  # scores for k policies
        self.current_task = 0
        self.current_policy = 0
        self.alpha = 0.2  # update parameter for scores

        self.output = [0 for i in range(len(task_list))]
        self.input_nn = []
        self.update_input()
        self.initialize_policies(self.k)

    def test_gridworld_connection(self):
        print("Position ", self.pos)
        print("input_nn", self.input_nn)
        self.initialize_policies(self.k)
        print(self.weights_ho)
        # print(self.act(0))
        self.mutate_policies()
        print(self.weights_ho)

        # print(self.W_ho)

    def initialize_policies(self, k):
        print("initializing weights for population of k policies")
        for i in range(k):
            self.weights_ih.append(np.random.randn(self.input_space,
                                                   self.hidden_space))  # (k*16*16) weight matrix from input to hidden layer
            self.weights_ho.append(np.random.randn(self.hidden_space,
                                                   self.output_space))  # (k*16*4) weight matrix from hidden to output layer

    def select_task(self):
        print("selecting task for agent")
        soft_max_prob = self.act(self.current_policy)  # running neural net on best policy
        best_task = np.argmax(soft_max_prob)  # best task given the softmax values for each type of item in gridworld
        count = 0
        while task_list[best_task] == 0:
            count += 1
            if count == len(task_list):
                return -5
            soft_max_prob[best_task] = 0
            best_task = np.argmax(soft_max_prob)


        self.current_task = best_task
        update_tasklist(self.current_task)

    def get_time_steps(self, current_task):
        print("computing time steps for current task")
        t_agent_item = abs(self.grid_world.items_pos[current_task][0] - self.pos[0]) + abs(
            self.grid_world.items_pos[current_task][1] - self.pos[1])
        # self.pos=copy.deepcopy(self.grid_world.items_pos[current_task])
        # t_item_bin = abs(self.grid_world.bins_pos[current_task][0]-self.pos[0]) + abs(self.grid_world.bins_pos[current_task][0] - self.pos[0])
        t_item_bin = abs(self.grid_world.bins_pos[current_task][0] - self.grid_world.items_pos[current_task][0]) + abs(
            self.grid_world.bins_pos[current_task][1] - self.grid_world.items_pos[current_task][1])
        total_timesteps = t_agent_item + t_item_bin
        return total_timesteps

    def update_input(self):
        print("updating input to neural net")
        self.input_nn = []
        self.input_nn = [self.pos[0], self.pos[1]]
        for i in task_list:
            self.input_nn.append(i)
        for i in range(len(grid_world.items)):
            self.input_nn.append(grid_world.items_pos[i][0])
            self.input_nn.append(grid_world.items_pos[i][1])
        for i in range(0, 1):  # increase range to select all bins if required
            self.input_nn.append(grid_world.bins_pos[i][0])  # selecting first bin only
            self.input_nn.append(grid_world.bins_pos[i][1])


    def update_states(self, current_task):
        print("updating states of agent")
        self.pos = copy.deepcopy(self.grid_world.bins_pos[current_task])

    def act(self, policy_idx):
        print('running neural network to get task selection output for particular policy')
        self.update_input()
        # dot product of input and weights matrix for input-hidden layer
        ih_result = np.dot(self.input_nn, self.weights_ih[policy_idx])
        h_layer = self.sigmoid(ih_result)  # activation function
        # dot product of hidden layer  and weights matrix for hidden-output layer
        ho_result = np.dot(h_layer, self.weights_ho[policy_idx])
        o_layer = self.sigmoid(ho_result)  # final activation function
        softmax_out = self.softmax(o_layer)
        return softmax_out

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def softmax(self, s):
        # compute softmax for output
        e_s = np.exp(s - np.max(s))
        return e_s / e_s.sum()

    def mutate_policies(self):
        print("mutating k policies to find 2k policies")
        mean, std_dev = 0, 0.1
        for i in range(self.k):
            noise_ih = np.random.normal(mean, std_dev, size=np.shape(self.weights_ih[i]))
            noise_ho = np.random.normal(mean, std_dev, size=np.shape(self.weights_ho[i]))
            new_weights_ih = np.add(self.weights_ih[i], noise_ih)
            new_weights_ho = np.add(self.weights_ho[i], noise_ho)
            self.weights_ih.append(new_weights_ih)
            self.weights_ho.append(new_weights_ho)
            self.scores.append(0)

    def get_random_policy(self):
        print("getting random policy from agent policies")
        self.current_policy = np.random.randint(0, self.k)
        return self.current_policy

    def get_best_policy(self):
        print("getting best policy from agent policies")
        self.current_policy = np.argmax(self.scores)
        return self.current_policy

    def update_score(self, score, policy_index):
        print("updating score of agent policies")
        self.scores[policy_index] += self.alpha * (score - self.scores[policy_index])

    def reset_scores(self):
        print("resetting scores")
        self.scores = []

    def get_next_gen(self):
        print("selecting best k from 2k policies")
        for i in range(self.k):
            low_value_idx = int(np.argmin(self.scores))
            self.weights_ih.pop(low_value_idx)
            self.weights_ho.pop(low_value_idx)
            self.scores.pop(low_value_idx)


class CCEA:
    def __init__(self, grid_world):
        x = 0
        self.gridWorld = grid_world
        self.team = []
        self.N_pop = 2  # number of populations
        self.agents = []
        self.iteration_count = 1
        self.generations = 1

    def get_random_tasklist(self):
        print("getting random task list for simulations")
        rand_list = []
        for i in range(len(task_list)):
            rand_list.append(np.random.randint(0, 5))
        r_list = np.array(rand_list)
        if r_list.sum() == 0:
            self.get_random_tasklist()
        return np.array(rand_list)

    def init_all_agent_populations(self, size):
        print("initializing all agent populations")
        self.agents = [Agent(self.gridWorld, i) for i in range(size)]

    def init_team(self):
        print("selecting random policy from each population and forming a team")
        for agent in self.agents:
            self.team.append(agent.get_random_policy())

    def run_sim(self, team):
        print("running team simulation")
        # reset environment
        total_sim_time = 0
        absent = None
        T_all = np.array([0 for i in range(len(self.agents))])
        print(team)
        print("original task list",task_list)
        for index, agent in enumerate(self.agents):
            if team[index] != -5:
                agent.select_task()
                print("agent ",index," current task=",agent.current_task)
                t_single = agent.get_time_steps(agent.current_task)
                T_all[index] = t_single
            else:
                absent = index
            print("task list", task_list,T_all)
        while T_all.sum() > 0:
            min_time_idx = np.argmin(T_all)
            count = 0
            while T_all[min_time_idx] == 0 and is_empty_tasklist():
                print("task list empty")
                count += 1
                smallest=float('inf')
                if count == len(T_all):
                    break
                for i in range(len(T_all)):
                    if i != min_time_idx and T_all[i]<smallest:
                        smallest = i
                min_time_idx = smallest
            print("min_time ",min_time_idx," ",T_all[min_time_idx])
            total_sim_time += T_all[min_time_idx]
            agent_same_time = []
            for index, agent in enumerate(self.agents):
                if index == min_time_idx and index != absent:
                    agent.update_states(agent.current_task)
                    print("agent ",index," pos= ",agent.pos)
                    if agent.select_task() != -5:
                        print("agent ", index, " current task=", agent.current_task)
                        agent_same_time.append(index)
            # T_all = T_all - T_all[min_time_idx]
            minimum_time = T_all[min_time_idx]
            for k in range(len(T_all)):
                if T_all[k] != 0:
                    T_all[k] = T_all[k] - minimum_time
                    print(T_all[k])
            print("subract ",T_all)
            for i in agent_same_time:
                T_all[i] += self.agents[i].get_time_steps(self.agents[i].current_task)
            print(task_list,T_all)
            x = input("enter")
        return total_sim_time

    def run_evolutions(self):
        print("running evolution")
        analysis = []
        self.init_all_agent_populations(2)  # size of the population
        for i in range(self.generations):
            for agent in self.agents:
                agent.mutate_policies()
            self.stage1()
            self.build_hof_diff()
            global_reward = self.evolution()
            analysis.append(global_reward)
            self.gridWorld.reset()
            for agent in self.agents:
                agent.reset_scores()

        # computation += stage1()
        # computation1 +=build_hof_diff()
        # global reward = evolution()
        # analysis.append((computation,global_reward))
        # reset scores

    def evolution(self):
        print("evolving each population")
        best_team = []
        for agent in self.agents:
            agent.get_next_gen()
            best_policy = agent.get_best_policy()
            best_team.append(best_policy)
        best_score = self.run_sim(best_team)
        return best_score

    def stage1(self):
        global task_list
        print("executing stage 1")
        for i in range(self.iteration_count):
            task_list = self.get_random_tasklist()
            team = []
            for agent in self.agents:
                team.append(agent.get_random_policy())
            score = self.run_sim(team)
            print("score stage1: ", score)
            for agent in self.agents:
                agent.update_score(score, agent.current_policy)

    def build_hof_diff(self):
        print("building hof_diff model")
        generations_count = 100
        for i in range(generations_count):
            print("running generation ", i)
            task_list = self.get_random_tasklist()
            for index, agent in enumerate(self.agents):
                team = []
                rand_policy = agent.get_random_policy()
                team.append(rand_policy)
                for other_agent in self.agents:
                    if other_agent != agent:
                        best_policy = other_agent.get_best_policy()
                        team.append(best_policy)
                score_with_a = self.run_sim(team)
                present_policy = team[index]
                team[index] = -5
                score_without_a = self.run_sim(team)
                diff_score = score_with_a - score_without_a
                agent.update_score(diff_score, present_policy)

    def test_func(self):
        print("testing function")
        self.run_evolutions()



if __name__ == "__main__":
    grid_world = Gridworld.World(Gridworld.root, 10, 10)
    grid_world.initialize_world()
    TA = CCEA(grid_world)

    t = threading.Thread(target=TA.test_func())
    t.daemon = True
    t.start()
    Gridworld.start_world()

