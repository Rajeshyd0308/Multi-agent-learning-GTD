import numpy as np
import copy
import math
import itertools
import matplotlib.pyplot as plt
import pandas as pd

class FourierBasis(object):
    """
    Parameters
    ----------
    d: int, optional
        The degree of the fourier basis

    n: int
        The order of the fourier basis
    """

    def __init__(self, d, n = 3):
        self.d = d
        self.n = n
        self.multipliers = FourierBasis._multipliers(d, n)

    def get_num_basis_functions(self):
        """Gets the number of basis functions
        """
        if hasattr(self, 'num_functions'):
            return self.num_functions

        self.num_functions = (self.n + 1.0) ** self.d
        return self.num_functions

    def compute_features(self, features):
        """Computes the nth order fourier basis for d variables
        """
        return self.compute_scaled_features(features)

    def compute_scaled_features(self, scaled_features):
        """Computes the nth order fourier basis for d variables
        """
        if len(scaled_features) == 0:
            return np.ones(1)
        return np.cos(np.pi * np.dot(self.multipliers, scaled_features))

    def compute_gradient(self, scaled_features):
        """Computes the gradient of the fourier basis
        """
        if len(scaled_features) == 0:
            return np.zeros(1)

        # Calculate outer derivative
        outer_deriv = -np.sin(np.pi * np.dot(self.multipliers, scaled_features))

        # Calculate inner derivative
        inner_deriv = np.pi * self.multipliers

        return (inner_deriv.T * outer_deriv).T


    @staticmethod
    def _multipliers(d, n):
        """Generates multipliers for the fourier basis.
        This corresponds to the c vector in the paper
        """
        arrays = [list(range(n + 1)) for _ in itertools.repeat(None, d)]
        return FourierBasis._cartesian(arrays)

    @staticmethod
    def _cartesian(arrays, out=None):
        """Generate a cartesian product of input arrays.
        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.
        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.
        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])

        From scikit-learn's extra math
        """
        arrays = [np.asarray(x) for x in arrays]
        shape = (len(x) for x in arrays)
        dtype = arrays[0].dtype

        ix = np.indices(shape)
        ix = ix.reshape(len(arrays), -1).T

        if out is None:
            out = np.empty_like(ix, dtype=dtype)

        for n, arr in enumerate(arrays):
            out[:, n] = arrays[n][ix[:, n]]

        return out

class Maze:
    """
    Creates a 10x10 maze with mines
    0,1,2,3 are the actions. 0 is up, 1 is right, 2 down , 3 left
    state[0] is row, state[1] is col
    Reward at all the places is -1 and reward at mines is -50, at exit reward is 100
    mines is an array like [13, 16, 28,....] mines at location 13..
    You can treat is as any environment in Gym."""
    def __init__(self, mines):
        """Initializes the 10x10 maze with mines and reward"""
        self.layout =-1*np.ones((10,10))
        for m in mines:
            self.layout[int(m/10),int(m%10)] = -50
        self.layout[9,9] = 100

    def reset(self):
        """returns the start state"""
        ls = np.array([0, 0])
        return ls

    def step(self, current_state, action):
        """
        Takes the action at current state and returns next state
        params:
        current_state - current state of the agent in Maze environment
        action - action to be taken at current state
        return:
        new_state - next state after taking action at current state
        reward - reward returned after going to new state
        """
        new_state = np.array([0, 0])
        reward = None
        done = None
        new_state[0] = current_state[0]
        new_state[1] = current_state[1]
        if action==0:
            if current_state[0]!=0:
                new_state[0]=current_state[0]-1

        elif action==1:
            if current_state[1]!=9:
                new_state[1]=current_state[1]+1

        elif action==2:
            if current_state[0]!=9:
                new_state[0]=current_state[0]+1
        else:
            if current_state[1]!=0:
                new_state[1]=current_state[1]-1
        reward = self.layout[new_state[0], new_state[1]]
#         if new_state[0]==9 and new_state[1]==9:
#             done = True
        return new_state, reward, done
#GTD
class Agent:
    """GTD agent in the maze"""
    def __init__(self, dim):
        """
        w- corresponds to Parameters
        v- estimate of inverse gradient term in GTD
        temp- keeps track of error in GTD updates
        dim- dimension of vector w (decided by our fourier basis vector)"""
        self.dim = dim
        self.w = np.zeros(dim)
        self.gamma =0.95
        self.temp = np.zeros(dim)
        self.v = np.zeros(dim)

    def fourier_basis_vec(self, state, fourier_vec):
        """Returns the fourier basis vector
        params:
        state- state for whoch fourier basis vector is to be computed
        fourier_vec - object of Fourier basis class above"""
        s1 = state[0]/9
        s2 = state[1]/9
        st = np.array([s1, s2])

        return fourier_vec.compute_features(st)

    def update(self, current_state, next_state, action, reward, lr, fourier_vec, beta):
#         print(self.temp)
        """
        Updates the parameters w, v (parameters and inverse gradient term)
        params:
        current_state, next_state, action taken, reward, lr (alpha) slow parameter, beta- fast paramter
        fourier_vec-  object of Fourier basis class above
        """
        st = self.fourier_basis_vec(current_state, fourier_vec)
        nst = self.fourier_basis_vec(next_state, fourier_vec)
#         print(st)
        reward = reward/100
        val_st = np.dot(self.w, st)
        val_nst = np.dot(self.w, nst)
        delta_t = reward+ (self.gamma*val_nst) - val_st # delta

        inverse_term = np.dot(self.v, st) #z update
        diff = (delta_t-inverse_term)*st
        self.w = self.w+ lr*(st-self.gamma*nst)*np.dot(self.v, st)
        self.temp += (st-self.gamma*nst)*np.dot(self.v, st)
        self.v = self.v + beta*(diff)

#         print(self.temp)
# episodes=100
def train(com, num_agents, episodes):
    """Contains the main code to train agents in maze env. For each agent a new environment will be created
    com - number of time steps after which agents should communicate
    num_agents - Number of agents
    episode- number of episodes
    return:
    returns a list of errors (error computed after each episode)"""
    np.random.seed(1)
    env = []
    agent = []
    mine = []
    policy = []
    fourier_vec = FourierBasis(2, 3) # 3 since 3 elements in state action vector, 2 order of fourier basis [0, 1, 2]
    print(fourier_vec.get_num_basis_functions())
    """Creates many agnets and env"""
    for i in range(0, num_agents):
        """For each agent a new env will be created """
        mine = np.random.randint(0,99, size=4)
        env_temp = Maze(mine)
        env.append(env_temp)
#         mine = np.random.randint(0,99, size=4)
        """New agent created here"""
        agent_temp = Agent(int(fourier_vec.get_num_basis_functions()))
        agent.append(agent_temp)
        p = np.random.randint(0,4, size=(10,10))
        policy.append(p)

    # neu_freq = 100
    # """Number of episodes"""
    # episodes = 1000
    """Total steps per episode"""
    total_steps = 20
    num_steps = 0
    comm = com
    neu_freq = 20
    start_num = num_steps
    neu_1 = []
    neu_2 = []
    neu = []
    for i in range(0, episodes):
        state = []
        for k in range (0, num_agents):
            state.append(env[k].reset())

        for j in range(0, total_steps):
            """Below are the learning rates lr is slow and beta is fast"""
            lr = 1/((num_steps+1)**1)
            beta = 1/((num_steps+1)**0.7)
            for l in range(0,num_agents):
                p = policy[l]
                s = state[l]
                action = p[s[0], s[1]]
                ns, reward, done = env[l].step(s, action)
                a = agent[l]
                """Update of w, v and neu rae computed in next line"""
                a.update(s, ns, action, reward, lr, fourier_vec, beta)
                state[l] = ns
            num_steps+=1
            """After every comm time steps the averaging happens"""
            if num_steps%comm==0:
                new_w = np.zeros(int(fourier_vec.get_num_basis_functions()))
                new_v = np.zeros(int(fourier_vec.get_num_basis_functions()))
                for l in range(0, num_agents):
                    new_w += agent[l].w
                    new_v += agent[l].v

                new_w = new_w/num_agents
                new_v = new_v/num_agents
                for l in range(0, num_agents):
                    agent[l].w = new_w
                    agent[l].v = new_v
            # if num_steps-start_num==neu_freq:
        """Computes the error across all the agents"""
        start_num = num_steps
        tem = np.zeros(int(fourier_vec.get_num_basis_functions()))
        for l in range(0, num_agents):
            tem+=agent[l].temp
            agent[l].temp = np.zeros(int(fourier_vec.get_num_basis_functions()))
        tem = tem/(num_agents)
        neu.append((1/total_steps)*np.sqrt(np.sum(tem**2)))
    return neu

"""neu corresponds to the error. train(1, 20) creates 20 agents in 20 different environments which communicate after every 1 timestep"""
"""These are the main lines, if we need to change number fo agents, local steps between communication..."""
episodes=100
neu_C_1 = train(1, 20, episodes)
neu_C_3 = train(3, 20, episodes)
neu_C_7 = train(7, 20, episodes)
neu_C_9 = train(9, 20, episodes)

"""The code below is for plotting the graphs in the same order"""
ls = []
neu_C_1_disp = []
neu_C_3_disp = []
neu_C_7_disp = []
neu_C_9_disp = []
# ls.append(1)
for i in range(0, episodes):
    ls.append(i+1)
    neu_C_1_disp.append(neu_C_1[int(i)])
    neu_C_3_disp.append(neu_C_3[int(i)])
    neu_C_7_disp.append(neu_C_7[int(i)])
    neu_C_9_disp.append(neu_C_9[int(i)])
print(len(ls))
df=pd.DataFrame({'x': ls, 'neu_C_1': neu_C_1_disp, 'neu_C_3': neu_C_3_disp, 'neu_C_7': neu_C_7_disp, 'neu_C_9': neu_C_9_disp})
plt.rcParams["figure.figsize"] = (14, 12)
plt.plot( 'x', 'neu_C_1', data=df, marker='o', markerfacecolor='blue', markersize=2, color='skyblue', linewidth=2, label="neu_C_1")
plt.plot( 'x', 'neu_C_3', data=df, marker='', color='green', linewidth=2, label="neu_C_3")
plt.plot( 'x', 'neu_C_7', data=df, marker='', color='yellow', linewidth=2, label="neu_C_7")
plt.plot( 'x', 'neu_C_9', data=df, marker='', color='red', linewidth=2, linestyle='dashed', label="neu_C_9")
# plt.plot( 'x', 'tabular30_lam0.5', data=df, marker='', color='magenta', linewidth=2)
plt.xlabel("Number of episodes")
plt.ylabel("Norm of Expected TD Update")
plt.legend()
plt.show()
