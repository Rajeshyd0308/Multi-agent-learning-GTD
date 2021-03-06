# Multi-agent-learning-GTD
Developed a multi-agent Gradient TD Learning algorithm for solving an MDP using which each agent learns common set of parameters without sharing the data samples.
Analyzed the decay of TD error of this algorithm under communication constraints in multi-agent variant 2D Maze environments. We used Linear function approximation (fourier basis) for approximating the value function. 

# About the experiments:
We investigate the decay of norm of expected TD errors which expalins how fast the proposed algorithm converges. For this part of experiments we use 20 different maze likeenvironments.   Each  maze  consists  of  a  start,  end  and  blocks  through  whichagent in the maze cannot traverse.  Each maze environment differs from othersby the location of the blocks.  Each individual maze consists of an agent and allof the agents can communicate their parameter vector with a centralized server.Each agent communicates its parameters with the centralized server for every’t’ time steps.  Each agent follows a fixed deterministic policy.  Overall, in ourexperimental setting a multi-agent multi task policy evaluation problem, whereall the agents are collectively solving for parameters which approximate theirrespective value functions.To analyze the convergence of parameters we look at NEU(Norm of Expectedtd  Updates)  as  an  indicator  for  convergence.   We  show  that  NEU  convergesto  zero  as  the  time  steps  increase.   We  also  experiment  with  time  interval  ofcommunication between the agents i.e.  for how many time steps does each agentcommunicate with the centralized server.  We found that the more frequentlyeach agent communicates with the centralized server the faster the NEU decays.

# Some of the results:
![alt text](https://github.com/Rajeshyd0308/Multi-agent-learning-GTD/blob/main/results.png)
neu_C_1 corresponds to Norm of Expected TD Updates for experimental setting in which each agent communicates for every time step.
neu_C_3 corresponds to Norm of Expected TD Updates for experimental setting in which each agent communicates for every 3 time steps.
