# Bomber Man Game
Bomberman game based on reinforcement learning
<br>
## INTRODUCTION<br>
### A.	Analysis rule of the game<br>
The environment of the game is a maze, which can be represented by a two-dimensional matrix. There are two main characters in the game, bomber man and monster. They have different attributes; bomber man is twice as fast as monster. The two characters can choose different actions, bomber man can walk and release bombs, monster can attack bomber man and so on. After the action, the current state will be changed into a new state, such as displacement, attack, etc.<br>
### B.	The main idea of reinforced learning<br>
Q-learning, a famous method of the reinforced-learning. After setting the action value and state value, the best result of the next step can be obtained through realistic estimation and Q estimation. The core of the estimation lies in the maximum estimation of the next step decay and the current reward as the reality of this step. Epsilon greedy is used in the decision on a strategy, such as Epsilon = 0.9, it has 90% of the time I will be in accordance with the optimal value of Q meter choice behavior, 10% of the time use random behavior. <br>
<img src="https://morvanzhou.github.io/static/results/ML-intro/q4.png" width="675" height="278" alt="From mofan">
<br>
## MODEL SIMPLIFICATION<br>
### A.	Different version of the program<br>
Version1: one maze and one bomber man without monster, the aim of this version is to train the bomber man to walk through the maze and avoid the wall.<br>
Version2: add the monster, the monster can eat the bomber man, and the bomber man can’t set bomb yet
Version3: the bomber man has ability to set the boom and after a period of time the boom explode, it may kill the monster or itself.<br>
Version4: like the real game more monsters, and further modification.<br>
Versions of the program can be seen at: bombermanV2
<br>
### B.	Main logic<br>
The main functions and what they do:<br>
First, initialize the environment in the main function, that is, establish the maze, set the starting point of bomber man and monster, and initialize various parameters. Since the speed of the bomber man is faster, I let the monster to pause for 1 step each time.<br>
 Then, build a game loop that allows AI players to gain experience through multiple game attempts.
 In q-learning, we will initialize Q table through ‘build_q_table’ function, which will record the quality and degree of each state corresponding action. Then ‘choose_action’ function will be used to select the action according to Q table, and then the action will be input into ‘get_env_feedback’ function to get environment feedback, and the corresponding reward will be used to accumulate experience. Finally, use the move function to move the character to the corresponding position to print the image, when the condition can be reached to end the game.
<br>
## FUNCTION SPECIFICATIONS<br>
### A.	Choose action
This function is using the Greedy arithmetic to predict the action, after setting the parameter epsilon, it will choice the best action according to the Q table in epsilon present of time.<br>

### B.	Update environment function and the reward<br>
In order to teach the character playing strategy, every time it takes an action, a reward (positive or negative) will be given.<br>
For each selected action, it will move to a different state, for each different state we can predict the outcome, like knock into the wall, or meet the monster. The reward will further been put into the Q table and help further action choosing.<br>
## FURTHER MODIFICATION
1.	Set different levels of difficulty for the game<br>
In my code, the monster will learn don't hit the wall and avoid the bomb, but did not learn to find bomber man, because the prize is manually set, so can let monsters into different degrees by changing the parameters of the reward of learning, from random walk to conscious escape bombs and looking for bomber man, and gradually raise the difficulty.<br>
2.	Randomly set the number of monster and boom<br>
Since the number of monster and boom are fixed in the question we can use the array to store the monster and boom object since the logic is the same <br>
3.	To optimize the interface<br>
With the python gym package, you can create a more user-friendly interface<br>
4.	Using the neural network to choose action<br>
The fully integrated link layer in TensorFlow can be used to generate the required Q and q-target networks. Input is the input of the network; output is the number of nodes input (output dimensional unit), where is the number of action Spaces, which is 4.<br>
### OUTCOME OF MY CODE
After observing the results of each round, I found that at the beginning, bomber man is likely to be killed by the bomb set by himself, which is same as the result when I was playing the game. Also, when in some round bomber man will be eaten by monsters. 
As the number of rounds of the game increases, bomber man tends to make his reward close to 0 instead of negative number by blowing up himself and the monster at the same time. Training such games requires many rounds.
## REFERENCE
[1]	Reinforce learning.  MoFan, [Online]. Available: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/1-1-A-RL/<br>
[2]	Gym (OpenAI). [Online]. Available: https://gym.openai.com/envs/BankHeist-v0/<br>


