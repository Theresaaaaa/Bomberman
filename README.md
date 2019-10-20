# Bomberman
Bomberman game based on reinforcement learning
<br>
## INTRODUCTION<br>
### A.	Analysis rule of the game<br>
The environment of the game is a maze, which can be represented by a two-dimensional matrix. There are two main characters in the game, bomber man and monster. They have different attributes; bomber man is twice as fast as monster. The two characters can choose different actions, bomber man can walk and release bombs, monster can attack bomber man and so on. After the action, the current state will be changed into a new state, such as displacement, attack, etc.<br>
### B.	The main idea of reinforced learning<br>
Q-learning, a famous method of the reinforced-learning. After setting the action value and state value, the best result of the next step can be obtained through realistic estimation and Q estimation. The core of the estimation lies in the maximum estimation of the next step decay and the current reward as the reality of this step. Epsilon greedy is used in the decision on a strategy, such as Epsilon = 0.9, it has 90% of the time I will be in accordance with the optimal value of Q meter choice behavior, 10% of the time use random behavior. 
<br>
## MODEL SIMPLIFICATION<br>
### A.	Different version of the program<br>
Version1: one maze and one bomber man without monster, the aim of this version is to train the bomber man to walk through the maze and avoid the wall.
Version2: add the monster, the monster can eat the bomber man, and the bomber man can’t set bomb yet
Version3: the bomber man has ability to set the boom and after a period of time the boom explode, it may kill the monster or itself.
Version4: like the real game more monsters, and further modification.
Versions of the program can be seen at: here
<br>
### B.	Main logic<br>
The main functions and what they do:
First, initialize the environment in the main function, that is, establish the maze, set the starting point of bomber man and monster, and initialize various parameters. Since the speed of the bomber man is faster, I let the monster to pause for 1 step each time.
 Then, build a game loop that allows AI players to gain experience through multiple game attempts.
 In q-learning, we will initialize Q table through ‘build_q_table’ function, which will record the quality and degree of each state corresponding action. Then ‘choose_action’ function will be used to select the action according to Q table, and then the action will be input into ‘get_env_feedback’ function to get environment feedback, and the corresponding reward will be used to accumulate experience. Finally, use the move function to move the character to the corresponding position to print the image, when the condition can be reached to end the game.
<br>
## FUNCTION SPECIFICATIONS<br>
###A.	Choose action<br>
This function is using the Greedy arithmetic to predict the action, after setting the parameter epsilon, it will choice the best action according to the Q table in epsilon present of time.
