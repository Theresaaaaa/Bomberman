# -*- coding: utf-8 -*-
"""
Created on Mon Sep  23

--------------------------------Bomerman 游戏规则---------------------------------

用10*10矩阵代表迷宫，可以走的部分是0，不能走的部分是1
B表示爆炸人，就是玩家角色
M表示怪物，简单版本设置怪物固定为3个，怪物的【行进路线】是【随机】的，不能撞墙，所以怪物可以看作是走迷宫里面追寻动态终点的玩家
    行进速度：v
    行动：up， down， left ，right
    奖励 ：抓到爆炸人
    惩罚【可选项目】：撞墙（暂停一次）被爆炸人炸掉（游戏结束）
    *********************优化方案：让怪物也可以尝试强化学习，增加难度）********************************

爆炸人B的【行进路线】不用涉及撞墙，【Al玩家拥有完整的棋盘视野】， 但是由于玩家是AI，实际上也需要尝试走迷宫的方式
    行进速度：2v
    行动：up， down， left ，right，释放炸弹
    奖励 ：炸弹炸掉怪物M，以及行走过程中没有撞墙（r=-0.1）
    惩罚：撞墙（暂停一次）被爆炸人炸掉（游戏结束）

游戏结束方式
炸弹人烧伤自己或被怪物砸碎时

假设你必须为一个Al玩家编写程序来控制炸弹人去轰炸并杀死一个带有随机墙壁的10×10的棋盘上的所有3个怪物。Al玩家拥有完整的棋盘视野(即它能看到所有的墙壁和怪物的位置)，
并且走的速度是怪物的两倍，而怪物是随机地在棋盘上走的。
炸弹人一次只能投下一枚强度为2的炸弹(四个方向各2格)来杀死怪物，当炸弹人烧伤自己或被怪物砸碎时，游戏结束。里美你可以假设游戏没有时间限制


version: 2.0

@author: Theresa
"""
import numpy as np
import pandas as pd
import time
import os

WIDE = 12
HEIGHT = 12
N_STATES = WIDE * HEIGHT  # 除去墙其实也是一个 5*5
ACTIONS_M = ['left', 'right', 'up', 'down']  # monster's available actions
ACTIONS_B = ['left', 'right', 'up', 'down', 'setBoom']  # bomberman's available actions
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 150  # maximum episodes  基本上就能够跑出来最小四步辽
FRESH_TIME = 0.3  # fresh time for one move
BOOMTIMES = 1  # 炸弹可以爆炸的次数
BOOM_WAIT = 3

maze = np.asarray([["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
                   ["X", "B", " ", " ", " ", " ", " ", "X", " ", " ", " ", "X"],
                   ["X", " ", " ", " ", " ", " ", " ", "X", " ", " ", " ", "X"],
                   ["X", " ", " ", " ", "X", "X", "X", " ", " ", " ", " ", "X"],
                   ["X", "X", " ", " ", " ", " ", "X", " ", " ", "X", "X", "X"],
                   ["X", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "X"],
                   ["X", " ", " ", " ", "X", "X", " ", " ", " ", " ", " ", "X"],
                   ["X", " ", " ", " ", " ", " ", " ", " ", "X", " ", " ", "X"],
                   ["X", " ", " ", " ", " ", " ", " ", " ", "X", " ", " ", "X"],
                   ["X", "X", "X", "X", "X", "X", "X", " ", "X", " ", " ", "X"],
                   ["X", " ", " ", " ", " ", " ", " ", " ", " ", " ", "M", "X"],
                   ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],])  # 迷宫的样子
index = np.arange(WIDE * HEIGHT).reshape(WIDE, HEIGHT)
record_S_M = []
boom_state = []

print(index)

start_sign_B = np.asarray(["B"])  # 初始化主人公B的位置
start_state_B = int(index[maze == start_sign_B])
print(start_state_B)

start_sign_M = np.asarray(["M"])  # 初始化 M 的位置        【要实现实时跟踪M的位置】
start_state_M = int(index[maze == start_sign_M])
print(start_state_M)

wall_state = []  # 初始化所有墙的集合
wall_sign = np.asarray(["X"])
for wall in (index[maze == wall_sign]):
    wall_state.append(wall)
print(wall_state)  # 所有的墙

path_state = []  # 初始化所有通道的集合
path_sign = np.asarray([" "])
for path in (index[maze == path_sign]):
    path_state.append(path)
print(path_state)  # 所有的通道

print('--------------stop preproducing----------------------')


# -------------------------------------------------首先实现 B 走迷宫--------------------------------------------------
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values 行是STATE 列是ACTION
        columns=actions,  # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table, character):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    # 扔炸弹次数没有限制
    if character == 'M':
        if (np.random.uniform() > EPSILON) or (
                (state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(ACTIONS_M)
        else:  # act greedy
            action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas

    else:
        if (np.random.uniform() > EPSILON) or (
        (state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(ACTIONS_B)
        else:  # act greedy
            action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    # 当扔炸弹次数有限制
    '''
    if count_boom == 0:
        if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(ACTIONS_M)
        else:  # act greedy
            action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    else:
        if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(ACTIONS_B)
        else:  # act greedy
            action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    '''
    return action_name


def setBoom(S_boom):
    boom_range_modified = []
    boom_state.append(S_boom)
    # print('S_boom')
    # print(S_boom)
    boom_range = np.asarray([S_boom, S_boom + 1, S_boom - 1, S_boom + WIDE, S_boom - WIDE])
    for range in boom_range:
        if range not in wall_state:
            boom_range_modified.append(range)  # 爆炸的范围确定，避开墙
    # print("boom_range_modified")
    # print(boom_range_modified)
    return boom_range_modified


def Boom_triger():
    pass


def get_env_feedback_boom(S_M, A_M, S_B, A_B, boom_range):  # B可能遇到的情况
    monster_dead = False
    bomberman_dead = False
    S_B_next = S_B
    S_M_next = S_M
    # if (S_B + 1 in boom_range) or (S_M - 1 in boom_range) or (S_M - WIDE in boom_range) or (S_M - WIDE in boom_range):
    if A_B == 'right':  # move right 向右走一步，分析导致//撞墙// 以及//能走// //遇到M//三种情况
        if (S_B + 1) in wall_state:  # 向右走撞墙，撞墙之后这一次就停止作为教训
            S_B_next = S_B
            R_B = -0.7
        if S_B + 1 in boom_range:  # 被炸到了
            # print("move right and will be booooooooooooooooooooomed")
            bomberman_dead = True
            S_B_next = S_B + 1
            R_B = -1

        if ((S_B + 1) not in wall_state) and (S_B + 1 not in boom_range):  # 能走，是通路
            S_B_next = S_B + 1
            R_B = -0.1
    elif A_B == 'left':  # move left
        if (S_B - 1) in wall_state:  # 在右走撞墙，撞墙之后这一次就停止作为教训
            S_B_next = S_B
            R_B = -0.7
        if S_B - 1 in boom_range:  # 遇到怪物，即下一步的选择方法和M的选择方法一致  //最可怕情况，挂掉
            # print("move left and will be booooooooooooooooooooomed")
            bomberman_dead = True
            S_B_next = S_B - 1
            R_B = -1

        if ((S_B - 1) not in wall_state) and (S_B - 1 not in boom_range):  # 能走，是通路
            S_B_next = S_B - 1
            R_B = -0.1
    elif A_B == 'up':  # move up
        if (S_B - WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
            S_B_next = S_B
            R_B = -0.7
        if (S_B - WIDE) in boom_range:  # terminate  //向左走一步就达到终点， 说明现在位置 end_state + 1
            # print("move up and will be booooooooooooooooooooomed")
            bomberman_dead = True
            S_B_next = S_B - WIDE
            R_B = -1

        if ((S_B - WIDE) not in wall_state) and ((S_B - WIDE) not in boom_range):  # 能走，是通路
            S_B_next = S_B - WIDE
            R_B = -0.1
    elif A_B == 'down':  # move up
        if (S_B + WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
            S_B_next = S_B
            R_B = -0.7
        if (S_B + WIDE) in boom_range:  # terminate  //向左走一步就达到终点， 说明现在位置 end_state + 1
            # print("move down and will be booooooooooooooooooooomed")
            bomberman_dead = True
            S_B_next = S_B + WIDE
            R_B = -1

        if ((S_B + WIDE) not in wall_state) and ((S_B + WIDE) not in boom_range):  # 能走，是通路
            S_B_next = S_B + WIDE
            R_B = -0.1

    # ------------------------------------------------   M   --------------------------------------------------

    if A_M == 'right':  # move right 向右走一步，分析导致//撞墙// 以及//能走// //遇到B//三种情况
        if S_M + 1 in boom_range:  # 被炸到了
            # print('boom the monster!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            monster_dead = True
            R_B = 1
            R_M = -0.2
        elif (S_M + 1) in wall_state:  # 在右走撞墙，撞墙之后这一次就停止作为教训
            S_M_next = S_M
            R_M = -0.7
        else:  # 能走，是通路
            S_M_next = S_M + 1
            R_M = -0.1
    elif A_M == 'left':  # move left
        if S_M - 1 in boom_range:  # 被炸到了
            # print('boom the monster!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            monster_dead = True
            R_B = 1
            R_M = -0.2
        elif (S_M - 1) in wall_state:  # 在右走撞墙，撞墙之后这一次就停止作为教训
            S_M_next = S_M
            R_M = -0.7
        else:  # 能走，是通路
            S_M_next = S_M - 1
            R_M = -0.1
    elif A_M == 'up':  # move up
        if (S_M - WIDE) in boom_range:  # 被炸到了
            # print('boom the monster!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            monster_dead = True
            R_B = 1
            R_M = -0.2
        elif (S_M - WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
            S_M_next = S_M
            R_M = -0.7
        else:  # 能走，是通路
            S_M_next = S_M - WIDE
            R_M = -0.1
    elif A_M == 'down':  # move up
        if (S_M - WIDE) in boom_range:  # 被炸到了
            # print('boom the monster!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            monster_dead = True
            R_B = 1
            R_M = -0.2
        elif (S_M + WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
            S_M_next = S_M
            R_M = -0.7
        else:  # 能走，是通路
            S_M_next = S_M + WIDE
            R_M = -0.1
    if monster_dead and bomberman_dead:
        R_B = -1
    if monster_dead and (not bomberman_dead):
        R_B = 1
    '''
    print("get_env_feedback_boom    data")
    print(S_M_next)
    print(R_M)
    print(S_B_next)
    print(R_B)
    print(monster_dead)
    print(bomberman_dead)
    '''


    return S_M_next, R_M, S_B_next, R_B, monster_dead, bomberman_dead


def get_env_feedback_B(S_B, A, S_M_next):  # the state bomber man might face
    # This is how agent will interact with the environment
    S_B_next = 0
    R_B = 0
    bomberman_dead = False
    # move right leading to 3 conditions://knock the wall// &//nothing// //meet monster//
    if A == 'right':
        if (S_B + 1) in wall_state:  # move to the left it will knock into the wall
            S_B_next = S_B
            R_B = -0.7
        if S_B == S_M_next - 1:  # 遇到怪物，即下一步的选择方法和M的选择方法一致  //最可怕情况，挂掉
            bomberman_dead = True
            S_B_next = S_M_next
            R_B = -1
        if ((S_B + 1) not in wall_state) and (S_B != S_M_next - 1):  # 能走，是通路
            S_B_next = S_B + 1
            R_B = -0.1
    elif A == 'left':  # move left
        if (S_B - 1) in wall_state:  # 在右走撞墙，撞墙之后这一次就停止作为教训
            S_B_next = S_B
            R_B = -0.7
        if S_B == S_M_next + 1:  # 遇到怪物，即下一步的选择方法和M的选择方法一致  //最可怕情况，挂掉
            bomberman_dead = True
            S_B_next = S_M_next
            R_B = -1
        if ((S_B - 1) not in wall_state) and (S_B != S_M_next + 1):  # 能走，是通路
            S_B_next = S_B - 1
            R_B = -0.1
    elif A == 'up':  # move up
        if (S_B - WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
            S_B_next = S_B
            R_B = -0.7
        if S_B == S_M_next + WIDE:  # terminate  //向左走一步就达到终点， 说明现在位置 end_state + 1
            bomberman_dead = True
            S_B_next = S_M_next
            R_B = -1

        if ((S_B - WIDE) not in wall_state) and (S_B != S_M_next + WIDE):  # 能走，是通路
            S_B_next = S_B - WIDE
            R_B = -0.1
    elif A == 'down':  # move up
        if (S_B + WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
            S_B_next = S_B
            R_B = -0.7
        if S_B == S_M_next - WIDE:  # terminate  //向左走一步就达到终点， 说明现在位置 end_state + 1
            bomberman_dead = True
            S_B_next = S_M_next
            R_B = -1

        if ((S_B + WIDE) not in wall_state) and (S_B != S_M_next - WIDE):  # 能走，是通路
            S_B_next = S_B + WIDE
            R_B = -0.1
    return S_B_next, R_B, bomberman_dead


def get_env_feedback_M(S_M, A):  # 怪物可能遇到的情况
    # This is how agent will interact with the environment
    if A == 'right':  # move right 向右走一步，分析导致//撞墙// 以及//能走// //遇到B//三种情况
        '''
        if S_M == current_state_B - 1:   # 
            S_M_next = current_state_B
            R_M = 0
            '''
        if (S_M + 1) in wall_state:  # 在右走撞墙，撞墙之后这一次就停止作为教训
            S_M_next = S_M
            R_M = -1
        else:  # 能走，是通路
            S_M_next = S_M + 1
            R_M = -0.1
    elif A == 'left':  # move left
        '''
        if S_M == current_state_B + 1:   # terminate  //向左走一步就达到终点， 说明现在位置 end_state + 1
            S_M_next = current_state_B
            R_M = 0
            '''
        if (S_M - 1) in wall_state:  # 在右走撞墙，撞墙之后这一次就停止作为教训
            S_M_next = S_M
            R_M = -1
        else:  # 能走，是通路
            S_M_next = S_M - 1
            R_M = -0.1
    elif A == 'up':  # move up
        '''
        if S_M == current_state_B + WIDE:  # terminate  //向左走一步就达到终点， 说明现在位置 end_state + 1
            S_M_next = current_state_B
            R_M = 0
            '''
        if (S_M - WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
            S_M_next = S_M
            R_M = -1
        else:  # 能走，是通路
            S_M_next = S_M - WIDE
            R_M = -0.1
    elif A == 'down':  # move up
        '''
        if S_M == current_state_B - WIDE:  # terminate  //向左走一步就达到终点， 说明现在位置 end_state + 1
            S_M_next = current_state_B
            R_M = 0
            '''
        if (S_M + WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
            S_M_next = S_M
            R_M = -1
        else:  # 能走，是通路
            S_M_next = S_M + WIDE
            R_M = -0.1

    return S_M_next, R_M


def move(S_M, S_B, episode, step_counter, is_terminated, boom_range, is_setBoom, is_boom):
    # This is how environment be updated
    env_list = maze.copy()
    if is_terminated:
        env_list[S_M // WIDE][S_M % WIDE] = 'm'
        env_list[S_B // WIDE][S_B % WIDE] = 'b'
        if is_boom:
            for range in boom_range:
                env_list[range // WIDE][range % WIDE] = '*'
            if S_B in boom_range:
                env_list[S_B // WIDE][S_B % WIDE] = '#'
                print('you are boomed')
            if S_M in boom_range:
                env_list[S_M // WIDE][S_M % WIDE] = '&'
                print("monster boomed")
        else:
            env_list[S_M // WIDE][S_M % WIDE] = '@'
            print("you are eaten!!!!!!")
        interaction_front = env_list
        print('\r{}'.format(interaction_front), end='')
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        print('\n')
        time.sleep(2)

    else:
        # print('S_B in move ')
        # print(S_B)
        env_list[S_M // WIDE][S_M % WIDE] = 'm'
        env_list[S_B // WIDE][S_B % WIDE] = 'b'
        if is_setBoom:
            place_boom = boom_range[0]
            env_list[place_boom // WIDE][place_boom % WIDE] = 'o'
        if is_boom:
            for range in boom_range:
                env_list[range // WIDE][range % WIDE] = '*'
                #print("boom! you all survived!")
        interaction = env_list
        # print('S!!!!')
        # print(S)
        os.system("cls")
       #print('------------------------------maze looks--------------------------')
        print('\r{}'.format(interaction), end='')
        print('\n')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table_M = build_q_table(N_STATES, ACTIONS_M)
    q_table_B = build_q_table(N_STATES, ACTIONS_B)  # action_B里面包括了扔炸弹，算作一个动作并不在choose函数的时候特殊处理，重点要考虑的是一旦扔炸弹之后的问题

    for episode in range(MAX_EPISODES):
        step_counter = 1
        S_M = start_state_M
        S_B = start_state_B
        is_terminated = False
        is_boom = False
        is_setBoom = False
        boom_range = []
        boom_loading = 0
        monster_dead = False
        bomber_dead = False
        move(S_M, S_B, episode, step_counter, is_terminated, boom_range, is_setBoom, is_boom)
        while not is_terminated:
            is_boom = False
            if step_counter % 2 == 1:
                A_M = choose_action(S_M, q_table_M, 'M')  # 获取M的动作

                A_B = choose_action(S_B, q_table_B, 'B')  # 获取B的动作
                if A_B == 'setBoom':
                    A_B = choose_action(S_B, q_table_M, 'M')
                    is_setBoom = True
                    boom_range = setBoom(S_B)
                if boom_loading == BOOM_WAIT:  # 让你先跑3回，下一回的动作就决定奖罚了

                    S_M_next, R_M_out, S_B_next, R_B_out, monster_dead, bomber_dead = get_env_feedback_boom(S_M, A_M,
                                                                                                            S_B, A_B,
                                                                                                            boom_range)

                else:
                    S_M_next, R_M_out = get_env_feedback_M(S_M, A_M)  # take action & get next state and reward
                    S_B_next, R_B_out, bomber_dead = get_env_feedback_B(S_B, A_B, S_M_next)  # take action & get next state and reward
                    '''print("B动作")
                    print(A_B)
                    print("B下一步")
                    print(S_B_next)

                    print('R_B_out(when eaten)')
                    print(R_B_out) '''

                if is_setBoom:
                    boom_loading += 1

                q_predict_M = q_table_M.loc[S_M, A_M]
                q_predict_B = q_table_B.loc[S_B, A_B]
                if bomber_dead or monster_dead:
                    #print(S_M_next, bomber_dead, monster_dead)
                    q_target_M = R_M_out  # next state is terminal
                    q_target_B = R_B_out  # next state is terminal
                    is_terminated = True  # terminate this episode
                    #print('is_terminated 1')
                    #print(is_terminated)

                    if bomber_dead:
                        print("bomber_dead")
                    if monster_dead:
                        print("monster_dead")
                else:
                    q_target_M = R_M_out + GAMMA * q_table_M.iloc[S_M_next, :].max()  # next state is not terminal
                    q_target_B = R_B_out + GAMMA * q_table_B.iloc[S_B_next, :].max()  # next state is not terminal

                q_table_M.loc[S_M, A_M] += ALPHA * (q_target_M - q_predict_M)  # update
                q_table_B.loc[S_B, A_B] += ALPHA * (q_target_B - q_predict_B)  # update
                S_M = S_M_next  # move to next state
                record_S_M.append(S_M)
                S_B = S_B_next  # move to next state

                if boom_loading == BOOM_WAIT + 1:
                    is_boom = True
                    if (S_B in boom_range) or (S_M in boom_range):
                        is_terminated = True
                        #print('is_terminated 2')
                        #print(is_terminated)
                    move(S_M, S_B, episode, step_counter + 1, is_terminated, boom_range, is_setBoom, is_boom)
                    boom_loading = 0
                    is_setBoom = False
                    is_boom = False
                    boom_range = []
                else:
                    move(S_M, S_B, episode, step_counter + 1, is_terminated, boom_range, is_setBoom, is_boom)
                if is_terminated:
                    if monster_dead and (not bomber_dead):
                        ''' print("R_B_out!")
                        print(R_B_out)
                        print(S_B) '''
                        print('\n')
                        print('----------------CONGRATULATIONS!!!!!! you win!!!!!!-----------------')
                        print('\n')
                    elif bomber_dead:
                        print('\n')
                        print('----------------------what a pity!!!!!!!!!!!!!!!!!!--------------------')
                        '''print("A")
                        print("R_B_out!")
                        print(R_B_out)
                        print(S_B)'''
                        print('\n')

                step_counter += 1
                continue

            # --------------------------------------------------------------------------------------------------------------------------------

            else:  # M 不动

                A_B = choose_action(S_B, q_table_B, 'B')  # 获取B的动作
                if A_B == 'setBoom':
                    A_B = choose_action(S_B, q_table_M, 'M')

                    is_setBoom = True
                    boom_range = setBoom(S_B)
                if boom_loading == BOOM_WAIT:
                    S_M_next, R_M_out, S_B_next, R_B_out, monster_dead, bomber_dead = get_env_feedback_boom(S_M, A_M,
                                                                                                            S_B, A_B,
                                                                                                            boom_range)
                else:
                    S_M_next, R_M_out = get_env_feedback_M(S_M, A_M)  # take action & get next state and reward
                    #print("M下一步")
                    #print(S_M_next)
                    S_B_next, R_B_out, bomber_dead = get_env_feedback_B(S_B, A_B,S_M_next)  # take action & get next state and reward
                    '''
                    print("B动作")
                    print(A_B)
                    print("B下一步")
                    print(S_B_next)
                    print('R_B_out(when eaten)')
                    print(R_B_out)'''

                if is_setBoom:
                    boom_loading += 1
                S_M_next = record_S_M[-1]
                q_predict_B = q_table_B.loc[S_B, A_B]
                if bomber_dead or monster_dead:
                    q_target_B = R_B_out  # next state is terminal
                    is_terminated = True  # terminate this episode
                    print('is_terminated 3')
                    print(is_terminated)

                    if bomber_dead:
                        print("bomber_dead")
                    if monster_dead:
                        print("monster_dead")
                else:
                    q_target_B = R_B_out + GAMMA * q_table_B.iloc[S_B_next, :].max()  # next state is not terminal

                q_table_B.loc[S_B, A_B] += ALPHA * (q_target_B - q_predict_B)  # update
                S_M = S_M_next
                S_B = S_B_next  # move to next state

                if boom_loading == BOOM_WAIT + 1:
                    is_boom = True
                    if (S_B in boom_range) or (S_M in boom_range):
                        is_terminated = True
                        #print('is_terminated 4')
                        #print(is_terminated)
                    move(S_M, S_B, episode, step_counter + 1, is_terminated, boom_range, is_setBoom, is_boom)
                    boom_loading = 0
                    is_setBoom = False
                    is_boom = False
                    boom_range = []
                else:
                    move(S_M, S_B, episode, step_counter + 1, is_terminated, boom_range, is_setBoom, is_boom)

                # move(S_M, S_B, episode, step_counter + 1, is_terminated, boom_range, is_setBoom, is_boom)
                if is_terminated:
                    if monster_dead and (not bomber_dead):
                        print('\n')
                        print('----------------CONGRATULATIONS!!!!!! you win!!!!!!-----------------')
                        print('\n')
                        '''print("R_B_out!")
                        print(R_B_out)
                        print(S_B)'''

                    if bomber_dead:
                        print('\n')
                        print('----------------------what a pity!!!!!!!!!!!!!!!!!!--------------------')
                        print('\n')
                        '''print("R_B_out!")
                        print(R_B_out)
                        print(S_B)'''


                step_counter += 1

    return q_table_B, q_table_M


'''
        while not is_terminated:

            A = choose_action(S_M, q_table)
            #print('action! %s' %(A))
            S_M_next, R_M_out = get_env_feedback_M(S_M, A)  # take action & get next state and reward
            q_predict = q_table.loc[S_M, A]
            if S_M_next != current_state_B:
                q_target = R_M_out + GAMMA * q_table.iloc[S_M_next, :].max()   # next state is not terminal
            else:
                q_target = R_M_out     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S_M, A] += ALPHA * (q_target - q_predict)  # update
            S_M = S_M_next  # move to next state

            update_env(S_M, episode, step_counter+1)
            step_counter += 1

    return q_table
'''

if __name__ == "__main__":
    q_table_B, q_table_M = rl()
    print('\r\nQ-table_B:\n')
    print(q_table_B)
    print('\r\nQ-table_M:\n')
    print(q_table_M)
