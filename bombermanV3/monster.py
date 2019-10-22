class monster:
    q_predict_M = 0
    A_M = 'none'
    S_M_next = 0
    R_M_out = 0
    q_target_M = 0
    monster_dead = False
    record_S_M = []
    def __init__(self, S_M, q_table_M, wall_state,WIDE,HEIGHT):
        self.S_M = S_M
        self.q_table_M = q_table_M
        self.wall_state = wall_state
        self.WIDE = WIDE
        self.HEIGHT = HEIGHT

    def get_env_feedback_M(self,S_M, A,wall_state,WIDE,HEIGHT):  # 怪物可能遇到的情况
        # This is how agent will interact with the environment
        if A == 'right':  # move right 向右走一步，分析导致//撞墙// 以及//能走// //遇到B//三种情况

            if (S_M + 1) in wall_state:  # 在右走撞墙，撞墙之后这一次就停止作为教训
                S_M_next = S_M
                R_M = -1
            else:  # 能走，是通路
                S_M_next = S_M + 1
                R_M = -0.1
        elif A == 'left':  # move left

            if (S_M - 1) in wall_state:  # 在右走撞墙，撞墙之后这一次就停止作为教训
                S_M_next = S_M
                R_M = -1
            else:  # 能走，是通路
                S_M_next = S_M - 1
                R_M = -0.1
        elif A == 'up':  # move up

            if (S_M - WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
                S_M_next = S_M
                R_M = -1
            else:  # 能走，是通路
                S_M_next = S_M - WIDE
                R_M = -0.1
        elif A == 'down':  # move up

            if (S_M + WIDE) in wall_state:  # 向上走撞墙，撞墙之后这一次就停止作为教训
                S_M_next = S_M
                R_M = -1
            else:  # 能走，是通路
                S_M_next = S_M + WIDE
                R_M = -0.1

        return S_M_next, R_M