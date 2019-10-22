import numpy as np

class env:
    maze = np.asarray([["X", "X", "X", "X", "X", "X"],
                           ["X", "M", " ", " ", " ", "X"],
                           ["X", " ", " ", " ", " ", "X"],
                           ["X", " ", "B", " ", " ", "X"],
                           ["X", " ", " ", " ", " ", "X"],
                           ["X", "X", "X", "X", "X", "X"]])  # 迷宫的样子
    index = np.arange(36).reshape(6, 6)
    wall_state = []  # 初始化所有墙的集合
    wall_sign = np.asarray(["X"])
    for wall in (index[maze == wall_sign]):
        wall_state.append(wall)

    print(wall_state)  # 所有的墙





