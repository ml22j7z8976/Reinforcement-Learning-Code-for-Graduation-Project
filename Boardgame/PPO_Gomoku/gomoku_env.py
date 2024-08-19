import gym
from gym import spaces
import numpy as np

class GomokuEnv(gym.Env):
    def __init__(self):
        super(GomokuEnv, self).__init__()
        self.size = 8  # 8x8 board
        self.action_space = spaces.Discrete(self.size * self.size)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=np.int32)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.done = False
        self.current_player = 1
        return self.board

    def step(self, action):
        if self.done:
            return self.board, 0, self.done, {}

        x, y = divmod(action, self.size)
        
        self.board[x, y] = self.current_player
        if self.check_win(x, y):
            self.done = True
            return self.board, 1, self.done, {}
        
        if self.check_win(x, y):
            self.done = True
            # 给获胜者 1 分，给失败者 -1 分
            reward = 1 if self.current_player == 1 else -1
            return self.board, reward, self.done, {}
        
        if np.all(self.board != 0):
            self.done = True
            return self.board, 0, self.done, {}  # Draw

        self.current_player = 3 - self.current_player  # Switch player
        return self.board, 0, self.done, {}

    def check_win(self, x, y):
        player = self.board[x, y]
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for d in [1, -1]:
                nx, ny = x, y
                while True:
                    nx += d * dx
                    ny += d * dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                        count += 1
                    else:
                        break
            if count >= 5:
                return True
        return False

    def render(self, mode='human'):
        print(self.board)

# Example usage
if __name__ == "__main__":
    env = GomokuEnv()
    env.reset()
    env.render()
