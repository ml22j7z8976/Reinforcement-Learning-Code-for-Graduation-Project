import torch
from pathlib import Path
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym.wrappers import FrameStack
from model.agent import Mario
from utils.pro_img import SkipFrame, GrayScaleObservation, ResizeObservation
import time

if __name__ == "__main__":
    # 初始化 Super Mario 环境
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    # 限制动作空间为：向右走和向右跳
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env.reset()

    # 应用环境包装器
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    # 实例化 Mario
    save_dir = Path("checkpoints")  # 假设模型保存在这个目录下
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    # 加载训练好的模型
    #checkpoint_path = "checkpoints/path_to_saved_model.chkpt" # 替换为实际的模型路径
    checkpoint_path = "checkpoints/2024-07-21T19-19-05/mario_net_1.pt"
  

    if Path(checkpoint_path).is_file():
        checkpoint = torch.load(checkpoint_path)
        mario.net.load_state_dict(checkpoint['model'])
        mario.exploration_rate = checkpoint['exploration_rate']
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"No model found at {checkpoint_path}")

    # 使用加载的模型进行游戏呈现
    state = env.reset()
    while True:
        action = mario.act(state)
        time.sleep(0.01)
        env.render()
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            state = env.reset()
