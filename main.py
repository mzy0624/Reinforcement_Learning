from ValueBased import Q_Learning, SARSA, NaiveDQN, DQN
from PolicyBased import Reinforce, ActorCritic
from Env import FrozenLakeEnv, CartPoleV1
from fire import Fire
env_list = [
    FrozenLakeEnv,
    CartPoleV1
]
alg_list = [
    Q_Learning,
    SARSA,
    NaiveDQN,
    DQN,
    Reinforce,
    ActorCritic
]
def main(alg=-1, env=0):
    Env = env_list[env]
    Alg = alg_list[alg]
    env = Env()
    agent = Alg(env)
    print(f'{Env.__name__ = }, {Alg.__name__ = }')
    agent.train()
    input('trained')
    agent.play()
    input('played')

if __name__ == '__main__':
    Fire(main)