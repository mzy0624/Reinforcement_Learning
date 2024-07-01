from ValueBased import Q_Learning, SARSA, NaiveDQN, DQN
from Env.FrozenLakeEnv import FrozenLakeEnv
def main(alg='Q_Learning'):
    env = FrozenLakeEnv()
    # agent = Q_Learning(env)
    # agent = SARSA(env)
    # agent = naiveDQN(env)
    agent = DQN(env, max_epsilon=0.3)
    agent.train()
    input('trained')
    agent.play()
    input()

if __name__ == '__main__':
    main()