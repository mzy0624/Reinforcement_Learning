from ValueBased import Q_Learning, SARSA, NaiveDQN, DQN
from PolicyBased import Reinforce, ActorCritic
from Env import FrozenLakeEnv, CartPoleV1
def main(alg='Q_Learning'):
    env = CartPoleV1()
    env = FrozenLakeEnv()
    # agent = Q_Learning(env)
    # agent = SARSA(env)
    # agent = NaiveDQN(env)
    agent = DQN(env)
    # agent = Reinforce(env)
    # agent = ActorCritic(env)
    agent.train()
    input('trained')
    agent.play()
    input()

if __name__ == '__main__':
    main()