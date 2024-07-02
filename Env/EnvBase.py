class EnvBase:
    def __init__(self, env, state_type, action_type, state_dim, action_dim):
        self.env = env
        self.state_type   = state_type              # Discrete / Continuous
        self.action_type  = action_type             # Discrete / Continuous
        self.state_dim    = state_dim               # The dimension of state
        self.action_dim   = action_dim              # The dimension of action
        self.state_space  = env.observation_space   # State space
        self.action_space = env.action_space        # Action space
    
    def one_hot_encode(self, x, N):
        one_hot_vec = [0.] * N
        one_hot_vec[x] = 1
        return one_hot_vec

    def sample_action(self):
        return self.action_space.sample()