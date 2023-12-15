import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
import random
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.activ_func = activation
        self.weights_initialization = 3e-3
        layers = []

        layers.append(nn.Linear(self.input_dim, self.hidden_size))
        layers.append(nn.ReLU())

        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
                        
        linear_output = nn.Linear(self.hidden_size, self.output_dim)
        linear_output.weight.data.uniform_(-self.weights_initialization, self.weights_initialization)
        linear_output.bias.data.uniform_(-self.weights_initialization, self.weights_initialization)
        layers.append(linear_output)

        self.model = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        return self.model(s)
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.reparam_noise = 1e-6
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        env_test = get_env(g=10.0, train=False)
        self.action_space = env_test.action_space
        self.nn_actor = NeuralNetwork(self.state_dim, self.action_dim * 2, hidden_size=self.hidden_size,
                                            hidden_layers=self.hidden_layers, activation='relu').to(self.device)
        self.opt_actor = optim.Adam(self.nn_actor.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    
    

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        actor_output = self.nn_actor(state)
        action_means = actor_output[:, 0]
        action_log_stds = actor_output[:, 1]
        clamped_log_stds = self.clamp_log_std(action_log_stds)
        action_stds = torch.exp(clamped_log_stds)

        epsilon_dist = Normal(0, 1)
        epsilons = epsilon_dist.sample(action_means.unsqueeze(1).shape).to(self.device)

        action_def = action_means.unsqueeze(1) + action_stds.unsqueeze(1) * epsilons
        action = torch.tanh(action_def)

        prob_dist = Normal(action_means.unsqueeze(1), action_stds.unsqueeze(1))
        log_prob = prob_dist.log_prob(action_def) - torch.log(1 - action.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob
        

class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.nn_critic = NeuralNetwork(self.state_dim + self.action_dim, 2, hidden_size=self.hidden_size,
                                            hidden_layers=self.hidden_layers, activation='relu').to(self.device)
        self.opt_critic = optim.Adam(self.nn_critic.parameters(), lr=self.critic_lr)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 256
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        self.reward_scale_par = 12
        self.gamma_par = 0.98#0.98
        self.alpha_par = 0.25#0.25
        self.polyak_par = 0.015#0.015
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        self.nn_policy = Actor(hidden_size=512, hidden_layers=2, actor_lr=0.003, state_dim = self.state_dim, action_dim = self.action_dim, device = self.device) 
        self.nn1_q_soft = Critic(hidden_size=512, hidden_layers=2, critic_lr=0.003, state_dim = self.state_dim, action_dim = self.action_dim, device = self.device) 
        self.nn2_q_soft = Critic(hidden_size=512, hidden_layers=2, critic_lr=0.003, state_dim = self.state_dim, action_dim = self.action_dim, device = self.device) 
        self.nn_value = NeuralNetwork(self.state_dim, self.action_dim, hidden_size=512,
                                            hidden_layers=2, activation='relu').to(self.device)
        self.nn_best_value = NeuralNetwork(self.state_dim, self.action_dim, hidden_size=512,
                                            hidden_layers=2, activation='relu').to(self.device)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
    :param s: np.ndarray, state of the pendulum. shape (3, )
    :param train: boolean to indicate if you are in eval or train mode. 
                You can find it useful if you want to sample from deterministic policy.
    :return: np.ndarray, action to apply on the environment, shape (1,)
    """
        
        tensor_state = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        policy_output = self.nn_policy.nn_actor(tensor_state)
        action_mean = policy_output[:, 0]
        action_std = torch.exp(torch.clamp(policy_output[:, 1], min=-20, max=2))

        noise_dist = Normal(torch.zeros_like(action_mean), torch.ones_like(action_mean))
        noise_sample = noise_dist.sample().to(self.device)
        computed_action = torch.tanh(action_mean + action_std * noise_sample)

        action = computed_action.detach().cpu().numpy()

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray'
        return action[0]
    
    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        # This part of the code made use of the following github repo: 
        # https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/sac_pendulum.py

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        
        loss_function_value = loss_function_q1 = loss_function_q2 = nn.MSELoss()

        lr_value = 3e-4
        lr_q = 3e-4
        lr_policy = 3e-4

        opt_value = optim.Adam(self.nn_value.parameters(), lr=lr_value)
        opt_q1_soft = optim.Adam(self.nn1_q_soft.nn_critic.parameters(), lr=lr_q)
        opt_q2_soft = optim.Adam(self.nn2_q_soft.nn_critic.parameters(), lr=lr_q)
        opt_actor = optim.Adam(self.nn_policy.nn_actor.parameters(), lr=lr_policy)

        q1_estimated = self.nn1_q_soft.nn_critic(torch.cat([s_batch, a_batch], dim=1))
        q2_estimated = self.nn2_q_soft.nn_critic(torch.cat([s_batch, a_batch], dim=1))
        value_estimated = self.nn_value(s_batch)
        action_new, log_prob_action = self.nn_policy.get_action_and_log_prob(s_batch, deterministic=False)

        value_target = self.nn_best_value(s_prime_batch)
        q_target = self.gamma_par * value_target + self.reward_scale_par * r_batch
        loss_q1 = loss_function_q1(q1_estimated, q_target.detach())
        loss_q2 = loss_function_q2(q2_estimated, q_target.detach())

        opt_q1_soft.zero_grad()
        loss_q1.backward()
        opt_q1_soft.step()

        opt_q2_soft.zero_grad()
        loss_q2.backward()
        opt_q2_soft.step()
        
        q_min = torch.min(
            self.nn1_q_soft.nn_critic(torch.cat([s_batch, action_new], dim=1)),
            self.nn2_q_soft.nn_critic(torch.cat([s_batch, action_new], dim=1))
        )
        scaled_act = self.alpha_par * log_prob_action
        value_target_function = q_min - scaled_act
        loss_value = loss_function_value(value_estimated, value_target_function.detach())

        opt_value.zero_grad()
        loss_value.backward()
        opt_value.step()

        loss_policy = (scaled_act - q_min).mean()

        opt_actor.zero_grad()
        loss_policy.backward()
        opt_actor.step()

        for param_target, param in zip(self.nn_best_value.parameters(), self.nn_value.parameters()):
            param_target.data.copy_(
                (self.polyak_par * param.data) + ((1.0 - self.polyak_par) * param_target.data)
            )


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 100
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()