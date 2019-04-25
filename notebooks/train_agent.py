import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
from tqdm import tqdm

from gridworlds.nn.phinet import PhiNet
from gridworlds.domain.gridworld.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld
from gridworlds.agents.randomagent import RandomAgent
from gridworlds.agents.dqnagent import DQNAgent
from gridworlds.utils import reset_seeds, get_parser
from gridworlds.sensors import *


parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
parser.add_argument('-a','--agent', type=str, required=True,
                    choices=['random','dqn'], help='Type of agent to train')
parser.add_argument('-n','--n_trials', type=int, default=1,
                    help='Number of trials')
parser.add_argument('-e','--n_episodes', type=int, default=10,
                    help='Number of episodes per trial')
parser.add_argument('-m','--max_steps', type=int, default=100,
                    help='Maximum number of steps per episode')
parser.add_argument('-r','--rows', type=int, default=7,
                    help='Number of gridworld rows')
parser.add_argument('-c','--cols', type=int, default=4,
                    help='Number of gridworld columns')
parser.add_argument('-b','--batch_size', type=int, default=32,
                    help='Number of experiences to sample per batch')
parser.add_argument('-lr','--learning_rate', type=float, default=0.0001,
                    help='Learning rate for Adam optimizer')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('-v','--video', action='store_true',
                    help='Show video of agent training')
parser.set_defaults(save=False)
parser.set_defaults(video=False)
args = parser.parse_args()

if args.video:
    import matplotlib.pyplot as plt

log_dir = 'logs/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)
log = open(log_dir+'/scores-{}-{}.txt'.format(args.agent, args.seed), 'w')

reset_seeds(args.seed)

#%% ------------------ Define MDP ------------------
env = GridWorld(rows=args.rows, cols=args.cols)
gamma = 0.9
sensor = SensorChain([
    OffsetSensor(offset=(0.5,0.5)),
    NoisySensor(sigma=0.05),
    ImageSensor(range=((0,env._rows), (0,env._cols)), pixel_density=3),
    # ResampleSensor(scale=2.0),
    BlurSensor(sigma=0.6, truncate=1.),
])
x0 = sensor.observe(env.get_state())

#%% ------------------ Load abstraction ------------------
n_latent_dims = 2
# modelfile = 'models/{}/phi-{}.pytorch'.format(args.tag, args.seed)
# phinet = PhiNet(input_shape=x0.shape, n_latent_dims=n_latent_dims, n_hidden_layers=1, n_units_per_layer=32, lr=args.learning_rate)
# phinet.load(modelfile)

class NullAbstraction:
    def __call__(self, x):
        return x
    def freeze(self):
        pass
sensor = SensorChain([])
phinet = NullAbstraction()

#%% ------------------ Load agent ------------------
n_actions = 4
if args.agent == 'random':
    agent = RandomAgent(n_actions=n_actions)
elif args.agent == 'dqn':
    agent = DQNAgent(n_latent_dims=n_latent_dims, n_actions=n_actions, phi=phinet, lr=args.learning_rate, batch_size=args.batch_size)
else:
    assert False, 'Invalid agent type: {}'.format(args.agent)

#%% ------------------ Train agent ------------------
if args.video:
    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    fig.show()

    def plot_value_function(ax):
        s = np.asarray([[np.asarray([x,y]) for x in range(args.cols)] for y in range(args.rows)])
        v = np.asarray(agent.q_values(s).detach().numpy()).max(-1)
        xy = OffsetSensor(offset=(0.5,0.5)).observe(s).reshape(args.cols, args.rows, -1)
        ax.contourf(np.arange(0.5,args.cols+0.5), np.arange(0.5,args.rows+0.5), v, vmin=-10, vmax=0)

    def plot_states(ax):
        data = pd.DataFrame(agent.replay.memory)
        data[['x.r','x.c']] = pd.DataFrame(data['x'].tolist(), index=data.index)
        data[['xp.r','xp.c']] = pd.DataFrame(data['xp'].tolist(), index=data.index)
        sns.scatterplot(data=data, x='x.c',y='x.r', hue='done', style='done', markers=True, size='done', size_order=[1,0], ax=ax, alpha=0.3, legend=False)
        ax.invert_yaxis()

for trial in tqdm(range(args.n_trials), desc='trials'):
    env.reset_goal()
    agent.reset()
    total_reward = 0
    total_steps = 0
    losses = []
    rewards = []
    q_values = []
    for episode in tqdm(range(args.n_episodes), desc='episodes'):
        env.reset_agent()
        ep_rewards = []
        for step in range(args.max_steps):
            s = env.get_state()
            x = sensor.observe(s)

            a = agent.act(x)
            sp, r, done = env.step(a)
            xp = sensor.observe(sp)
            ep_rewards.append(r)
            if args.video:
                q_values.append(agent.q_values(x).detach().numpy().max())
            total_reward += r

            loss = agent.train(x, a, r, xp, done, gamma)
            losses.append(loss)
            rewards.append(r)

            if done:
                break

        if args.video:
            [a.clear() for a in ax]
            plot_value_function(ax[0])
            env.plot(ax[0])
            ax[1].plot(q_values)
            ax[2].plot(rewards, c='C3')
            ax[3].plot(losses, c='C1')
            # plot_states(ax[3])
            ax[1].set_ylim([-10,0])
            fig.canvas.draw()
            fig.canvas.flush_events()

        total_steps += step
        score_info = {
            'trial': trial,
            'episode': episode,
            'reward': sum(ep_rewards),
            'total_reward': total_reward,
            'total_steps': total_steps,
            'steps': step
        }
        json_str = json.dumps(score_info)
        log.write(json_str+'\n')
        log.flush()
print('\n\n')
