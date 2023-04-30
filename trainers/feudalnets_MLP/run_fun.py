import argparse
import numpy as np

import torch

from .utils import take_action
from .feudalnet import FeudalNetwork, feudal_loss
from .storage import Storage
from .logger import Logger


parser = argparse.ArgumentParser(description='Feudal Nets')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--episodes',type=int,default=100000,help='episodes')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='BreakoutNoFrameskip-v0',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default= 1,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=50,  # TODO: used to be 400!
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e8),  # TODO: this used to be 1e8!!
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
parser.add_argument('--grad-clip', type=float, default=5.,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='Entropy coefficient to encourage exploration.')
parser.add_argument('--mlp', type=int, default=1,
                    help='toggle to feedforward ML architecture')

# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--time-horizon', type=int, default=10,
                    help='Manager horizon (c)')
parser.add_argument('--MLP_dim', type=int, default=64,
                    help='hidden dimension for MLP')
parser.add_argument('--hidden-dim-manager', type=int, default=64,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=16,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.99,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.999,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=int(1e-5),
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation', type=int, default=5,
                    help='Dilation parameter for manager LSTM.')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='baseline',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()


def experiment(args, env, Episodes):
    # run_name = args.run_name
    save_steps = list(torch.arange(0, int(args.max_steps), int(args.max_steps) // 10).numpy())

    logger = Logger(args.run_name, args)
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # envs = make_envs(args.env_name, args.num_workers, args.seed)
    feudalnet = FeudalNetwork(
        num_workers=args.num_workers,
        input_dim=env.observation_space.shape,
        # hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        dim=args.MLP_dim,
        n_actions=env.action_space.n,
        time_horizon=args.time_horizon,
        # dilation=args.dilation,
        device=device,
        mlp=args.mlp,
        args=args)

    optimizer = torch.optim.RMSprop(feudalnet.parameters(), lr=args.lr,
                                    alpha=0.99, eps=1e-5)
    goals, states, masks = feudalnet.init_obj()  #[torch.zeros_like(template).to(self.device) for _ in range(2*self.c+1)]
        # states = [torch.zeros_like(template).to(self.device) for _ in range(2*self.c+1)]
        # masks = [torch.ones(self.b, 1).to(self.device) for _ in range(2*self.c+1)]

    for i in range(Episodes):
        args.run_name = f"EP{i}"
        x = env.reset()
        done = False
        step = 0
        action_d = 0
        while not done:
            count = 0
            # Detaching LSTMs and goals
            # feudalnet.repackage_hidden()
            goals = [g.detach() for g in goals]
            storage = Storage(size=args.num_steps,
                              keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                    's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                    'adv_m', 'adv_w'])

            for i in range(args.num_steps):
                if not done:
                    action_dist, goals, states, value_m, value_w = feudalnet(x, goals, states, masks[-1])
                    # Take a step, log the info, get the next state
                    action_d = action_dist
                    action, logp, entropy = take_action(action_dist)
                    x, reward, done, info = env.step(action[-1])
                    # logger.log_episode(info, step)
                # if done and count==0:
                #     count+=1
                #     print(f"done at num_step{i}/{args.num_steps} and step{step}/{args.max_steps}")
                #     # break
                # print(f"step{i}")
                mask = torch.FloatTensor(1 - np.array(done).reshape(1,)).unsqueeze(-1).to(args.device)
                masks.pop(0)
                masks.append(mask)

                storage.add({
                    'r': torch.FloatTensor(np.array(reward).reshape(1,)).unsqueeze(-1).to(device),
                    'r_i': feudalnet.intrinsic_reward(states, goals, masks),
                    'v_w': value_w,
                    'v_m': value_m,
                    'logp': logp.unsqueeze(-1),
                    'entropy': entropy.unsqueeze(-1),
                    's_goal_cos': feudalnet.state_goal_cosine(states, goals, masks),
                    'm': mask
                })

                step += args.num_workers


            with torch.no_grad():
                *_, next_v_m, next_v_w = feudalnet(x, goals, states, mask, save=False)
                next_v_m = next_v_m.detach()
                next_v_w = next_v_w.detach()

            optimizer.zero_grad()
            loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feudalnet.parameters(), args.grad_clip)
            optimizer.step()

            # logger.log_scalars(loss_dict, step)

            if len(save_steps) > 0 and step > save_steps[0]:
                torch.save({
                    'model': feudalnet.state_dict(),
                    'args': args,
                    'processor_mean': feudalnet.preprocessor.rms.mean,
                    'optim': optimizer.state_dict()},
                    f'models/{args.env_name}_{args.run_name}_step={step}.pt')
                save_steps.pop(0)

        # torch.save({
        #     'model': feudalnet.state_dict(),
        #     'args': args,
        #     'processor_mean': feudalnet.preprocessor.rms.mean,
        #     'optim': optimizer.state_dict()},
        #     f'models/{args.env_name}_{args.run_name}_steps={step}.pt')

        # print(f"action dist is {action_d}")


def main(args, env):
    # args.seed = seed
    experiment(args, env, args.episodes)


# if __name__ == '__main__':
#     main(args)
