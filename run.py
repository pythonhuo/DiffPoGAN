import argparse
import gym
import numpy as np
import os
import torch
import json

import time
from coolname import generate_slug
import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter
from agents.diffpogan import DiffpoGAN as Agent

hyperparameters = {
    "halfcheetah-medium-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "freq": 2,
        "lambda_min": 0,
        "target_kl": 0.06,
        "gn": 9.0,
    },
    "hopper-medium-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.05,
        "gn": 9.0,
        "freq": 2,
    },
    "walker2d-medium-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.03,
        "gn": 1.0,
        "freq": 2,
    },
    "halfcheetah-medium-replay-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.06,
        "gn": 2.0,
        "freq": 2,
    },
    "hopper-medium-replay-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.03,
        "gn": 4.0,
        "freq": 2,
    },
    "walker2d-medium-replay-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.03,
        "gn": 4.0,
        "freq": 2,
    },
    "halfcheetah-medium-expert-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.04,
        "gn": 7.0,
        "freq": 2,
    },
    "hopper-medium-expert-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.03,
        "gn": 5.0,
        "freq": 2,
    },
    "walker2d-medium-expert-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.04,
        "gn": 5.0,
        "freq": 2,
    },
    "antmaze-umaze-v0": {
        "lr": 3e-4,
        "lambda": 3,
        "max_q_backup": False,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 2.0,
        "freq": 2,
    },
    "antmaze-umaze-diverse-v0": {
        "lr": 3e-4,
        "lambda": 3,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.09,
        "gn": 3.0,
        "freq": 2,
    },

    "maze2d-umaze": {
        "lr": 3e-4,
        "lambda": 3,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.09,
        "gn": 3.0,
        "freq": 2,
    },
        "maze2d-medium": {
        "lr": 3e-4,
        "lambda": 1,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.28,
        "target_kl": 0.2,
        "gn": 2.0,
        "freq": 2,
    },
            "maze2d-large": {
        "lr": 3e-4,
        "lambda": 1,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.28,
        "target_kl": 0.2,
        "gn": 2.0,
        "freq": 2,
    },
    "antmaze-medium-play-v0": {
        "lr": 1e-3,
        "lambda": 1,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.28,
        "target_kl": 0.2,
        "gn": 2.0,
        "freq": 2,
    },
    "antmaze-medium-diverse-v0": {
        "lr": 3e-4,
        "lambda": 1,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.25,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
    "antmaze-large-play-v0": {
        "lr": 1e-4,
        "lambda": 0.5,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.18,
        "target_kl": 0.2,
        "gn": 10.0,
        "freq": 2,
    },
    "antmaze-large-diverse-v0": {
        "lr": 3e-4,
        "lambda": 0.5,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 7.0,
        "freq": 2,
    },
        "pen-human-v1": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "normalize",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
    "pen-cloned-v1": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "normalize",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
    "kitchen-complete-v0": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 250,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
    "kitchen-partial-v0": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
    "kitchen-mixed-v0": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
    "door-human-v1": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "normalize",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
    "door-cloned-v1": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "normalize",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
}


def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner("Loaded buffer")

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=args.discount,
        tau=args.tau,
        max_q_backup=args.max_q_backup,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.T,
        LA=args.LA,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_maxt=args.num_epochs,
        grad_norm=args.gn,
        policy_freq=args.policy_freq,
        target_kl=args.target_kl,
        LA_max=args.lambda_max,
        LA_min=args.lambda_min,
        kl_alpha = args.kl_alpha,
        mle_beta = args.mle_beta,
        gama = args.gama,
        f_div=args.f_div
    )

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.0)


    # writer = SummaryWriter(work_dir)
    utils.snapshot_src('.', os.path.join(work_dir, 'src'), '.gitignore') 
    # SummaryWriter(output_dir)
    
    utils.print_banner(f"Current time", separator="*", num_star=30)
    from datetime import datetime

    current_time = datetime.now().strftime("%H:%M:%S")
    print("Current time:", current_time)
    utils.print_banner(f"Parameters", separator="*", num_star=30)
    print("kl_alpha", args.kl_alpha)
    print("mle_beta", args.mle_beta)
    print("gama", args.gama)
    print("seed", args.seed)
    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.0
    utils.print_banner(f"Training Start", separator="*", num_star=30)
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(
            data_sampler,
            iterations=iterations,
            batch_size=args.batch_size,
            # log_writer=writer,
        )
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=30)
        logger.record_tabular("Trained Epochs", curr_epoch)
        logger.record_tabular("KL Loss", np.mean(loss_metric["kl_loss"]))
        logger.record_tabular("Actor Loss", np.mean(loss_metric["actor_loss"]))
        logger.record_tabular("Critic Loss", np.mean(loss_metric["critic_loss"]))
        logger.record_tabular("mlt", np.mean(loss_metric["mlt"]))
        logger.record_tabular("mltall", np.mean(loss_metric["mltall"]))
        logger.record_tabular("mlt_fix", np.mean(loss_metric["mlt_fix"]))
        logger.record_tabular("q_loss", np.mean(loss_metric["q_loss"]))
        logger.record_tabular("gen_loss", np.mean(loss_metric["gen_loss"]))
        logger.record_tabular("g_loss", np.mean(loss_metric["g_loss"]))
        # logger.record_tabular("g_flow_loss", np.mean(loss_metric["g_flow_loss"]))
        logger.record_tabular("log_li", np.mean(loss_metric["log_li"]))
        logger.record_tabular("gen_loss_log", np.mean(loss_metric["gen_loss_log"]))
        logger.dump_tabular()

        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(
            agent, args.env_name, args.seed, eval_episodes=args.eval_episodes
        )
        evaluations.append(
            [
                eval_res,
                eval_res_std,
                eval_norm_res,
                eval_norm_res_std,
                np.mean(loss_metric["kl_loss"]),
                # np.mean(loss_metric["ql_loss"]),
                np.mean(loss_metric["actor_loss"]),
                np.mean(loss_metric["critic_loss"]),
                curr_epoch,
            ]
        )
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.record_tabular("Average Episodic Reward", eval_res)
        logger.record_tabular("Average Episodic N-Reward", eval_norm_res)
        logger.dump_tabular()

        kl_loss = np.mean(loss_metric["kl_loss"])

        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

    scores = np.array(evaluations)
    best_id = np.argmax(scores[:, 2])
    best_res = {
        "epoch": scores[best_id, -1],
        "best normalized score avg": scores[best_id, 2],
        "best normalized score std": scores[best_id, 3],
        "best raw score avg": scores[best_id, 0],
        "best raw score std": scores[best_id, 1],
        "best_kl_alpha": args.kl_alpha,
        "best_mle_beta": args.mle_beta,
        "best_gama":args.gama,
        "best_seed":args.seed,
        "best_T":args.T,
    }
    with open(os.path.join(output_dir, f"{args.env_name}_best_score.txt"), "a") as f:
        f.write(json.dumps(best_res)+'\n')
        
    with open(os.path.join('runs', f"{args.env_name}_best_score.txt"), "a") as f:
        f.write(json.dumps(best_res)+'\n')


"""
Runs policy for X episodes and returns average reward
A fixed seed is used for the eval environment
"""


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.0
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}"
    )
    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###

    parser.add_argument(
        "--device", default="cuda:0", type=int
    )  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument(
        "--env_name", default="halfcheetah-medium-v2", type=str
    )  # OpenAI gym environment name
    parser.add_argument(
        "--dir", default="DiffpoGAN_results", type=str
    )  # Logging directory
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action="store_true")
    # parser.add_argument('--early_stop', action='store_true')
    parser.add_argument("--save_best_model", action="store_true")

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default="vp", type=str)

    ### Algo Settings ###
    # parser.add_argument("--policy_freq", default=2, type=int)
    
    # set yourself target_kl 
    # parser.add_argument("--target_kl", default=0.06, type=float)
    parser.add_argument("--lambda_max", default=100, type=int)
    parser.add_argument("--lambda_min", default=0, type=float)

    # gen params
    parser.add_argument("--kl_alpha", default=0.05, type=float)
    parser.add_argument("--mle_beta", default=0.05, type=float)
    parser.add_argument("--dis_lr", default=3e-5, type=float)
    parser.add_argument("--gama", default=0.01, type=float)

    parser.add_argument("--f_div", default="wgan", type=str)

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f"{args.dir}"
    args.num_epochs = hyperparameters[args.env_name]["num_epochs"]

    args.eval_episodes = 10 if "v2" in args.env_name else 100
    args.lr = hyperparameters[args.env_name]["lr"]
    args.LA = hyperparameters[args.env_name]["lambda"]
    args.policy_freq = hyperparameters[args.env_name]["freq"]
    args.max_q_backup = hyperparameters[args.env_name]["max_q_backup"]
    args.reward_tune = hyperparameters[args.env_name]["reward_tune"]
    args.gn = hyperparameters[args.env_name]["gn"]
    args.eval_freq = hyperparameters[args.env_name]["eval_freq"]
    
    # default kl, comment it when you set yourself kl parameters
    args.lambda_min = hyperparameters[args.env_name]["lambda_min"]
    args.target_kl = hyperparameters[args.env_name]["target_kl"]

    # loggg&result
    # Build work dir
    base_dir = 'runs'
    # utils.make_dir(base_dir)
    work_dir = os.path.join(base_dir, args.env_name)
    cooldir = generate_slug(2)
    # utils.make_dir(work_dir)
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H-%M", ts)
    exp_name = str(args.env_name) + '-' + ts + '-bs' + str(args.batch_size) + '-s' + str(args.seed) + '-a' + str(args.kl_alpha) + '-b' + str(args.mle_beta) + '-g' + str(args.gama) + '-gans' + str(args.f_div) + '-T' + str(args.T)
    exp_name += '-' + cooldir
    work_dir = work_dir + '/' + exp_name
    # Setup Logging 
    file_name = f"{args.env_name}-T-{args.T}"

    file_name += f"-seed_{args.seed}"
    file_name += f"-target_kl-{args.target_kl}"
    file_name += f"-lambda_min-{args.lambda_min}"

    results_dir = os.path.join(work_dir, file_name)
    # print(results_dir)
    # exit()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    utils.print_banner(f"Saving location: {results_dir}")

    variant = vars(args)
    variant.update(version=f"DiffpoGAN")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(
        f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}"
    )
    thread = 16
    torch.set_num_threads(int(thread))

    train_agent(env, state_dim, action_dim, max_action, args.device, results_dir, args)
