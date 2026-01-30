import os
import sys
import time
import random

import torch
import torch as T
import numpy as np
from make_env import make_env
# import torch.optim.lr_scheduler as lr_scheduler
import glob
# from pretrained_models.pretrained_predators_10 import get_predator_actions
from pretrained_models.predefined_reasonable_policy import get_predator_action_rs
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.ppo_fnn import PPO_FNN
from models.policy_representation import Policy_representation
from utils.clustering import *
import pickle
import wandb


def logit2onehot(action):
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action

DEVICE = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=345, help='random seed setting')
parser.add_argument('--bs', default=512, help='batch size for agent modeling training')
parser.add_argument('--freq', default=500, help='ppo algorithm update frequency')
parser.add_argument('--ratio', default=0.3, help='mask ratio')
parser.add_argument('--temp', default=0.5, help='contrastive temperature')
parser.add_argument('--has-vq', action='store_true')
parser.add_argument('--no-vq', dest='has-vq', action='store_false')
parser.set_defaults(has_vq=False)
parser.add_argument('--n-rules', type=int,default=3, help='number of fuzzy rules')
parser.add_argument('--order', type=int,default=3, help='order of TSK')
parser.add_argument('--update-data', type=str,default="cluster", help='using cluster center or buffer data to update rule')
args = parser.parse_args()

if __name__=="__main__":
    ####### initialize environment hyperparameters ######
    env_name = "simple_tag"
    max_ep_len = 50 # max timesteps in one episode
    total_test_episodes = int(2e4)  # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 100  # print avg reward in the interval (in num timesteps)
    save_model_freq = int(2e3)  # save model frequency (in num timesteps)
    random_seed = int(args.seed)  # set random seed if required (0 = no random seed)
    run_num_pretrained = 2
    render = False
    policy_num = 5
    print(f'policy type number: {policy_num}')
    ############ Agent Modeling Model parameters ########
    embedding_dim = 20
    agent_modeling_lr = 0.003
    # policy_type_num = 4
    buffer_size = 10000
    batch_size = int(args.bs)
    mask_ratio = float(args.ratio)
    contrast_temp = float(args.temp)
    embedding_vector = np.zeros((20))
    has_vq=args.has_vq
    n_rules=args.n_rules
    order=args.order
    update_data=args.update_data
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "fuzzy_clam_vq[{}], [{}]order_[{}]rules_at[{}]".format(has_vq,order, n_rules, current_time)
    wandb.init(project="fuzzy_clam", name=name)
    ################ PPO hyperparameters ################
    has_continuous_action_space = False  # continuous action space; else discrete
    update_timestep = max_ep_len * int(args.freq) # update policy every n timesteps
    update_episode = 50
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)

    ######################### log file setup ##############################
    results_dir = "results/clam_fnn_log"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    fuzzy_center_file = results_dir + '/' + env_name + "_fuzzy_center" + ".pkl"
    policy_embedding_file = results_dir + '/' + env_name + "_policy_embedding" + ".pkl"
    log_directory = results_dir + '/' + env_name + name
    
    writer = SummaryWriter(log_directory)
    
    ###################### print the hyperparameters #####################
    print('--------------------------------------------------')
    print('Hyperparameter list:')
    print("Random seed: ", args.seed)
    print('Batch size:', args.bs)
    print('PPO_update_frequency:', args.freq)
    print('Contrastive temperature:', args.temp)
    print('Mask ratio:', args.ratio)
    #######################################################################

    env = make_env(env_name)
    # state space dimension
    # agent_modeling_state_dim = env.observation_space[3].shape[0]
    n_agents = env.n
    state_dim = []
    for i in range(n_agents):
        state_dim.append(env.observation_space[i].shape[0])
    # agent_modeling_state_dim = sum(state_dim) # use all the agents' observation
    prey_obs_dim = state_dim[3]
    agent_modeling_state_dim = 21
    # state_dim = sum(actor_dims)
    state_dim_agent_modeling = int(prey_obs_dim) + int(embedding_dim)
    action_dim = env.action_space[0].n

    policy_represent = Policy_representation(buffer_size, batch_size, max_ep_len, agent_modeling_lr,
                                             agent_modeling_state_dim, embedding_dim, mask_ratio, contrast_temp, DEVICE,
                                             writer)
    ppo_prey = PPO_FNN(DEVICE,prey_obs_dim, action_dim, embedding_dim, has_vq, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                   has_continuous_action_space, action_std,n_rules=n_rules,order=order)
    print("============================================================================================")
    ################### random seed ####################
    if random_seed:
        T.manual_seed(random_seed)
        np.random.seed(random_seed)
        T.cuda.manual_seed_all(random_seed)
        random.seed(random_seed)
        env.seed(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    episode = 0
    steps = 0
    score_history_pred = []
    score_history_prey = []
    fuzzy_center_data = []
    policy_embedding = []
    fss = []
    for ep in range(0, total_test_episodes):
        w_max,w_min=ppo_prey.policy.actor.get_width_max_min()
        wandb.log({"Max width": w_max}, step=steps)
        wandb.log({"Min width": w_min}, step=steps)
        if ep % (update_episode*2) == 0 and ep > 0:
            if update_data=='cluster':
                center, std = policy_represent.get_center_std()


            center_data = [ep,  len(ppo_prey.policy.actor.rules),
                           [ppo_prey.policy.actor.rules[i].center for i in range (len(ppo_prey.policy.actor.rules))]]
            fuzzy_center_data.append(center_data)
            print('fuzzy center number:', len(ppo_prey.policy.actor.rules))
            writer.add_scalar('Cluster Center number', len(ppo_prey.policy.actor.rules), steps)
            wandb.log({"Cluster Center number": len(ppo_prey.policy.actor.rules)}, step=steps)
            policy_embedding.append(embedding_vector)
            if  has_vq:
                if update_data == 'cluster':
                    print("using cluster center to update fuzzy center")
                    ppo_prey.policy.actor.update_rules(torch.from_numpy(center).to(DEVICE))
                else:

                    print("using all policy embedding to update fuzzy center")
                    embedding=policy_represent.cluster_embedding()
                    ppo_prey.policy.actor.update_rules(embedding)
                # it should return number of rules and log it

            else:
                fs = ppo_prey.policy_old.actor[0].fss
                fss.append(fs)
                # log_f.write('{}, {}, {}\n'.format(ep, len(center), center.flatten()))
                print('fire strength:', fs.shape,fs.min(),fs.max())
                ppo_prey.policy.actor[0].update_centers(center, std)

        for i in range(policy_num):
            state = env.reset()
            episode_step = 0
            current_ep_reward_pred = 0
            current_ep_reward_prey = 0
            embedding_vector = np.zeros((20))
            # print('type:', i)
            for t in range(0, max_ep_len):
                actions = get_predator_action_rs(state[0:3], i)
                local_state = state[3]  # use only the observation of the controlled agent
                with T.no_grad():
                    embedding_vector = policy_represent.embedding(episode_step)

                state_with_context = np.concatenate((local_state, embedding_vector))
                if has_vq:
                    prey_action, vq_loss = ppo_prey.select_action(state_with_context)
                    writer.add_scalar('VQ_Loss', vq_loss.item(), steps)
                    wandb.log({"VQ_Loss": vq_loss.item()}, step=steps)
                else:
                    prey_action = ppo_prey.select_action(state_with_context)

                prey_action = logit2onehot(prey_action)
                actions.append(prey_action)
                # print(actions)
                if render:
                    env.render()
                    time.sleep(0.1)
                # state_all = obs2state(state) # use all the agents' observation

                # print(state_all)
                predator_actions = np.array(actions[0:3]).reshape((15))
                # print(predator_actions)
                state_action = np.concatenate((np.array(local_state[8:14]), np.array(predator_actions)))
                policy_represent.store_trajectories(episode, t, state_action)
                policy_represent.store_current_trajectories(t, state_action)
                state, reward, done, _ = env.step(actions)

                current_ep_reward_pred += reward[2]
                current_ep_reward_prey += reward[3]

                # saving reward and is_terminals
                ppo_prey.buffer.rewards.append(reward[3])
                ppo_prey.buffer.is_terminals.append(done[3])

                episode_step += 1
                steps += 1
                # print('step:', steps)

                if steps % update_timestep == 0:
                    ppo_prey.update()
                    # print("update")

            policy_represent.store_type_idx(i)
            episode += 1
            score_history_pred.append(current_ep_reward_pred)
            score_history_prey.append(current_ep_reward_prey)
        policy_represent.train()
        # ppo_prey.buffer.clear()

        # printing average reward
        if steps % print_freq == 0:
            # print average reward till last episode
            avg_score1 = np.mean(score_history_pred[-100:])
            avg_score2 = np.mean(score_history_prey[-100:])
            avg_score1 = round(avg_score1, 2)
            avg_score2 = round(avg_score2, 2)
            # avg_episode_step = statistics.mean(all_episode_step[-50:])
            writer.add_scalar('Predator average reward', avg_score1, steps)
            writer.add_scalar('Prey average reward', avg_score2, steps)

            wandb.log({"Predator average reward": avg_score1}, step=steps)
            wandb.log({"Prey average reward": avg_score2}, step=steps)
            print("Episode : {} \t\t Timestep : {} \t\t Predator average reward : {} \t\t Prey average reward : {}"
                  .format(episode, steps, avg_score1, avg_score2))

        # save model weights
        if (ep + 1) % save_model_freq == 0:
            ################### checkpoint ###################
            directory = "results/model/agent_modeling_co_training/"
            if not os.path.exists(directory):
                os.makedirs(directory)

            teaming_policy_path = directory + "PPO_policy_fnn_mlp_{}_{}_{}_{}.pth". \
                format(env_name, random_seed, ep, current_time)
            encoder_save_directory = directory + "clam_encoder_fnn_mlp_{}_{}_{}.pth".format(args.seed, ep,current_time)
            pooling_save_directory = directory + "clam_pooling_fnn_mlp_{}_{}_{}.pth".format(args.seed, ep,current_time)
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + encoder_save_directory)
            T.save(policy_represent.encoder.state_dict(), encoder_save_directory)
            T.save(policy_represent.pooling.state_dict(), pooling_save_directory)
            ppo_prey.save(teaming_policy_path)
            print("model saved")
            print("----------------------------------------------")
    with open(fuzzy_center_file, 'wb') as file:
        pickle.dump(fuzzy_center_data, file)
    with open(policy_embedding_file, 'wb') as file:
        pickle.dump(policy_embedding, file)
