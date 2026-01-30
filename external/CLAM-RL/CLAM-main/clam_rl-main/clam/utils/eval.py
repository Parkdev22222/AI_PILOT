import os
import sys
import time
import random
import csv
import torch
import torch as T
import numpy as np
from make_env import make_env
# import torch.optim.lr_scheduler as lr_scheduler
import glob
from clam.pretrained_models.pretrained_predators_10 import get_predator_actions
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
from clam.models.ppo_fnn import PPO_FNN
from clam.models.policy_representation import Policy_representation
from clustering import *
import pickle
import wandb
import matplotlib.pyplot as plt
# from torchsummary import summary


def logit2onehot(action):
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action

DEVICE = T.device('cuda:1' if T.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=111, help='random seed setting')
parser.add_argument('--bs', default=512, help='batch size for agent modeling training')
parser.add_argument('--freq', default=500, help='ppo algorithm update frequency')
parser.add_argument('--ratio', default=0.3, help='mask ratio')
parser.add_argument('--temp', default=0.5, help='contrastive temperature')
parser.add_argument('--has-vq', action='store_true')
parser.add_argument('--no-vq', dest='has-vq', action='store_false')
parser.set_defaults(has_vq=False)
parser.add_argument('--n-rules', type=int,default=13, help='number of fuzzy rules')
parser.add_argument('--order', type=int,default=3, help='order of TSK')
parser.add_argument('--update-data', type=str,default="cluster", help='using cluster center or buffer data to update rule')
args = parser.parse_args()



def evaluate():
    print("============================================================================================")
    ################## directories ##################
    env_name = "simple_tag"
    directory = "/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_CLAM/agent_modeling_co_training/5rules/"

    ######################################### prey policy ###########################################################
    # prey_checkpoint_path = directory + 'PPO_policy_fnn_mlp_simple_tag_345_19999.pth'
    prey_checkpoint_path = directory + 'PPO_policy_fnn_mlp_simple_tag_345_19999_20231222_160220.pth'
    ######################################### policy modeling module ###########################################################
    # pooling_checkpoint_path = directory + "clam_pooling_fnn_mlp_345_19999.pth"
    # encoder_checkpoint_path = directory + "clam_encoder_fnn_mlp_345_19999.pth"
    pooling_checkpoint_path = directory + "clam_pooling_fnn_mlp_345_19999_20231222_160220.pth"
    encoder_checkpoint_path = directory + "clam_encoder_fnn_mlp_345_19999_20231222_160220.pth"
    ######################### log file setup ##############################
    results_dir = "results/clam_fnn_log"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fuzzy_center_file = results_dir + '/' + env_name + "_fuzzy_center" + ".pkl"
    policy_embedding_file = results_dir + '/' + env_name + "_policy_embedding" + ".pkl"
    log_directory = results_dir + '/' + env_name

    writer = SummaryWriter(log_directory)

    ################## hyperparameters ##################
    max_ep_len = 50  # max timesteps in one episode
    total_test_episodes = int(1)  # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 100  # print avg reward in the interval (in num timesteps)
    save_model_freq = int(2e3)  # save model frequency (in num timesteps)
    # random_seed = int(args.seed)  # set random seed if required (0 = no random seed)
    random_seed = False
    run_num_pretrained = 2
    render = True
    policy_num = 10
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
    has_vq = args.has_vq
    n_rules = args.n_rules
    order = args.order
    update_data = args.update_data
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "fuzzy_clam_vq[{}], [{}]order_[{}]rules_at[{}]".format(has_vq, order, n_rules, current_time)
    # wandb.init(project="fuzzy_clam", name=name)

    ################ PPO hyperparameters ################
    has_continuous_action_space = False  # continuous action space; else discrete
    update_timestep = max_ep_len * int(args.freq)  # update policy every n timesteps
    update_episode = 50
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)

    #####################################################

    env = make_env(env_name)
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

    # initialize a PPO agent and policy encoder
    ppo_prey = PPO_FNN(DEVICE, prey_obs_dim, action_dim, embedding_dim, has_vq, lr_actor, lr_critic, gamma, K_epochs,
                       eps_clip, has_continuous_action_space, action_std, n_rules=n_rules, order=order)
    policy_represent = Policy_representation(buffer_size, batch_size, max_ep_len, agent_modeling_lr,
                                             agent_modeling_state_dim, embedding_dim, mask_ratio, contrast_temp, DEVICE,
                                             writer)
    ppo_prey.load(prey_checkpoint_path)
    # print('ppo model:', ppo_prey.policy_old.parameters)
    policy_represent.load(encoder_checkpoint_path, pooling_checkpoint_path)
    # print('policy model:', policy_represent.target_encoder)

    print("--------------------------------------------------------------------------------------------")

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
    policy_embedding = []

    trajectory_pred0 = []
    trajectory_pred1 = []
    trajectory_pred2 = []
    trajectory_prey0 = []
    fss = []
    fss_argmax = []
    cq_outs = []
    policy_id = []
    imgs = []
    embs=[]
    ################## start evaluation ######################
    for ep in range(0, total_test_episodes):
        for i in range(policy_num):
        # for i in policy_num:
            state = env.reset()
            episode_step = 0
            current_ep_reward_pred = 0
            current_ep_reward_prey = 0
            embedding_vector = np.zeros((20))
            # print('type:', i)
            for t in range(0, max_ep_len):
                actions = get_predator_actions(state[0:3], i)
                local_state = state[3]  # use only the observation of the controlled agent
                trajectory_pred0.append(state[0][2:4])
                trajectory_pred1.append(state[1][2:4])
                trajectory_pred2.append(state[2][2:4])
                trajectory_prey0.append(state[3][2:4])
                # print('local obs:', local_state)
                with T.no_grad():
                    embedding_vector = policy_represent.embedding(episode_step)
                    embs.append(embedding_vector)
                state_with_context = np.concatenate((local_state, embedding_vector))
                with T.no_grad():
                    if has_vq:
                        prey_action, vq_loss = ppo_prey.select_action_eval(state_with_context)
                    else:
                        prey_action = ppo_prey.select_action_eval(state_with_context)

                prey_action = logit2onehot(prey_action)
                actions.append(prey_action)
                # get fs anc cq
                fs, cq = ppo_prey.policy_old.actor.get_fs_cq(state_with_context)
                # print("fs and cq",fs.shape, cq.shape)
                fs=fs[0].tolist()
                cq=cq[0].tolist()
                fss.append(fs)
                cq_outs.append(cq)
                policy_id.append(i)
                max_ind=fs.index(max(fs))
                # print("max_ind",max_ind)
                fss_argmax.append(max_ind)
                # print('fss:', ppo_prey.policy_old.actor.fss)
                # print(actions)
                if render:
                    img = env.render('rgb_array')
                    # img = env.viewers
                    imgs.append(img)
                    # plt.imsave('image.png', img)
                    # print(img[0].render)
                    # print(img[0].shape)
                    time.sleep(0.1)
                # state_all = obs2state(state) # use all the agents' observation

                # print(state_all)
                predator_actions = np.array(actions[0:3]).reshape((15))
                # print(predator_actions)
                state_action = np.concatenate((np.array(local_state[8:14]), np.array(predator_actions)))
                policy_represent.store_current_trajectories(t, state_action)
                state, reward, done, _ = env.step(actions)

                current_ep_reward_pred += reward[2]
                current_ep_reward_prey += reward[3]

                episode_step += 1
                steps += 1

            episode += 1
    combined_data = list(zip(trajectory_pred0, trajectory_pred1, trajectory_pred2, trajectory_prey0, embs, fss, cq_outs, policy_id, fss_argmax))
    np.savez(f'./policy{policy_num}_image.npz', *imgs)
    filename = './all_data_policy_{}.csv'.format(policy_num)
    with open(filename, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['predator_1', 'predator_2','predator_3', 'prey_1', 'embeddings','fss', 'cq_outs', 'policy_id', 'fss_argmax'])
        writer.writerows(combined_data)
        fuzzy_center_data = ppo_prey.policy_old.actor.get_centers().tolist()
        writer.writerow(fuzzy_center_data)
    print(f'Data saved to {filename}')

    env.close()


if __name__=="__main__":
    evaluate()