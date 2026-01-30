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
from clam.utils.clustering import *
import pickle
import wandb
import matplotlib.pyplot as plt
# from torchsummary import summary
from sklearn.manifold import TSNE
import seaborn as sns
from itertools import chain
import umap
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
parser.add_argument('--n-rules', type=int,default=5, help='number of fuzzy rules')
parser.add_argument('--order', type=int,default=3, help='order of TSK')
parser.add_argument('--update-data', type=str,default="cluster", help='using cluster center or buffer data to update rule')
args = parser.parse_args()



def evaluate():
    print("============================================================================================")
    ################## directories ##################
    env_name = "simple_tag"
    # directory = "/projects/CIBCIGroup/00DataUploading/wenhao/clam/ckpts/clam_fuzzy_1223/"
    directory = "/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_CLAM/agent_modeling_co_training/5rules/"

    ######################################### prey policy ###########################################################
    # prey_checkpoint_path = directory + 'PPO_policy_fnn_mlp_simple_tag_345_19999_20231222_132135.pth'
    prey_checkpoint_path = directory + 'PPO_policy_fnn_mlp_simple_tag_345_19999_20231222_160220.pth'

    ######################################### policy modeling module ###########################################################
    # pooling_checkpoint_path = directory + "clam_pooling_fnn_mlp_345_19999_20231222_132135.pth"
    # encoder_checkpoint_path = directory + "clam_encoder_fnn_mlp_345_19999_20231222_132135.pth"
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
    total_test_episodes = int(100)  # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 100  # print avg reward in the interval (in num timesteps)
    save_model_freq = int(2e3)  # save model frequency (in num timesteps)
    # random_seed = int(args.seed)  # set random seed if required (0 = no random seed)
    random_seed = False
    run_num_pretrained = 2
    render = False
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
    fuzzy_center_data = []
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
    centers = []
    embs=[]
    for r in ppo_prey.policy_old.actor.rules:
        centers.append(r.center.cpu().detach().numpy())
    for i in range(policy_num):
        exec('embedding_vector{} = []'.format(i))

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
                    # print("embedding_vector",embedding_vector)
                    embs.append(embedding_vector)
                state_with_context = np.concatenate((local_state, embedding_vector))
                with T.no_grad():
                    if has_vq:
                        prey_action, vq_loss = ppo_prey.select_action_eval(state_with_context)
                    else:
                        prey_action = ppo_prey.select_action_eval(state_with_context)

                prey_action = logit2onehot(prey_action)
                actions.append(prey_action)

                fs, cq = ppo_prey.policy_old.actor.get_fs_cq(state_with_context)
                # print("fs and cq",fs.shape, cq.shape)
                fs = fs[0].tolist()
                cq = cq[0].tolist()
                fss.append(fs)
                cq_outs.append(cq)
                policy_id.append(i)
                max_ind = fs.index(max(fs))
                # print("max_ind",max_ind)
                fss_argmax.append(max_ind)
                # print('fss:', ppo_prey.policy_old.actor.fss)
                # print(actions)
                if render:
                    img = env.render('rgb_array')
                    # img = env.viewers
                    # print(img.shape)
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
                if t == (max_ep_len - 1):
                    exec('embedding_vector{}.append(embedding_vector)'.format(i))
                episode_step += 1
                steps += 1
            score_history_prey.append(current_ep_reward_prey)
            episode += 1

    ############################################# save the fuzzy relatd date ###########################################
    # combined_data = list(zip(trajectory_pred0, trajectory_pred1, trajectory_pred2, trajectory_prey0, fss, cq_outs, policy_id, fss_argmax))
    # np.savez(f'./eval_data/policy{policy_num}_image.npz', *imgs)
    # filename = './eval_data/all_data_policy_{}.csv'.format(policy_num)
    # with open(filename, mode="w", newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['predator_1', 'predator_2','predator_3', 'prey_1', 'fss', 'cq_outs', 'policy_id', 'fss_argmax'])
    #     writer.writerows(combined_data)
    # print(f'Data saved to {filename}')
    ####################################################################################################################
    env.close()

    ##################### Scikit-learn t-SNE ###########################
    X = []
    for i in range(policy_num):
        exec('X.append(np.stack(embedding_vector{}))'.format(i))
    X.append(np.stack(centers))
    X = np.concatenate((X), axis=0)
    y = np.ones(total_test_episodes, dtype=int)
    Y = []
    label = np.arange(policy_num)
    for i in range(policy_num):
        Y.append(y * label[i])
    # Y.append(np.ones(len(centers), dtype=int) * 20)
    Y = np.concatenate(Y)
    Y = np.transpose(Y)
    # data_with_label = np.concatenate((X, np.expand_dims(Y, axis=1)), axis=1)

    # df = pd.DataFrame(data_with_label)
    # df.to_csv('{}_percent_feature.csv'.format(PERCENT), index=False)
    # print('data_with_label shape', data_with_label.shape)

    ############################## 2d-plot ##################################################
    method='umap'
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
        # n_components = 2
        # tsne = TSNE(n_components, verbose=1)
    tsne_result = reducer.fit_transform(X)
    df = pd.DataFrame()
    print("tsne_result df",tsne_result.shape,df.shape)
    df['t-SNE-x'] = tsne_result[:-5, 0]
    df['t-SNE-y'] = tsne_result[:-5, 1]
    df['Policy index'] = Y+1
    x_point = tsne_result[-5:, 0]
    y_point = tsne_result[-5:, 1]

    ################################ seaborn 2-d plot ########################################
    # plt.figure(figsize=(10, 10))
    # colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # # colors = ['blue', 'green']
    #
    # palettes = []
    # for color in colors:
    #     palette = sns.light_palette(color, n_colors=7)
    #     palettes.append(palette[5:6])
    #
    # sns.scatterplot(
    #     x="t-SNE-x", y="t-SNE-y",
    #     hue="Policy index",
    #     palette=list(chain.from_iterable(palettes[0:policy_num])),
    #     # palette="deep",
    #     data=df,
    #     legend="full",
    #     alpha=0.8
    # )
    # plt.title('t-SNE Embedding')
    #
    # plt.show()

    # print("============================================================================================")

    plt.figure(figsize=(10, 10))
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # 添加了一个新颜色

    # 针对新加入的类别，我们需要添加一个新的形状
    # 假设你的新类别标签是10，我们会添加一个 'X' 形状来代表这个类别
    markers = ['o'] * policy_num # 假设原来的类别都使用圆圈'o'

    palettes = []
    for color in colors:
        palette = sns.light_palette(color, n_colors=7)
        palettes.append(palette[5:6])

    # 'style' 参数用于根据 'Policy index' 指定不同的标记形状
    # 新的数据类别需要在 'df' DataFrame 中正确标记
    sns.scatterplot(
        x="t-SNE-x", y="t-SNE-y",
        hue="Policy index",
        style="Policy index",  # 添加style参数
        markers=markers,  # 指定每个类别的标记形状
        palette=list(chain.from_iterable(palettes[0:(policy_num+1)])),
        data=df,
        legend="full",
        alpha=0.8
    )
    plt.title('t-SNE Embedding')
    special_points_x = tsne_result[-5:, 0]  # 假设这是你的特殊点的 x 坐标
    special_points_y = tsne_result[-5:, 1]  # 假设这是你的特殊点的 y 坐标
    annotations = ["center 1", "center 2", "center 3", "center 4", "center 5"]  # 对应的注释

    # 为每个特殊点添加注释
    for x, y, label in zip(special_points_x, special_points_y, annotations):
        plt.scatter(x, y, marker='*', s=500)
        plt.text(x, y, label, horizontalalignment='left', size='medium', color='black', weight='semibold')
    plt.show()
    # printing average reward
    if steps % print_freq == 0:
        # print average reward till last episode
        avg_score1 = np.mean(score_history_prey)
        avg_score1 = round(avg_score1, 2)
        print("Episode : {} \t\t Timestep : {} \t\t Prey average reward : {}"
              .format(episode, steps, avg_score1))
    print("============================================================================================")


if __name__=="__main__":
    evaluate()



