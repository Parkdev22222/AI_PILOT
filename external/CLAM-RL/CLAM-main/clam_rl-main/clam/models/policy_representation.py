import math
import random
import torch as T
import torch.optim as optim
import numpy as np
from clam.utils.contrastive_loss import SupConLoss
from clam.utils.clustering import *
from clam.models.basic_models import *

class Policy_representation():
    def __init__(self, buffer_size, batch_size, trajectory_step, lr, obs_act_dim, embedding_dim, mask_ratio,
                 contrast_temp, device, writer):
        self.device = device
        self.writer = writer
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # self.type_num = type_num
        self.trajectory_step = trajectory_step
        self.mem_cntr = 0
        self.timestep = 0
        self.tau = 0.99
        self.current_trajectory_size = 1
        self.cluster_sample_num  = 5000
        self.mask_ratio = mask_ratio
        self.contrast_temp = contrast_temp
        self.lr = lr
        self.encoder = TrajectoryEncoder(obs_act_dim, self.device)
        self.target_encoder = TrajectoryEncoder(obs_act_dim, self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.pooling = EmbeddingPooling(embedding_dim, self.device)
        self.projector = ProjectionHead(embedding_dim, 64, self.device)
        self.target_pooling = EmbeddingPooling(embedding_dim, self.device)
        self.target_pooling.load_state_dict(self.pooling.state_dict())
        self.pos_embedding = PositionalEmbedding(obs_act_dim).to(self.device)
        self.params = list(self.encoder.parameters()) + list(self.pooling.parameters()) + list(self.projector.parameters())
        self.optimizer = optim.Adam(self.params, lr)
        self.observation_memory = np.zeros((self.buffer_size, self.trajectory_step, obs_act_dim),
                                           dtype=np.float32)
        self.type_idx = np.zeros(self.buffer_size, dtype=np.float32)
        self.current_observation_memory = np.zeros((self.current_trajectory_size, self.trajectory_step, obs_act_dim),
                                           dtype=np.float32)
        self.current_action_memory = np.zeros((self.current_trajectory_size, self.trajectory_step, 1), dtype=np.float32)

    def store_trajectories(self, index, timestep, observation):
        self.index = index % self.buffer_size
        self.observation_memory[self.index][timestep] = observation
        self.mem_cntr = index

    def store_type_idx(self, idx:int):
        self.type_idx[self.index] = idx

    def store_current_trajectories(self, timestep, observation):
        self.current_observation_memory[0][timestep] = observation

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.buffer_size)
        if max_mem < batch_size:
            max_mem = batch_size
        batch = np.random.choice(max_mem, batch_size, replace=False)
        observation = self.observation_memory[batch]
        type_idx = self.type_idx[batch]

        return observation, type_idx
    def sample_sequence_buffer(self, batch_size):
        all_obs=[]
        max_mem = min(self.mem_cntr, self.buffer_size)
        if max_mem < batch_size:
            max_mem = batch_size
        for l in range (1,self.trajectory_step):
            batch = np.random.choice(max_mem, batch_size, replace=False)
            all_obs.append(self.observation_memory[batch,:l,:])
        all_obs=all_obs

        return all_obs
    # random mask certain percent of trajectory points
    def random_mask_trajectory(self, observation, percent):
        trajectory = observation.copy()
        mask_matrix = T.zeros((trajectory.shape[0], trajectory.shape[1])).to(self.device)
        if percent == 0:
            return trajectory, mask_matrix
        num_zero_rows = round(percent)
        for i in range(trajectory.shape[0]):
            zero_idxs = set(random.sample(range(trajectory.shape[1]), num_zero_rows))
            trajectory[i] = [([0] * len(trajectory[i][j]) if j in zero_idxs else trajectory[i][j]) for j in
                             range(len(trajectory[i]))]
            mask_matrix[i, list(zero_idxs)] = 1

        return trajectory, mask_matrix

    def random_crop_trajectory(self, observation, crop_length):
        trajectory = observation.copy()
        start_pos = np.random.randint(0, 50 - crop_length)
        trajectory = trajectory[:, start_pos:start_pos + crop_length, :]

        return trajectory, start_pos

    def train(self):
        if self.mem_cntr < self.batch_size:
            return

        crop_length_1 = np.random.randint(8, 49)
        mask_ratio_1 = self.mask_ratio
        mask_length_1 = math.ceil(crop_length_1 * mask_ratio_1)
        crop_length_2 = np.random.randint(8, 49)
        mask_ratio_2 = 0.0
        mask_length_2 = math.ceil(crop_length_2 * mask_ratio_2)

        # sample the trajectory batch
        observation, type_idx = self.sample_buffer(self.batch_size)
        observation_crop_1, start_pos_1 = self.random_crop_trajectory(observation, crop_length_1)
        observation_crop_2, start_pos_2 = self.random_crop_trajectory(observation, crop_length_2)

        # create the masked trajectory batches
        observation_masked, mask_matrix = self.random_mask_trajectory(observation_crop_1, mask_length_1)
        observation_masked_1, _ = self.random_mask_trajectory(observation_crop_2, mask_length_2)

        # convert to tensors and add position embedding, here we give the trajectory real position embedding
        observation_masked = T.tensor(observation_masked, dtype=T.float).to(self.device)
        observation_masked = self.pos_embedding(observation_masked) + observation_masked
        observation_masked_1 = T.tensor(observation_masked_1, dtype=T.float).to(self.device)
        observation_masked_1 = self.pos_embedding(observation_masked_1) + observation_masked_1

        # target_encoder update
        with T.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * self.tau + param_q.data * (1. - self.tau)
            for param_q, param_k in zip(self.pooling.parameters(), self.target_pooling.parameters()):
                param_k.data = param_k.data * self.tau + param_q.data * (1. - self.tau)

        # encode the two view of masked trajectories
        trajectory_representation = self.encoder(observation_masked)
        trajectory_representation_1 = self.encoder(observation_masked_1)

        # generate embedding vector, embedding trajectory step feature to policy representation
        trajectory_representation = trajectory_representation
        trajectory_representation_1_detach = trajectory_representation_1
        embedding_obs_masked = self.projector(self.pooling(trajectory_representation))
        embedding_obs_masked_1 = self.projector(self.pooling(trajectory_representation_1_detach))

        # embedding vector normalization
        obs_masked_norm = embedding_obs_masked.norm(p=2, dim=1, keepdim=True)
        embedding_obs_masked = embedding_obs_masked.div(obs_masked_norm.expand_as(embedding_obs_masked))
        obs_masked_norm_1 = embedding_obs_masked_1.norm(p=2, dim=1, keepdim=True)
        embedding_obs_masked_1 = embedding_obs_masked_1.div(obs_masked_norm_1.expand_as(embedding_obs_masked_1))

        # concatenate different views of trajectories' representation
        features = T.cat(
            [embedding_obs_masked.unsqueeze(1), embedding_obs_masked_1.unsqueeze(1)], dim=1)
        labels = T.tensor(type_idx).to(self.device)

        criterion = SupConLoss(temperature=self.contrast_temp)

        contrast_loss = criterion(features)
        loss = contrast_loss

        # two optimizers need to do the backward first then use optimizer.step() method one by one, otherwise, the graph parameter
        # would change and could not backward again by another optimizer
        with T.autograd.set_detect_anomaly(True):
            # this is for end-to-end encoder training
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #########################################
            # # this is for training the encoder and pooling part separately
            # self.optimizer1.zero_grad()
            # self.optimizer2.zero_grad()
            # recon_loss.backward(retain_graph=True)
            # contrast_loss.backward()
            #
            # self.optimizer1.step()
            # self.optimizer2.step()
            #########################################

        self.writer.add_scalar('contrastive_loss', contrast_loss, self.timestep)

        self.timestep += 1

    def embedding(self, trajectory_len):
        trajectory = self.current_observation_memory[0]
        trajectory_len = max(1, trajectory_len)
        trajectory = T.tensor(trajectory[0:trajectory_len], dtype=T.float).unsqueeze(0).to(self.device)
        trajectory = self.pos_embedding(trajectory) + trajectory
        embedding_vector = self.target_pooling(self.target_encoder(trajectory))

        # embedding vector normalization
        embedding_vector_norm = embedding_vector.norm(p=2, dim=0, keepdim=True)
        embedding_vector = embedding_vector.div(embedding_vector_norm.expand_as(embedding_vector))
        embedding_vector = embedding_vector.detach().cpu().numpy()

        return embedding_vector
    def cluster_embedding(self):
        all_embedding_vector=[]
        with T.no_grad():
            sq_ob=self.sample_sequence_buffer(self.cluster_sample_num)
            for observation in sq_ob:
                observation = T.tensor(observation, dtype=T.float).to(self.device)
                trajectory = self.pos_embedding(observation) + observation
                # traj_rep = self.encoder(observation).detach().cpu().numpy()
                embedding_vector = self.target_pooling(self.target_encoder(trajectory))

                # embedding vector normalization
                # print('embedding vector shape', embedding_vector.shape)
                embedding_vector_norm = embedding_vector.norm(p=2, dim=1, keepdim=True)
                embedding_vector = embedding_vector.div(embedding_vector_norm.expand_as(embedding_vector))
                all_embedding_vector.append(embedding_vector)
        all_embedding_vector=T.stack(all_embedding_vector,dim=0)
        return all_embedding_vector.view(-1,all_embedding_vector.shape[-1])


    def get_center_std(self):
        print("finding center from k-mean")
        embedding_vector = self.cluster_embedding()
        embedding_vector =embedding_vector.detach().cpu().numpy()
        # centers, std = silhouette_best(embedding_vector)
        centers, std = hdscan_best(embedding_vector)
        # print(type(centers), type(std))
        print("found centers from k-mean")
        return centers, std

    def load(self, encoder_path, pooling_path):
        self.target_encoder.load_state_dict(T.load(encoder_path, map_location=lambda storage, loc: storage))
        self.target_pooling.load_state_dict(T.load(pooling_path, map_location=lambda storage, loc: storage))

        self.target_encoder.eval()
        self.target_pooling.eval()
