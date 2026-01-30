import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
import torch.nn.functional as F


class Fuzzy_1_order_rule(nn.Module):
    """ Membership layer """
    def __init__(self, center, width, ob_dim, output_dim=1, distance_metric='L1', positive=True):
        super().__init__()
        self.center = torch.Tensor(center).to('cuda')
        self.ante_dim = center.shape[0]
        self.cq_dim = ob_dim
        self.positive = positive
        self.output_dim = output_dim
        self.input_dim = self.center.shape[-1]
        self.widths = torch.Tensor(width).to('cuda')
        self.distance_metric = distance_metric
        self.consequent = nn.Sequential(
            nn.Linear(self.cq_dim, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, self.output_dim, bias=True),
        ).to('cuda')

    def get_dist(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if self.distance_metric == 'L1':
            # torch_dist = torch.norm(x - self.center, dim=1)
            torch_dist = x - self.center

        return torch_dist

    def get_Membership_values(self, x):
        # print("fuzzy input",x.shape)
        aligned_x = x
        dist = self.get_dist(aligned_x)
        # dist is already sum/prod, not on all dimensions
        # aligned_w=torch.clamp(self.widths,0.1,1)
        aligned_w = self.widths**2
        # print("dist {} aligned_w {} widths {}".format(dist.get_device(), aligned_w.get_device(), self.widths.get_device()))
        prot = torch.div(dist, aligned_w)
        # print("prot",prot.shape)
        root =- 0.5*(prot**2)
        membership_values = torch.exp(root)
        # print("gmv",x.shape,dist.shape,prot.shape,root.shape,membership_values.shape)

        return membership_values

    def get_FS(self, x):
        mvs = self.get_Membership_values(x)
        print("mvs",mvs)
        fs=mvs.prod(-1) + 0.0001
        print("fs",mvs.prod(-1) ,mvs.shape)
        return fs

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        fs = self.get_FS(x[:, self.cq_dim:])


        conq = self.consequent(x[:, :self.cq_dim])
        # conq=torch.Tensor(conq).to('cuda')
        out = conq*fs.unsqueeze(1)

        return out
class Fuzzy_HTSK(nn.Module):
    """ Membership layer """
    def __init__(self, center, width, ob_dim, output_dim=1, distance_metric='L1', positive=True):
        super().__init__()
        self.center = torch.Tensor(center).to('cuda')
        self.ante_dim = center.shape[0]
        self.cq_dim = ob_dim
        self.positive = positive
        self.output_dim = output_dim
        self.input_dim = self.center.shape[-1]
        self.widths = torch.Tensor(width).to('cuda')
        self.distance_metric = distance_metric
        self.consequent = nn.Sequential(
            nn.Linear(self.cq_dim, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, self.output_dim, bias=True),
        ).to('cuda')

    def get_dist(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if self.distance_metric == 'L1':
            # torch_dist = torch.norm(x - self.center, dim=1)
            torch_dist = x - self.center

        return torch_dist

    def get_Membership_z_values(self, x):
        # print("fuzzy input",x.shape)
        aligned_x = x
        dist = self.get_dist(aligned_x)
        # dist is already sum/prod, not on all dimensions
        # aligned_w=torch.clamp(self.widths,0.1,1)
        aligned_w = self.widths**2
        # print("dist {} aligned_w {} widths {}".format(dist.get_device(), aligned_w.get_device(), self.widths.get_device()))
        prot = torch.div(dist, aligned_w)
        # print("prot",prot.shape)
        root =- 0.5*(prot**2)
        # membership_values = torch.exp(root)
        # print("gmv",x.shape,dist.shape,prot.shape,root.shape,membership_values.shape)

        return root

    def get_z(self, x):
        mvs = self.get_Membership_z_values(x)
        fs_z=mvs.mean(-1)
        return fs_z

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        fs_z = self.get_z(x[:, self.cq_dim:])


        conq = self.consequent(x[:, :self.cq_dim])
        # conq=torch.Tensor(conq).to('cuda')
        # out = conq*fs.unsqueeze(1)

        return fs_z, conq
class Fuzzy_VQ_HTSK(nn.Module):
    """ Membership layer """
    def __init__(self, policy_dim, ob_dim, output_dim=1, distance_metric='L1', center=None,order=3):
        super().__init__()
        self.ante_dim = policy_dim
        self.cq_dim = ob_dim
        self.order=order
        self.output_dim = output_dim
        if center is None:
            self.center = nn.Parameter(torch.rand(policy_dim)*2-1)
        else:
            self.center = nn.Parameter(center)
        self.input_dim = self.center.shape[-1]
        self.widths = nn.Parameter(torch.ones(policy_dim)*0.25)
        self.distance_metric = distance_metric
        if self.order == 0:
            self.consequent = nn.Parameter(torch.rand(output_dim))
        elif self.order == 1:
            self.consequent = nn.Sequential(nn.Linear(self.cq_dim, output_dim, bias=True), nn.Softmax())
        elif self.order >= 2:
            layers = [nn.Linear(self.cq_dim, 64)]
            for l in range(self.order - 1):
                layers.append(nn.Linear(64, 64))
            layers.append(nn.Linear(64, output_dim))
            layers.append(nn.Softmax(dim=-1))
            self.consequent = nn.Sequential(*layers)

    def get_dist(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if self.distance_metric == 'L1':
            # torch_dist = torch.norm(x - self.center, dim=1)
            torch_dist = x - self.center

        return torch_dist

    def get_Membership_z_values(self, x):
        # print("fuzzy input",x.shape)
        aligned_x = x
        dist = self.get_dist(aligned_x)
        # dist is already sum/prod, not on all dimensions
        # aligned_w=torch.clamp(self.widths,0.1,1)
        aligned_w = self.widths
        # print("dist {} aligned_w {} widths {}".format(dist.get_device(), aligned_w.get_device(), self.widths.get_device()))
        prot = torch.div(dist, aligned_w)
        # print("prot",prot.shape)
        root =- 0.5*(prot**2)
        # membership_values = torch.exp(root)
        # print("gmv",x.shape,dist.shape,prot.shape,root.shape,membership_values.shape)

        return root

    def get_z(self, x):
        mvs = self.get_Membership_z_values(x)
        fs_z=mvs.mean(-1)
        return fs_z

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # print("x",x.min(),x.max())
        # print("center",self.center.min(), self.center.max())
        fs_z = self.get_z(x[:, self.cq_dim:])


        conq = self.consequent(x[:, :self.cq_dim])
        # conq=torch.Tensor(conq).to('cuda')
        # out = conq*fs.unsqueeze(1)

        return fs_z, conq
class FNN(nn.Module):
    def __init__(self, policy_dim, ob_dim, action_dim, k_clusters=None, stds=None):
        super(FNN, self).__init__()
        self.ob_dim = ob_dim
        self.action_dim = action_dim
        # self.rules = []
        # at begining we don't have data, just random create one rule
        if k_clusters is None:
            self.n_rules = 1
            self.rules = nn.ModuleList([Fuzzy_HTSK(torch.rand(policy_dim)*2-1,
                                                 torch.rand(policy_dim), ob_dim, output_dim=action_dim)])
        else:
        # generate rules
            self.n_rules = k_clusters.shape[0]
            self.rules = nn.ModuleList([Fuzzy_HTSK(k_clusters[i], stds[i] , ob_dim, output_dim=action_dim) for i in range(len(k_clusters))])
            # for i in range(len(k_clusters)):
            #     center=k_clusters[i]
            #     width=stds[i] # need to find propor width
            #     self.rules.append(Fuzzy_1_order_rule(center, width, ob_dim, output_dim=action_dim))

    def update_centers(self, candidate_centers, candidate_std):
        print("updating centers")
        # Create a copy of the candidate centers list to track used candidates
        # remaining_candidates = torch.from_numpy(candidate_centers).float().to('cuda')
        remaining_candidates = candidate_centers
        updated_centers = []
        for rule in self.rules:
        # for old_center in old_centers:
            old_center = rule.center
            # Calculate the distance between the old center and all remaining candidates
            distances = torch.cdist(old_center.unsqueeze(0), remaining_candidates)

            # Find the index of the candidate with the smallest distance
            closest_candidate_idx = torch.argmin(distances)

            # Add the closest candidate to the updated centers
            # updated_centers.append(remaining_candidates[closest_candidate_idx])
            rule.center = remaining_candidates[closest_candidate_idx]
            rule.widths = torch.from_numpy(candidate_std[closest_candidate_idx]).float().to('cuda')
            # Remove the used candidate from the list of remaining candidates
            remaining_candidates = torch.cat(
            (remaining_candidates[:closest_candidate_idx],
             remaining_candidates[closest_candidate_idx + 1:]), dim=0)
            del candidate_std[closest_candidate_idx]
            if len(candidate_std) == 0:
                break
        if len(candidate_std) > 0:
            for j in range (remaining_candidates.shape[0]):
                center = remaining_candidates[j]
                widths = torch.from_numpy(candidate_std[j]).float().to('cuda')  # need to find propor width
                self.rules.append(Fuzzy_HTSK(center.cpu(), widths.cpu(), self.ob_dim, output_dim=self.action_dim))

    def forward(self, X):
        if len(X.shape) == 1:
            X_tensor = X.unsqueeze(0)
        else:
            X_tensor = X
        # if torch.cuda.is_available():
        #     X_tensor = X_tensor.to('cuda')
        cq_outs = []
        z_outs = []
        for rule in self.rules:
            # rule = rule.to('cuda')
            # 1. get ruleout
            # print('x_tensor device:', X_tensor.get_device())
            z, cq = rule(X_tensor)
            cq_outs.append(cq)
            z_outs.append(z)
            # 2.get FS out
            # fs = rule.get_FS(X_tensor[:, rule.cq_dim:])
            # fs_outs.append(fs)
        z_outs = torch.stack(z_outs,dim=-1)
        fs_outs=F.softmax(z_outs,dim=-1)
        cq_outs = torch.stack(cq_outs,dim=-2)
        self.fss = fs_outs
        FNN_outs=cq_outs * fs_outs.unsqueeze(-1)
        FNN_outs=FNN_outs.sum(-2)

        return FNN_outs
class VQ_FNN(nn.Module):
    def __init__(self, n_rules, policy_dim, ob_dim, action_dim, k_clusters=None, stds=None,device=None,order=3):
        super(VQ_FNN, self).__init__()
        self.policy_dim=policy_dim
        self.device=device
        self.ob_dim = ob_dim
        self.action_dim = action_dim
        self._commitment_cost = 0.25
        self.rule_dropout=nn.Dropout(0.0)
        self.order=order
        # self.rules = []
        # at begining we don't have data, just random create one rule
        self.softmax = nn.Softmax(dim=-1)
        self.rules = nn.ModuleList([Fuzzy_VQ_HTSK(policy_dim, ob_dim, output_dim=action_dim) for i in range(n_rules)])
    def get_all_fs(self,policy_embeding):
        z_outs=[]
        with torch.no_grad():
            for rule in self.rules:
                z = rule.get_z(policy_embeding)
                z_outs.append(z)
            z_outs = torch.stack(z_outs,dim=-1)
            all_fs=F.softmax(z_outs,dim=-1)
        return all_fs
    def update_rules(self, x):
        self.drop_rules(x)
        print("after drop",len(self.rules))
        self.add_rules(x)
        print("after add",len(self.rules))
    def drop_rules(self,x,threshold=0.8):
        while len(self.rules)>=2:
            all_fs=self.get_all_fs(x)
            print("all_fs max ind",all_fs.argmax(-1))
            mask = torch.all(all_fs > threshold, dim=0)
            # Get the indices where the condition is not met
            mask=[not elem for elem in mask]
            if any(mask):
                if False in mask:
                    self.rules = nn.ModuleList([module for module, include in zip(self.rules, mask) if include])
                else:
                    break
            else:
                break
            # most_fired_rule=all_fs.argmax(-1)
            # unique_values, counts = most_fired_rule.unique(return_counts=True)
            # most_frequent_index = counts.argmax()
            # proportion = counts[most_frequent_index].item() / len(most_fired_rule)
            # print("in {} cluster, rule[{}] doni")
            # if proportion>=0.8:
            #     del self.rules[most_frequent_index]
            # else:
            #     break
        # all_fs = self.get_all_fs(x)
        # small_rule_condition = (all_fs < 2/len(self.rules)).all(dim=0)
        # # Get the indices where the condition is True
        # small_indices = torch.nonzero(small_rule_condition).squeeze().tolist()
        # if small_indices is list and len(small_indices)>=1:
        #     self.rules = nn.ModuleList([self.rules[i] for i in range(len(self.rules)) if i not in small_indices])
    def add_rules(self, x):
        ood_samples=x
        while len(ood_samples)>2 and len(self.rules) <=20:
            threshold=1.1/len(self.rules)
            if len(self.rules)<=1:
                self.rules.append(Fuzzy_VQ_HTSK(self.policy_dim, self.ob_dim, output_dim=self.action_dim, center=ood_samples[0]).to(self.device))
                ood_samples = ood_samples[1:]
                pass
            all_fs=self.get_all_fs(ood_samples)
            max_probabilities, predicted_classes = torch.max(all_fs, dim=1)
            confused_mask = max_probabilities < threshold
            print("max_probabilities",max_probabilities)
            ood_indices = torch.nonzero(confused_mask).squeeze()
            if ood_indices.dim() == 0:
                break
            ood_samples=ood_samples[ood_indices]
            if len(ood_samples)>2:
                self.rules.append(Fuzzy_VQ_HTSK(self.policy_dim, self.ob_dim, output_dim=self.action_dim, center=ood_samples[0]).to(self.device))
                ood_samples=ood_samples[1:]
    def get_width_max_min(self):
        widths=[]
        for rule in self.rules:
            widths.append(rule.widths)
        widths=torch.stack(widths)
        return widths.max(), widths.min()
    def get_fs_cq(self, X):
        if len(X.shape) == 1:
            X_tensor = X.unsqueeze(0)
        else:
            X_tensor = X
        # if torch.cuda.is_available():
        #     X_tensor = X_tensor.to('cuda')
        cq_outs = []
        z_outs = []
        for rule in self.rules:
            # rule = rule.to('cuda')
            # 1. get ruleout
            # print('x_tensor device:', X_tensor.get_device())
            z, cq = rule(X_tensor)
            cq_outs.append(cq)
            z_outs.append(z)
            # 2.get FS out
            # fs = rule.get_FS(X_tensor[:, rule.cq_dim:])
            # fs_outs.append(fs)
        z_outs = torch.stack(z_outs,dim=-1)
        fs_outs=F.softmax(z_outs,dim=-1)
        fs_outs=self.rule_dropout(fs_outs)
        cq_outs = torch.stack(cq_outs,dim=-2)
        return fs_outs,cq_outs
    def forward(self, X):
        if len(X.shape) == 1:
            X_tensor = X.unsqueeze(0)
        else:
            X_tensor = X
        # if torch.cuda.is_available():
        #     X_tensor = X_tensor.to('cuda')
        cq_outs = []
        z_outs = []
        for rule in self.rules:
            # rule = rule.to('cuda')
            # 1. get ruleout
            # print('x_tensor device:', X_tensor.get_device())
            z, cq = rule(X_tensor)
            cq_outs.append(cq)
            z_outs.append(z)
            # 2.get FS out
            # fs = rule.get_FS(X_tensor[:, rule.cq_dim:])
            # fs_outs.append(fs)
        z_outs = torch.stack(z_outs,dim=-1)
        fs_outs=F.softmax(z_outs,dim=-1)
        fs_outs=self.rule_dropout(fs_outs)
        cq_outs = torch.stack(cq_outs,dim=-2)
        self.fss = fs_outs
        FNN_outs=cq_outs * fs_outs.unsqueeze(-1)
        FNN_outs=FNN_outs.sum(-2)
        # add vq loss here

        # Encoding
        encoding_indices = torch.argmax(fs_outs, dim=-1).unsqueeze(-1)
        encodings = torch.zeros(encoding_indices.shape[0], len(self.rules), device=X_tensor.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten
        # get centers
        centers = [self.rules[i].center for i in range(len(self.rules))]
        centers = torch.stack(centers, dim=0)
        quantized = torch.matmul(encodings, centers)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), X_tensor[:, self.ob_dim:])
        q_latent_loss = F.mse_loss(quantized, X_tensor[:, self.ob_dim:].detach())
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        vq_loss = torch.tensor(0)
        return self.softmax(FNN_outs), vq_loss