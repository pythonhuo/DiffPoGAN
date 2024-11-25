# Copyright 2023 Felix He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP, NICE
from agents.helpers import EMA

NEGATIVE_SLOPE = 1. / 100.


class DiscriminatorWithSigmoid(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscriminatorWithSigmoid, self).__init__()
        self.hidden_size = (400, 300)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size[0]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class DiffpoGAN(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            device,
            discount,
            tau,
            max_q_backup=False,
            LA=1.0,
            beta_schedule="linear",
            n_timesteps=100,
            ema_decay=0.995,
            step_start_ema=1000,
            update_ema_every=5,
            lr=3e-4,
            lr_decay=False,
            lr_maxt=1000,
            grad_norm=1.0,
            # policy_noise=0.2,
            # noise_clip=0.1,
            policy_freq=10,
            target_kl=0.05,
            LA_max=100,
            LA_min=0,
            kl_alpha = 0.05,
            mle_beta = 0.05,
            gama = 0.01,
            dis_lr=3e-5,
            f_div="wgan",
    ):
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.flow = NICE(xdim=state_dim + action_dim, mask_type='checkerboard0', no_of_layers=4,)

        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.model,
            max_action=max_action,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        # self.policy_noise = policy_noise
        # self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.LA = torch.tensor(LA, dtype=torch.float).to(device)  # Lambda
        self.LA_min = LA_min
        self.LA_max = LA_max

        self.discri = DiscriminatorWithSigmoid(state_dim, action_dim).to(device)
        self.discri_optimizer = torch.optim.Adam(self.discri.parameters(), lr=dis_lr)# 3e-5

        self.LA.requires_grad = True
        self.LA_optimizer = torch.optim.Adam([self.LA], lr=3e-5)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.lambda_lr_scheduler = CosineAnnealingLR(
                self.LA_optimizer, T_max=lr_maxt, eta_min=0.0
            )


        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau

        self.target_kl = target_kl
        self.device = device
        self.max_q_backup = max_q_backup
        self.dis_loss = torch.nn.BCELoss(reduction='mean')
        self.warm_start_epochs = 40
        self.real_data_pct = 0.05
        self.kl_alpha = kl_alpha
        self.mle_beta = mle_beta
        self.gama = gama
        self.f_div=f_div
        self.prior = "logistic"
        self.flow_optimizer = torch.optim.Adam(self.flow.parameters(), lr=1e-4, betas=(0.9,0.01), eps=1e-4,)

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {
            "kl_loss": [],
            # "ql_loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "Lambda": [],
            "ap": [],
            "generator_loss": [],
            "real_ac": [],
            "fake_ac": [],
            "real_loss": [],
            "fake_loss": [],
            "discriminator_loss": [],
            "c_fake_ac": [],
            "mlt": [],
            "mlt_fix": [],
            "q_loss": [],
            "gen_loss": [],
            "log_li": [],
            "mltall": [],
            "gen_loss_log":[],
            "g_loss": [],
        }
        _n_epochs = self.step
        real_data_pct_curr = min(max(1. - (_n_epochs - (self.warm_start_epochs + 1)) // 5 * 0.1, self.real_data_pct), 1.)
        n_real_data = int(real_data_pct_curr * batch_size)
        n_generated_data = batch_size - n_real_data
        warm_start = _n_epochs <= self.warm_start_epochs
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size
            )
            # print("state111",state)
            ##############################   flow-GAN
            gen_para, jac = self.flow(state, action)

            log_li = self.log_likelihood(gen_para, jac, self.prior) / batch_size





            advloss = nn.BCELoss(reduction='mean')
            ##############################   flow-GAN

            ex_state, ex_action,_, _,_ = replay_buffer.sample(
                batch_size
            )

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)
            true_samples1 = torch.cat([state, action], dim=1)
            true_samples2 = torch.cat([ex_state, ex_action], dim=1)
            true_samples = torch.cat([true_samples1, true_samples2], dim=0)

            ############################## flow-GAN
            real_data=true_samples.to(torch.float32)


            D_logits = self.discri(real_data)
            # print(D_logits.shape)
            
            ############################## flow-GAN

            real_ac = self.discri(true_samples)
            true_labels = torch.rand(size=(true_samples.size(0), 1), device=self.device) * (
                    1.0 - 0.80) + 0.80  # [0.80, 1.0)
            real_loss = self.dis_loss(real_ac, true_labels)
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                state_rpt = torch.repeat_interleave(state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                # action_rpt = self.ema_model(state_rpt)

                next_action_rpt = (next_action_rpt).clamp(
                    -self.max_action, self.max_action
                )

                state_rpt = torch.repeat_interleave(state, repeats=10, dim=0)
                action_rpt = self.ema_model(state_rpt)
                action_rpt = (action_rpt).clamp(
                    -self.max_action, self.max_action
                    )
                fake_samples_all = torch.cat([state_rpt, action_rpt], dim=1)
                next_fake_samples_all = torch.cat([next_state_rpt, next_action_rpt], dim=1)
                ind = torch.arange(0, 2560, 10).to(device=self.device)
                fake_samples1 = torch.index_select(fake_samples_all, dim=0, index=ind)
                fake_samples2 = torch.index_select(next_fake_samples_all, dim=0, index=ind)
                target_q1, target_q2 = self.critic_target(
                    next_state_rpt, next_action_rpt
                )
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)

            else:
                next_action = (self.ema_model(next_state)).clamp(
                    -self.max_action, self.max_action
                )
                pre_action = (self.ema_model(state)).clamp(
                    -self.max_action, self.max_action
                )
                fake_samples1 = torch.cat([state, pre_action], dim=1)
                fake_samples2 = torch.cat([next_state, next_action], dim=1)

                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            fake_samples = torch.cat([fake_samples1, fake_samples2], dim=0)

            fake_ac = self.discri(fake_samples.detach())
            ############################## flow-GAN
            D_logits_ = fake_ac

            ### Vanilla gan loss
            if self.f_div == 'ce':
                d_loss_real = torch.mean(
                    advloss(D_logits, torch.ones_like(D_logits)))
                d_loss_fake = torch.mean(
                    advloss(D_logits_, torch.zeros_like(D_logits_)))
            else:
            ### other gan losses
                if self.f_div == 'hellinger':
                    d_loss_real = torch.mean(torch.exp(-D_logits))
                    d_loss_fake = torch.mean(torch.exp(D_logits_) - 2.)
                
                elif self.f_div == 'rkl':
                    d_loss_real = torch.mean(torch.exp(D_logits))
                    d_loss_fake = torch.mean(-D_logits_ - 1.)
                
                elif self.f_div == 'kl':
                    d_loss_real = torch.mean(-D_logits)
                    d_loss_fake = torch.mean(torch.exp(D_logits_ - 1.))
                
                elif self.f_div == 'tv':
                    d_loss_real = torch.mean(-0.5 * torch.tanh(D_logits))
                    d_loss_fake = torch.mean(0.5 * torch.tanh(D_logits_))
                
                elif self.f_div == 'lsgan':
                    d_loss_real = 0.5 * torch.mean((D_logits - 0.9) ** 2)
                    d_loss_fake = 0.5 * torch.mean(D_logits_ ** 2)
                
                elif self.f_div == "wgan":
                    d_loss_real = -torch.mean(D_logits)
                    d_loss_fake = torch.mean(D_logits_)
                    # real=torch.cat((state, action), dim=1)
                    # fake=G.detach().data
                    real_data=true_samples.to(torch.float32)
                    fake_samples = torch.cat([fake_samples1.detach(), fake_samples2.detach()], dim=0)
                    gradient_penalty = self.calculate_gradient_penalty(self.discri,real_data.data, fake_samples.data)
                
                elif self.f_div == "ga":
                    d_loss_real = real_loss
                    fake_labels = torch.zeros(fake_samples.size(0), 1, device=self.device)
                    fake_loss = self.dis_loss(fake_ac, fake_labels)
                    d_loss_fake = fake_loss
                    # d_loss = (real_loss + fake_loss) / 2
        
            if self.f_div=="wgan":
                d_loss = d_loss_real+d_loss_fake+gradient_penalty * 0.5
            else:
                d_loss = d_loss_real+d_loss_fake
            ############################## flow-GAN



            
            
            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                current_q2, target_q
            )



            # discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss = d_loss
            # discriminator_loss = -real_ac.mean() + fake_ac.mean()
            self.discri_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            self.discri_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
            )
            self.critic_optimizer.step()

            # training policy every policy_freq steps

            if self.step % self.policy_freq == 0:
                """Policy Training"""
                # print(state.shape)
                kl_loss = self.actor.loss(action, state)

                new_action = self.actor(state)

                fake_samples3 = torch.cat([state, new_action], dim=1)
                fake_samples = torch.cat([fake_samples2, fake_samples3], dim=0)
                c_fake_ac = self.discri(fake_samples)
                ##############################    flow-GAN
                ### Vanilla gan loss
                D_logits1_ = c_fake_ac
                if self.f_div == 'ce':
                    g_loss = torch.mean(
                        advloss(D_logits1_, torch.ones_like(D_logits1_)*0.9))
                else:
                    ### other gan losses
                    if self.f_div == 'hellinger':
                        g_loss = torch.mean(torch.exp(-D_logits1_))
                    elif self.f_div == 'rkl':
                        g_loss = -torch.mean(-D_logits1_ - 1.)
                    elif self.f_div == 'kl':
                        g_loss = torch.mean(-D_logits1_)
                    elif self.f_div == 'tv':
                        g_loss = torch.mean(-0.5 * torch.tanh(D_logits1_))
                    elif self.f_div == 'lsgan':
                        g_loss = 0.5 * torch.mean((D_logits1_ - 0.9) ** 2)
                    elif self.f_div == "wgan":
                        g_loss = -torch.mean(D_logits1_)
                    elif self.f_div == "ga":
                        g_loss = self.dis_loss(c_fake_ac,
                                               torch.ones(fake_samples.size(0), 1, device=self.device))
                
                # generator_loss = self.dis_loss(c_fake_ac,
                #                                torch.ones(fake_samples.size(0), 1, device=self.device)) + log_li * 0.05
                generator_loss = g_loss + log_li * self.mle_beta
                ##############################    flow-GAN
                # generator_loss = (self.discri(fake_samples)).mean()
                q1_new_action, q2_new_action = self.critic(state, new_action)
                if np.random.uniform() > 0.5:
                    q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
                else:
                    q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()

                actor_loss = (

                    (c_fake_ac.mean()/real_ac.mean()).detach() * (self.kl_alpha * kl_loss) + q_loss + torch.log(generator_loss)*self.gama

                )
                
                self.actor_optimizer.zero_grad()
                self.flow_optimizer.zero_grad()
                actor_loss.backward()
                # if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )

                self.actor_optimizer.step()




                """ Lambda loss"""



                metric["actor_loss"].append(actor_loss.item())
                metric["kl_loss"].append(kl_loss.item())
                metric["generator_loss"].append(generator_loss.item())
                # metric["ql_loss"].append(q_loss.item())
                metric["real_ac"].append(real_ac.mean().item())
                metric["fake_ac"].append(fake_ac.mean().item())
                metric["c_fake_ac"].append(c_fake_ac.mean().item())
                metric["real_loss"].append(real_loss.item())
                # metric["fake_loss"].append(fake_loss.item())
                metric["critic_loss"].append(critic_loss.item())
                metric["discriminator_loss"].append(discriminator_loss.item())
                metric["mlt"].append((c_fake_ac.mean()/real_ac.mean()).item())
                metric["mlt_fix"].append(((c_fake_ac.mean()/real_ac.mean()).clamp(self.LA_min, self.LA_max)).item())
                metric["mltall"].append((c_fake_ac.mean()/real_ac.mean()*self.kl_alpha).item())
                metric["q_loss"].append((q_loss).item())
                metric["gen_loss"].append((generator_loss).item())
                metric["log_li"].append(log_li.item())
                metric["gen_loss_log"].append((torch.log(generator_loss)*self.gama).item())
                metric["g_loss"].append(g_loss.item())


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar(
                        "Actor Grad Norm", actor_grad_norms.max().item(), self.step
                    )
                    log_writer.add_scalar(
                        "Critic Grad Norm", critic_grad_norms.max().item(), self.step
                    )

                log_writer.add_scalar("KL Loss", kl_loss.item(), self.step)
                log_writer.add_scalar("real_ac", real_ac.mean().item(), self.step)
                log_writer.add_scalar("fake_ac", fake_ac.mean().item(), self.step)
                log_writer.add_scalar("c_fake_ac", c_fake_ac.mean().item(), self.step)
                log_writer.add_scalar("real_loss", real_loss.item(), self.step)
                # log_writer.add_scalar("fake_loss", fake_loss.item(), self.step)
                log_writer.add_scalar("discriminator_loss", discriminator_loss.item(), self.step)
                # log_writer.add_scalar("ap", ap.item(), self.step)
                # log_writer.add_scalar("QL Loss", q_loss.item(), self.step)
                log_writer.add_scalar("Critic Loss", critic_loss.item(), self.step)
                log_writer.add_scalar(
                    "Target_Q Mean", target_q.mean().item(), self.step
                )
                log_writer.add_scalar(
                    "generator_loss", generator_loss.item(), self.step
                )

  
        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            # self.discri_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # print(state.shape)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        # print(state_rpt.shape)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            # print(action.shape)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
            # print(idx.shape)
            # print(action[idx].cpu().data.numpy().flatten())
            # print(action[idx].cpu().data.numpy().flatten().shape)
            """
            Returns a tensor where each row contains num_samples indices sampled from the multinomial 
            probability distribution located in the corresponding row of tensor input.
            """
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{id}.pth")
            torch.save(self.critic.state_dict(), f"{dir}/critic_{id}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pth")
            torch.save(self.critic.state_dict(), f"{dir}/critic.pth")

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{id}.pth"))
            self.critic.load_state_dict(torch.load(f"{dir}/critic_{id}.pth"))
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pth"))
            self.critic.load_state_dict(torch.load(f"{dir}/critic.pth"))


    def compute_log_density_x(self,z, sum_log_det_jacobians, prior):

        zs = list(z.size())
        if len(zs) == 4:
            K = zs[1] * zs[2] * zs[3]  # dimension of the Gaussian distribution
            z = torch.reshape(z, (-1, K))
        else:
            K = zs[1]
        log_density_z = 0
        if prior == "gaussian":
            log_density_z = -0.5 * torch.sum(torch.square(z), 1) - 0.5 * K * np.log(2 * np.pi)
        elif prior == "logistic":
            m = nn.Softplus()
            temp=m(z)
            log_density_z = -torch.sum(-z + 2 * temp, 1)
        elif prior == "uniform":
            log_density_z = 0
        log_density_x = log_density_z + sum_log_det_jacobians
        
        return log_density_x

    def log_likelihood(self,z, sum_log_det_jacobians, prior):
        return -torch.sum(self.compute_log_density_x(z, sum_log_det_jacobians, prior))
    

    def calculate_gradient_penalty(self,model, real, fake):
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn((real.size(0), 1), device=self.device)

        # Get random interpolation between real and fake data
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)

        model_interpolates = model(interpolates)
        grad_outputs = torch.ones(model_interpolates.size(), requires_grad=False, device=self.device)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        )[0]
        gradients =gradients.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty