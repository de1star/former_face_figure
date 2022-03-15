import sys
sys.path.append("..")

import time
import datetime
from tqdm import tqdm
import numpy as np
import torch.nn
from others import utils
from data.mhi_mimicry import mhi_mimicry
from torch.utils.data import DataLoader
import json
from model.transformer import Encoder_TRANSFORMER_Interaction, Encoder_TRANSFORMER, Decoder_TRANSFORMER
from others.loss import compute_kl_loss, compute_rc_loss
from collections import defaultdict
import os
import torch
from others.video_generation import generate_video, generate_video_pair
# torch.set_printoptions(threshold=10_000,precision=3,sci_mode=False,linewidth=160)
class Solver(object):
    """@DynamicAttrs"""
    def __init__(self, config):
        """Initialize configurations."""
        for k, v in vars(config).items():
            setattr(self, k, v)
        self.config = config
        self.device = torch.device(f'cuda:{self.i_cuda}') if torch.cuda.is_available() else torch.device('cpu')
        if self.mode == 'train':
            self.data_loader = self.get_loader(config, "train")
            self.valid_loader = self.get_loader(config, "valid")
            self.log_file = utils.get_logger(os.path.join(config.output_folder,self.result_dir), 'log')
            self.logger = utils.build_tensorboard(os.path.join(config.output_folder,self.result_dir))
            self.log_file.info(f'Args: {json.dumps(vars(config), indent=4, sort_keys=True)}')
            self.build_model()
        elif self.mode in 'test':
            self.test_loader = self.get_loader(config, "valid", 1)
            self.build_model()
        else:
            self.test_loader = self.get_loader(config, "valid", 1)
            if self.mode == 'rand_gen':
                self.random_generation()
            elif self.mode == 'rand_sel':
                self.train_loader = self.get_loader(config, "train", 1)
                self.random_selection()
            elif self.mode == 'run_avg': #11
                self.running_average_generation()
            elif self.mode == 'mirror': #13
                self.mirroring_generation()
            elif self.mode =='gt':
                self.ground_truth_generation()



    def get_loader(self, config, split, batch_size=None):
        if config.dataset == "mhi_mimicry":
            dataset = mhi_mimicry(config, split)
        else:
            raise ValueError(f"{config.dataset} is not a valid dataset")
        loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size if batch_size is None else batch_size,
                                 num_workers=config.num_worker,
                                 shuffle=split=="train",
                                 pin_memory=False,
                                 drop_last=True)
        return loader

    def build_model(self):
        input_feats = self.expression_params+self.jaw_pose_params+self.rotation_params+self.translation_params
        self.E_i = Encoder_TRANSFORMER_Interaction(self.config, input_feats=input_feats, bs=self.batch_size,num_layers=self.num_layers)
        self.E = Encoder_TRANSFORMER(self.config, input_feats=input_feats, bs=self.batch_size,num_layers=self.num_layers)
        self.D = Decoder_TRANSFORMER(self.config, input_feats=input_feats, bs=self.batch_size,num_layers=self.num_layers)
        if self.remove_z:
            trainable_parameters = list(self.E.parameters()) + list(self.D.parameters())
        else:
            trainable_parameters = list(self.E_i.parameters()) + list(self.E.parameters()) + list(self.D.parameters())
        if self.optimizer_choice == "sgd":
            self.optimizer = torch.optim.SGD(trainable_parameters, lr=self.lr,momentum=0.9)
        else:
            self.optimizer = torch.optim.AdamW(trainable_parameters, self.lr)
        # utils.print_network(self.E_i, 'transformer_encoder_interaction', self.log_file)
        # utils.print_network(self.E, 'transformer_encoder', self.log_file)
        # utils.print_network(self.D, 'transformer_decoder', self.log_file)
        self.E_i.to(self.device)
        self.E.to(self.device)
        self.D.to(self.device)

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def restore_model(self, resume_epochs):
        if resume_epochs is not None:
            E_i_path = os.path.join(self.output_folder, self.result_dir, "model", '{}-E_i.ckpt'.format(resume_epochs))
            self.E_i.load_state_dict(torch.load(E_i_path, map_location=lambda storage, loc: storage))
            E_path = os.path.join(self.output_folder, self.result_dir, "model", '{}-E.ckpt'.format(resume_epochs))
            D_path = os.path.join(self.output_folder, self.result_dir, "model", '{}-D.ckpt'.format(resume_epochs))

            self.E.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
            return resume_epochs
        else:
            for model in [self.E, self.D, self.E_i]:
                for p in model.parameters():
                    if p.dim() > 1:
                        torch.nn.init.xavier_normal_(p)
            return 0

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def print_log(self, loss, start_time, step):
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = f"Elapsed [{et}], Iter [{step}]"
        for tag, value in loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        self.log_file.info(log)

        for tag, value in loss.items():
            self.logger.scalar_summary(tag, value, step+1)

    def save_model(self,step):
        E_i_path = os.path.join(self.output_folder, self.result_dir, 'model', '{}-E_i.ckpt'.format(step))
        E_path = os.path.join(self.output_folder, self.result_dir, 'model', '{}-E.ckpt'.format(step))
        D_path = os.path.join(self.output_folder, self.result_dir, 'model', '{}-D.ckpt'.format(step))
        torch.save(self.E_i.state_dict(), E_i_path)
        torch.save(self.E.state_dict(), E_path)
        torch.save(self.D.state_dict(), D_path)
        self.log_file.info('Saved model checkpoints...')

    def reparameterize(self, mu, logvar, seed=None):
        std = torch.exp(logvar / 2)
        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z


    def get_data(self, split):
        try:
            return next(self.data_iter)
        except:
            if split =="train":
                self.data_iter = iter(self.data_loader)
            elif split == "valid":
                self.valid_iter = iter(self.valid_loader)
            else:
                self.test_iter = iter(self.test_loader)
            return next(self.data_iter)

    def schedule_sampling(self, step, t_step, mode='Linear'):
        if mode == 'Naive':
            p = 0.8
        elif mode == 'Linear':
            decrement = 1 / t_step
            p = 1 - decrement * step
        elif mode == 'Exponential':
            p = 0.99 ** step
        return torch.rand(self.batch_size) < p

    def train(self):
        # Start training from scratch or resume training.
        start_iters = self.restore_model(self.resume_iters)
        # Start training.
        self.log_file.info('Start training...')
        start_time = time.time()
        loss = defaultdict(list)
        loss_avg = {}
        self.E_i.train()
        self.E.train()
        self.D.train()
        for curr_iters in tqdm(range(start_iters, self.num_iters), leave=False):
            agent, interlocutor, sessions, frame_range, agent_idx = self.get_data("train")
            agent = agent.to(self.device).permute((1, 0, 2))
            agent_l = agent.clone()
            interlocutor = interlocutor.to(self.device).permute((1, 0, 2))
            if self.remove_z:
                z = torch.zeros(self.batch_size, 256).to(self.device)
            else:
                mu, logvar, _ = self.E_i(torch.cat([agent.clone(), interlocutor.clone()], dim=0))
                z = self.reparameterize(mu.clone(), logvar.clone())

            E_output = self.E(interlocutor)

            # schedule sampling
            if self.scheduled_sampling:
                self.D.eval()
                with torch.no_grad():
                    predict = self.D(z.clone().detach(), E_output.clone().detach(), agent.clone().detach())[:-1]
                choice = self.schedule_sampling(curr_iters, 100000)
                result = []
                for batch_idx, teacher_force in enumerate(choice):
                    if teacher_force:
                        result.append(agent[:, batch_idx])
                    else:
                        result.append(predict[:, batch_idx])
                agent = torch.stack(result, dim=1).to(self.device)
                self.D.train()
                output = self.D(z, E_output, agent)[:-1]
            else:
                output = self.D(z, E_output)
            # expression=[0:50], pose=[50:56],rotation=[56:59],shape=[59:159]
            # torch.Size([270, 16, 159])
            # print(agent.shape)

            exp_loss = compute_rc_loss(agent_l[:, :, 0:50], output[:, :, 0:50],self.config)
            jaw_loss = compute_rc_loss(agent_l[:, :, 50:53], output[:, :, 50:53],self.config)
            rot_loss = compute_rc_loss(agent_l[:, :, 53:56], output[:, :, 53:56],self.config)
            rc_loss = exp_loss + jaw_loss + rot_loss
            if self.remove_z:
                kl_loss = 0
            else:
                kl_loss = compute_kl_loss(mu, logvar)
            loss_mixed = rc_loss + kl_loss * self.kl_lambda
            self.reset_grad()
            loss_mixed.backward()
            self.optimizer.step()
            # Logging.
            loss['train/exp_loss'].append(exp_loss.item())
            loss['train/neck_loss'].append(jaw_loss.item())
            loss['train/jaw_loss'].append(jaw_loss.item())
            loss['train/rot_loss'].append(rot_loss.item())
            loss['train/kl_loss'].append(kl_loss if self.remove_z else kl_loss.item())
            loss['train/mixed_loss'].append(loss_mixed.item())
            if curr_iters%self.eval_iters == 0:# print loss
                self.E_i.eval()
                self.E.eval()
                self.D.eval()
                # evaluation
                with torch.no_grad():
                    for agent, interlocutor, sessions, frame_range, agent_idx in self.valid_loader:
                        # print(interlocutor.shape)
                        agent = agent.to(self.device).permute((1, 0, 2))
                        agent_l = agent.clone()
                        interlocutor = interlocutor.to(self.device).permute((1, 0, 2))
                        if self.remove_z:
                            z = torch.zeros(self.batch_size, 256).to(self.device)
                        else:
                            mu, logvar, _ = self.E_i(torch.cat([agent.clone(), interlocutor.clone()], dim=0))
                            z = self.reparameterize(mu.clone(), logvar.clone())
                        E_output = self.E(interlocutor)
                        if self.scheduled_sampling:
                            output = self.D(z, E_output.clone(),self.D(z.clone(), E_output.clone(), agent.clone())[:-1])[:-1]
                        else:
                            output = self.D(z, E_output.clone())
                        exp_loss = compute_rc_loss(agent_l[:, :, 0:50], output[:, :, 0:50], self.config)
                        jaw_loss = compute_rc_loss(agent_l[:, :, 50:53], output[:, :, 50:53], self.config)
                        rot_loss = compute_rc_loss(agent_l[:, :, 53:56], output[:, :, 53:56], self.config)
                        rc_loss = exp_loss + jaw_loss + rot_loss
                        if self.remove_z:
                            kl_loss = 0
                        else:
                            kl_loss = compute_kl_loss(mu, logvar)
                        loss_mixed = rc_loss + kl_loss * self.kl_lambda
                        # Logging.
                        loss['valid/exp_loss'].append(exp_loss.item())
                        loss['valid/jaw_loss'].append(jaw_loss.item())
                        loss['valid/rot_loss'].append(rot_loss.item())
                        loss['valid/kl_loss'].append(kl_loss if self.remove_z else kl_loss.item())
                        loss['valid/mixed_loss'].append(loss_mixed.item())
                # random z valid
                with torch.no_grad():
                    for agent, interlocutor, sessions, frame_range, agent_idx in self.valid_loader:
                        # print(interlocutor.shape)
                        agent = agent.to(self.device).permute((1, 0, 2))
                        agent_l = agent.clone()
                        interlocutor = interlocutor.to(self.device).permute((1, 0, 2))
                        if self.remove_z:
                            z = torch.zeros(self.batch_size, 256).to(self.device)
                        else:
                            z = torch.randn(self.batch_size, 256).to(self.device)
                        E_output = self.E(interlocutor)
                        if self.scheduled_sampling:
                            output = self.D(z, E_output.clone(), self.D(z.clone(), E_output.clone(), agent.clone())[:-1])[:-1]
                        else:
                            output = self.D(z, E_output.clone())
                        exp_loss = compute_rc_loss(agent_l[:, :, 0:50], output[:, :, 0:50], self.config)
                        jaw_loss = compute_rc_loss(agent_l[:, :, 50:53], output[:, :, 50:53], self.config)
                        rot_loss = compute_rc_loss(agent_l[:, :, 53:56], output[:, :, 53:56], self.config)
                        rc_loss = exp_loss + jaw_loss + rot_loss
                        if self.remove_z:
                            kl_loss = 0
                        else:
                            kl_loss = compute_kl_loss(mu, logvar)
                        loss_mixed = rc_loss + kl_loss * self.kl_lambda
                        # Logging.
                        loss['zvalid/exp_loss'].append(exp_loss.item())
                        loss['zvalid/jaw_loss'].append(jaw_loss.item())
                        loss['zvalid/rot_loss'].append(rot_loss.item())
                        loss['zvalid/kl_loss'].append(kl_loss if self.remove_z else kl_loss.item())
                        loss['zvalid/mixed_loss'].append(loss_mixed.item())
                if self.scheduled_sampling and curr_iters%(self.eval_iters*4) == 0:
                    # random z, rnn valid
                    with torch.no_grad():
                        for agent, interlocutor, sessions, frame_range, agent_idx in self.valid_loader:
                            # print(interlocutor.shape)
                            agent = agent.to(self.device).permute((1, 0, 2))
                            agent_l = agent.clone()
                            interlocutor = interlocutor.to(self.device).permute((1, 0, 2))
                            if self.remove_z:
                                z = torch.zeros(self.batch_size, 256).to(self.device)
                            else:
                                z = torch.randn(self.batch_size, 256).to(self.device)
                            E_output = self.E(interlocutor)

                            seq_len = agent.shape[0]
                            feature_dim = agent.shape[2]
                            decoder_output_pre = torch.empty([0]+list(agent_l.shape[1:]), device=self.device)
                            for i in range(seq_len):
                                decoder_output = self.D(z, E_output, decoder_output_pre)
                                decoder_output_pre = torch.cat([decoder_output_pre, decoder_output[-1][None]], dim=0)
                            output = decoder_output_pre
                            exp_loss = compute_rc_loss(agent_l[:, :, 0:50], output[:, :, 0:50], self.config)
                            jaw_loss = compute_rc_loss(agent_l[:, :, 50:53], output[:, :, 50:53], self.config)
                            rot_loss = compute_rc_loss(agent_l[:, :, 53:56], output[:, :, 53:56], self.config)
                            rc_loss = exp_loss + jaw_loss + rot_loss
                            if self.remove_z:
                                kl_loss = 0
                            else:
                                kl_loss = compute_kl_loss(mu, logvar)
                            loss_mixed = rc_loss + kl_loss * self.kl_lambda
                            # Logging.
                            loss['rvalid/exp_loss'].append(exp_loss.item())
                            loss['rvalid/jaw_loss'].append(jaw_loss.item())
                            loss['rvalid/rot_loss'].append(rot_loss.item())
                            loss['rvalid/kl_loss'].append(kl_loss if self.remove_z else kl_loss.item())
                            loss['rvalid/mixed_loss'].append(loss_mixed.item())
                for key in loss.keys():
                    loss_avg[key] = np.mean(loss[key])
                self.print_log(loss_avg, start_time, curr_iters)
                loss = defaultdict(list)
                loss_avg = {}

            if curr_iters%self.save_iters == 0:
                self.save_model(curr_iters)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=f'cuda:{self.i_cuda}')) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def random_generation(self):# mhi_10s
        loss_all = []
        for split_idx, (agent, interlocutor, sessions, frame_range, agent_idx) in enumerate(
                tqdm(self.test_loader, leave=False)):
            if split_idx < 136:
                continue
            agent = agent.to(self.device).permute((1, 0, 2))
            output = torch.randn(agent.shape,device=self.device)
            exp_loss = compute_rc_loss(agent[:, :, 0:50], output[:, :, 0:50], self.config)
            jaw_loss = compute_rc_loss(agent[:, :, 50:53], output[:, :, 50:53], self.config)
            rot_loss = compute_rc_loss(agent[:, :, 53:56], output[:, :, 53:56], self.config)
            rc_loss = exp_loss + rot_loss + jaw_loss
            loss_all.append(rc_loss.item())
            output_path = os.path.join(self.output_folder, f"random_generation_visualization")
            generate_video_pair(interlocutor.to(self.device).squeeze(dim=0).permute((1, 0)),
                                output.squeeze(dim=1).permute((1, 0)),
                                f"s{sessions[0]}_f{frame_range[0]}_generated_agent.avi", self.config, self.test_loader, output_path)
        print(np.mean(loss_all))

    def random_selection(self):# mhi_10s
        loss_all = []
        train_loader_iter = iter(self.train_loader)
        for split_idx, (agent, interlocutor, sessions, frame_range, agent_idx) in enumerate(
                tqdm(self.test_loader, leave=False)):
            if split_idx < 22:
                continue
            output, _, _, _, _ = next(train_loader_iter)
            agent = agent.to(self.device).permute((1, 0, 2))
            output = output.to(self.device).permute((1, 0, 2))
            interlocutor = interlocutor.to(self.device).permute((1, 0, 2))
            exp_loss = compute_rc_loss(agent[:, :, 0:50], output[:, :, 0:50], self.config)
            jaw_loss = compute_rc_loss(agent[:, :, 50:53], output[:, :, 50:53], self.config)
            rot_loss = compute_rc_loss(agent[:, :, 53:56], output[:, :, 53:56], self.config)
            rc_loss = exp_loss + rot_loss + jaw_loss
            loss_all.append(rc_loss.item())
            output_path = os.path.join(self.output_folder, f"random_selection_visualization")
            # torch.Size([56, 250])
            generate_video_pair(interlocutor.squeeze(dim=1).permute((1, 0)),
                                output.squeeze(dim=1).permute((1, 0)),
                                f"s{sessions[0]}_f{frame_range[0]}_generated_agent.avi", self.config, self.test_loader,
                                output_path)
        print(np.mean(loss_all))

    def running_average_generation(self):# mhi_11s
        loss_all = []
        for split_idx, (agent, interlocutor, sessions, frame_range, agent_idx) in enumerate(
                tqdm(self.test_loader, leave=False)):
            if split_idx < 0:
                continue
            interlocutor_l = interlocutor.to(self.device).permute((1, 0, 2))[12:-13]
            agent_lo = agent.to(self.device).permute((1, 0, 2)).clone()[12:-13]
            output_r = utils.moving_avg(interlocutor.squeeze(dim=0),26).to(self.device).unsqueeze(dim=1)
            exp_loss = compute_rc_loss(agent_lo[:, :, 0:50], output_r[:, :, 0:50], self.config)
            jaw_loss = compute_rc_loss(agent_lo[:, :, 50:53], output_r[:, :, 50:53], self.config)
            rot_loss = compute_rc_loss(agent_lo[:, :, 53:56], output_r[:, :, 53:56], self.config)
            rc_loss = exp_loss + rot_loss + jaw_loss
            loss_all.append(rc_loss.item())
            output_path = os.path.join(self.output_folder, f"running_average_visualization")
            # torch.Size([56, 250])
            generate_video_pair(interlocutor_l.squeeze(dim=1).permute((1, 0)),
                                output_r.squeeze(dim=1).permute((1, 0)),
                                f"s{sessions[0]}_f{frame_range[0]}_generated_agent.avi", self.config, self.test_loader,
                                output_path)
        print(np.mean(loss_all))

    def mirroring_generation(self):# mhi_13s
        loss_all = []
        for split_idx, (agent, interlocutor, sessions, frame_range, agent_idx) in enumerate(
                tqdm(self.test_loader, leave=False)):
            if split_idx < 16:
                continue
            interlocutor_l = interlocutor.to(self.device).permute((1, 0, 2))[75:]
            agent_lo = agent.to(self.device).permute((1, 0, 2)).clone()[75:]
            output = interlocutor.to(self.device).permute((1, 0, 2))[:-75]
            exp_loss = compute_rc_loss(agent_lo[:, :, 0:50], output[:, :, 0:50], self.config)
            jaw_loss = compute_rc_loss(agent_lo[:, :, 50:53], output[:, :, 50:53], self.config)
            rot_loss = compute_rc_loss(agent_lo[:, :, 53:56], output[:, :, 53:56], self.config)
            rc_loss = exp_loss + rot_loss + jaw_loss
            loss_all.append(rc_loss.item())
            output_path = os.path.join(self.output_folder, f"mirror_visualization")
            # torch.Size([56, 250])
            generate_video_pair(interlocutor_l.squeeze(dim=1).permute((1, 0)),
                                output.squeeze(dim=1).permute((1, 0)),
                                f"s{sessions[0]}_f{frame_range[0]}_generated_agent.avi", self.config, self.test_loader,
                                output_path)
        print(np.mean(loss_all))

    def ground_truth_generation(self):# mhi_10s
        loss_all = []
        for split_idx, (agent, interlocutor, sessions, frame_range, agent_idx) in enumerate(tqdm(self.test_loader, leave=False)):
            if split_idx<39:
                continue
            interlocutor_l = interlocutor.to(self.device).permute((1, 0, 2))
            agent_r = agent.to(self.device).permute((1, 0, 2)).clone()
            output_path = os.path.join(self.output_folder, f"gt_visualization")
            # torch.Size([56, 250])
            generate_video_pair(interlocutor_l.squeeze(dim=1).permute((1, 0)),
                                agent_r.squeeze(dim=1).permute((1, 0)),
                                f"s{sessions[0]}_f{frame_range[0]}_generated_agent.avi", self.config, self.test_loader,
                                output_path)
        print(np.mean(loss_all))

    def test(self):# mhi_10s
        self.restore_model(self.resume_iters)
        self.E.eval()
        self.D.eval()
        self.E_i.eval()
        loss_all = []
        with torch.no_grad():
            for agent, interlocutor, sessions, frame_range, agent_idx in tqdm(self.test_loader,leave=False):
                agent = agent.to(self.device).permute((1, 0, 2))
                agent_l = agent.clone()
                interlocutor = interlocutor.to(self.device).permute((1, 0, 2))
                E_output = self.E(interlocutor)
                seq_len = agent.shape[0]
                latent_dim = E_output.shape[2]

                if self.random_z:
                    z = torch.randn(1, latent_dim).to(self.device)
                elif self.remove_z:
                    z = torch.zeros(1, latent_dim).to(self.device)
                else:
                    z, _, _ = self.E_i(torch.cat([agent.clone(), interlocutor.clone()], dim=0))

                if self.scheduled_sampling:
                    seq_len = agent.shape[0]
                    feature_dim = agent.shape[2]
                    decoder_output_pre = torch.empty([0]+list(agent_l.shape[1:]), device=self.device)
                    for i in range(seq_len):
                        decoder_output = self.D(z, E_output, decoder_output_pre)
                        decoder_output_pre = torch.cat([decoder_output_pre, decoder_output[-1][None]], dim=0)
                    output = decoder_output_pre
                else:
                    output = self.D(z, E_output.clone())

                exp_loss = compute_rc_loss(agent_l[:, :, 0:50], output[:, :, 0:50], self.config)
                jaw_loss = compute_rc_loss(agent_l[:, :, 50:53], output[:, :, 50:53], self.config)
                rot_loss = compute_rc_loss(agent_l[:, :, 53:56], output[:, :, 53:56], self.config)
                rc_loss = exp_loss + rot_loss + jaw_loss
                loss_all.append(rc_loss.item())

                output_path = os.path.join(self.output_folder, self.result_dir, f"{self.resume_iters}_visualization")
                generate_video_pair(interlocutor.squeeze(dim=1).permute((1, 0)),
                                    output.squeeze(dim=1).permute((1, 0)),
                                    f"s{sessions[0]}_f{frame_range[0]}_generated_agent.avi", self.config,
                                    self.test_loader,
                                    output_path)


                # generate_video(output.squeeze(dim=1).permute((1, 0)), f"s{sessions[0]}_f{frame_range[0]}_generated_agent.avi", self.config,self.test_loader)
                # output_path = os.path.join(self.output_folder, f"{self.test_data}_visualization")
                # # original agent
                # generate_video(agent.squeeze(dim=1).permute((1, 0)),f"s{sessions[0]}_f{frame_range[0]}_agent.avi",self.config,self.test_loader, output_path)
                # # original interlocutor
                # generate_video(interlocutor.squeeze(dim=1).permute((1, 0)), f"s{sessions[0]}_f{frame_range[0]}_interlocutor.avi", self.config,self.test_loader, output_path)
        print(np.mean(loss_all))



