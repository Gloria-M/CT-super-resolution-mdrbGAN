import os
from torch.utils.tensorboard import SummaryWriter

from models import *
from data_loader import make_loader
from utility_functions import *

class Trainer:
    def __init__(self, args):

        self._base_dir = args.base_dir
        self._train_dir = args.train_dir
        self._val_dir = args.val_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir
        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.tb_writer = SummaryWriter('runs/MultiResidualDenseBlocks_GAN_experiment')

        self._num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.end_epoch = self.start_epoch + self._num_epochs
        self.checkpoint_interval = args.checkpoint_interval
        self.log_interval = args.log_interval

        self._alpha = args.alpha_param
        self._lambda = args.lambda_coeff
        self._n_d_iter = args.n_d_iter

        self._lr = args.lr_init
        self._lr_decay = args.lr_decay

        self.model_gan = MDRB_GAN(self._device, self._alpha, self._lambda).to(self._device)
        self.optimizer_g = torch.optim.Adam(self.model_gan.generator.parameters(), lr=self._lr)
        self.optimizer_d = torch.optim.Adam(self.model_gan.discriminator.parameters(), lr=self._lr)

        self.train_dict = {'p_loss': [],
                           'g_loss': [],
                           'd_loss': [],
                           'psnr': [],
                           'sr_range': []
                           }
        self.val_dict = {'p_loss': [],
                         'g_loss': [],
                         'psnr': [],
                         'sr_range': []
                         }
                         
        train_names = ['{:d}'.format(slice_idx * 10 + 5) for slice_idx in range(16)]
        val_names = ['1028']
        self.train_loader = make_loader(self._train_dir, train_names, mode='train')
        self.val_loader = make_loader(self._val_dir, val_names, mode='train')

    def save_checkpoint(self, checkpoint_epoch):

        checkpoint_path = os.path.join(self._models_dir, 'checkpoint_{:d}.pt'.format(checkpoint_epoch))

        checkpoint_dict = {'generator': self.model_gan.generator.state_dict(),
                           'discriminator': self.model_gan.discriminator.state_dict(),
                           'optimizer_g': self.optimizer_g.state_dict(),
                           'optimizer_d': self.optimizer_d.state_dict(),
                           'train_loss': self.train_dict['p_loss'],
                           'train_psnr': self.train_dict['psnr'],
                           'train_range': self.train_dict['sr_range'],
                           'val_loss': self.val_dict['p_loss'],
                           'val_psnr': self.val_dict['psnr'],
                           'val_range': self.val_dict['sr_range']
                           }
        try:
            torch.save(checkpoint_dict, checkpoint_path)
            succes_message = 'Checkpoint(epoch {:d}) saved at {:s}'.format(checkpoint_epoch, checkpoint_path)
            print(success_format(succes_message))

        except Exception as ex:
            fail_message = 'Attempt to save checkpoint(epoch {:d}) at {:s} : {:s}'.format(
                checkpoint_epoch, checkpoint_path, ex.args[1])
            print(fail_format(fail_message))

    def load_checkpoint(self, checkpoint_epoch):

        checkpoint_path = os.path.join(self._models_dir, 'checkpoint_{:d}.pt'.format(checkpoint_epoch))
        try:
            checkpoint_dict = torch.load(checkpoint_path)
            succes_message = 'Checkpoint(epoch {:d}) at {:s} loaded'.format(checkpoint_epoch, checkpoint_path)
            print(success_format(succes_message))

        except Exception as ex:
            fail_message = 'Checkpoint(epoch {:d}) at {:s} : {:s}'.format(checkpoint_epoch, checkpoint_path, ex.args[1])
            print(fail_format(fail_message))
            return

        self.model_gan.generator.load_state_dict(checkpoint_dict['generator'])
        self.model_gan.generator.to(self._device)
        self.optimizer_g.load_state_dict(checkpoint_dict['optimizer_g'])
        for state in self.optimizer_g.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(self._device)

        self.model_gan.discriminator.load_state_dict(checkpoint_dict['discriminator'])
        self.model_gan.discriminator.to(self._device)
        self.optimizer_d.load_state_dict(checkpoint_dict['optimizer_d'])
        for state in self.optimizer_d.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(self._device)

        self.train_dict['p_loss'] = checkpoint_dict['train_loss']
        self.train_dict['psnr'] = checkpoint_dict['train_psnr']
        self.train_dict['sr_range'] = checkpoint_dict['train_range']

        self.val_dict['p_loss'] = checkpoint_dict['val_loss']
        self.val_dict['psnr'] = checkpoint_dict['val_psnr']
        self.val_dict['sr_range'] = checkpoint_dict['val_range']

    def update_learning_rate(self, epoch):

        self._lr *= self._lr_decay

        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] *= self._lr_decay

        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] *= self._lr_decay
        
        succes_message = 'Learning rate updated to {:.1e}'.format(self._lr)
        print(success_format(succes_message))

    def resume_epoch(self, checkpoint_epoch):

        self.start_epoch = checkpoint_epoch
        self.end_epoch = self.start_epoch + self._num_epochs

        self.load_checkpoint(checkpoint_epoch)
        for param_group in self.optimizer_g.param_groups:
            self._lr = param_group['lr']
            break

    def train(self):

        perceptual_loss = []
        generator_loss = []
        critic_loss = []
        psnr_score = []

        self.model_gan.generator.train()
        self.model_gan.discriminator.train()

        for batch_idx, (lr_ct, hr_ct) in enumerate(self.train_loader):
            lr_ct, hr_ct = lr_ct.squeeze(0), hr_ct.squeeze(0)
            lr_ct, hr_ct = lr_ct.to(self._device), hr_ct.to(self._device)

            for _ in range(self._n_d_iter):
                self.optimizer_d.zero_grad()
                d_loss = self.model_gan.w_loss(lr_ct, hr_ct)

                d_loss.backward()
                self.optimizer_d.step()

            self.optimizer_g.zero_grad()
            p_loss, g_loss = self.model_gan.g_loss(lr_ct, hr_ct)

            g_loss.backward()
            self.optimizer_g.step()

            psnr, sr_range = self.model_gan.psnr(lr_ct, hr_ct)

            perceptual_loss.append(p_loss.data.cpu().numpy())
            generator_loss.append(g_loss.data.cpu().numpy())
            critic_loss.append(d_loss.data.cpu().numpy())
            psnr_score.append(psnr)

        perceptual_loss = np.array(perceptual_loss).mean().item()
        generator_loss = np.array(generator_loss).mean().item()
        critic_loss = np.abs(np.array(critic_loss)).mean().item()
        psnr_score = np.array(psnr_score).mean().item()

        epoch_train_dict = {'p_loss': perceptual_loss,
                            'g_loss': generator_loss,
                            'd_loss': critic_loss,
                            'psnr': psnr_score,
                            'sr_range': sr_range
                            }

        return epoch_train_dict

    def validate(self):
        perceptual_loss = []
        generator_loss = []
        psnr_score = []

        self.model_gan.generator.eval()
        self.model_gan.discriminator.eval()

        with torch.no_grad():
            for batch_idx, (lr_ct, hr_ct) in enumerate(self.val_loader):
                lr_ct, hr_ct = lr_ct.squeeze(0), hr_ct.squeeze(0)
                lr_ct, hr_ct = lr_ct.to(self._device), hr_ct.to(self._device)

                p_loss, g_loss = self.model_gan.g_loss(lr_ct, hr_ct)
                psnr, sr_range = self.model_gan.psnr(lr_ct, hr_ct)

                perceptual_loss.append(p_loss.data.cpu().numpy())
                generator_loss.append(g_loss.data.cpu().numpy())
                psnr_score.append(psnr)

        perceptual_loss = np.array(perceptual_loss).mean().item()
        generator_loss = np.array(generator_loss).mean().item()
        psnr_score = np.array(psnr_score).mean()

        epoch_val_dict = {'p_loss': perceptual_loss,
                          'g_loss': generator_loss,
                          'psnr': psnr_score,
                          'sr_range': sr_range
                          }

        return epoch_val_dict

    def epoch_to_tensorboard(self, epoch, resume=False):
        
        if resume:
            for prev_epoch in range(epoch):                
                loss_dict = {'train': self.train_dict['p_loss'][prev_epoch],
                             'validation': self.val_dict['p_loss'][prev_epoch]}
                psnr_dict = {'train': self.train_dict['psnr'][prev_epoch],
                             'validation': self.val_dict['psnr'][prev_epoch]}

                self.tb_writer.add_scalars('Perceptual loss', loss_dict, global_step=prev_epoch+1)
                self.tb_writer.add_scalars('PSNR', psnr_dict, global_step=prev_epoch+1)
        else:
            loss_dict = {'train': self.train_dict['p_loss'][-1],
                         'validation': self.val_dict['p_loss'][-1]}
            psnr_dict = {'train': self.train_dict['psnr'][-1],
                         'validation': self.val_dict['psnr'][-1]}

            self.tb_writer.add_scalars('Perceptual loss', loss_dict, global_step=epoch+1)
            self.tb_writer.add_scalars('PSNR', psnr_dict, global_step=epoch+1)

    def make_ct_trio(self):

        for lr_ct, hr_ct in self.train_loader:
            lr_ct, hr_ct = lr_ct.squeeze(0), hr_ct.squeeze(0)
            lr_ct, hr_ct = lr_ct.to(self._device), hr_ct.to(self._device)
            break

        self.model_gan.generator.eval()
        with torch.no_grad():
            sr_ct = self.model_gan.generator(lr_ct)

        lr_ct, sr_ct, hr_ct = lr_ct.detach().cpu(), sr_ct.detach().cpu(), hr_ct.detach().cpu()

        return lr_ct, sr_ct, hr_ct
