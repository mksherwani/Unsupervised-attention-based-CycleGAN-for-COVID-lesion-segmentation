 import os
from os import makedirs
import time
import datetime
import tqdm
import itertools
import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv
from torchsummary import summary
from torchnet.meter import MovingAverageValueMeter
from collections import OrderedDict

# local module
from model import ResnetGenerator, NLayerDiscriminator, GANLoss
from data import create_dataset


class Config(object):
    def __init__(self):
        self.name = 'cycleGan_demo'
        self.dataset_name = 'unhealthy2healthy'
        self.comment = 'put your notes here for each experiment'

        self.dataroot = './datasets/' + self.dataset_name
        self.save_path = './checkpoints/' + self.name
        self.model_path = self.save_path + '/models'
        self.checkpoint_path = self.save_path + '/checkpoints'
        self.test_path = self.save_path + '/test_results'

        self.max_dataset_size = 2000
        self.dataset_mode = 'unaligned'
        self.direction = 'AtoB'
        self.preprocess = 'resize_and_crop'
        self.load_size = 320
        self.crop_size = 256
        self.no_flip = True
        self.num_threads = 4

        # network
        self.input_nc = 1                    # input channel number
        self.output_nc = 1                   # output channel number
        self.n_blocks = 7                    # ResNet Block numbers (affects networks size)
        self.beta1 = 0.5                     # Adam optimizer beta1
        self.gpu = True                      # use GPU?
        self.ngf = 64                        # number of gen filters in the last conv layer
        self.ndf = 64                        # number of discrim filters in the first conv layer
        self.use_dropout = False             # Dropout is not used in the original CycleGAN paper
        self.init_type = 'normal'            # normal | xavier | kaiming | orthogonal

        self.norm = 'instance'               # normalization, instance by default
        self.netG = f'resnet_{self.n_blocks}blocks'
        self.netD = 'basic'                  # The basic model is a 70x70 PatchGAN. n_layers=3
        self.gan_mode = 'lsgan'              # MSELoss | BCEWithLogitsLoss

        # training options
        self.phase = 'train'                 # train or test
        self.batch_size = 8
        self.max_epochs = 200
        self.g_lr = 2e-4                     # generator learning rate
        self.d_lr = 2e-4                     # discriminator learning rate
        self.G_path = None                   # for continue training
        self.D_path = None
        self.serial_batches = False

        # visualization options
        self.vis = True
        self.env = 'GAN'
        self.plot_every = 100  # iterations
        self.save_every = 10  # epochs

        # create directories
        makedirs(self.save_path, exist_ok=True)
        makedirs(self.model_path, exist_ok=True)
        makedirs(self.checkpoint_path, exist_ok=True)
        makedirs(self.test_path, exist_ok=True)


def train(**kwargs):
    t.cuda.empty_cache()

    """ Get options """

    opt = Config()
    print_options(opt)

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    iter_per_epoch = int(dataset_size / opt.batch_size)
    print(f'loaded {dataset_size} images for training')

    model_names = ['netG_x', 'netG_y', 'netD_x', 'netD_y']

    netG_x = ResnetGenerator(opt)
    netG_y = ResnetGenerator(opt)
    # print(netG_x)

    netD_x = NLayerDiscriminator(opt)
    netD_y = NLayerDiscriminator(opt)
    # print(netD_x)

    if opt.gpu:
        netG_x.to(device)
        summary(netG_x, input_size=(3, opt.crop_size, opt.crop_size))
        netG_y.to(device)

        netD_x.to(device)
        summary(netD_x, input_size=(3, opt.crop_size, opt.crop_size))
        netD_y.to(device)

    """ Define optimizer and Loss """
    optimizer_g = t.optim.Adam(itertools.chain(netG_x.parameters(), netG_y.parameters()),
                               lr=opt.g_lr,
                               betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(itertools.chain(netD_x.parameters(), netD_y.parameters()),
                               lr=opt.d_lr,
                               betas=(opt.beta1, 0.999))
    optimizers = [optimizer_g, optimizer_d]

    Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
    Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
    Identity loss (optional):
    lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)

    lambda_X = 10.0  # weight for cycle loss (A -> B -> A^)
    lambda_Y = 10.0  # weight for cycle loss (B -> A -> B^)
    lambda_identity = 0.5
    criterionGAN = GANLoss(gan_mode='lsgan')

    # cycle loss
    criterionCycle = nn.L1Loss()

    # identical loss
    criterionIdt = nn.L1Loss()


    # loss meters
    loss_X_meter = MovingAverageValueMeter(opt.plot_every)
    loss_Y_meter = MovingAverageValueMeter(opt.plot_every)
    score_Dx_real_y = MovingAverageValueMeter(opt.plot_every)
    score_Dx_fake_y = MovingAverageValueMeter(opt.plot_every)

    losses = {}
    scores = {}



    for epoch in range(opt.max_epochs):
        epoch_start_time = time.time()

        for i, data in enumerate(dataset):

            real_x = data['A'].to(device)
            real_y = data['B'].to(device)

            ######################
            # X -> Y' -> X^ cycle
            ######################

            optimizer_g.zero_grad()  # set g_x and g_y gradients to zero

            fake_y = netG_x(real_x)         # X -> Y'
            prediction = netD_x(fake_y)     #netD_x provide feedback to netG_x
            loss_G_X = criterionGAN(prediction, True)

            # cycle_consistance
            x_hat = netG_y(fake_y)          # Y' -> X^
            # Forward cycle loss x^ = || G_y(G_x(real_x)) ||
            loss_cycle_X = criterionCycle(x_hat, real_x) * lambda_X

            # identity loss
            if lambda_identity > 0:
                # netG_x should be identity if real_y is fed: ||netG_x(real_y) - real_y||
                idt_x = netG_x(real_y)
                loss_idt_x = criterionIdt(idt_x, real_y) * lambda_Y * lambda_identity
            else:
                loss_idt_x = 0.

            loss_X = loss_G_X + loss_cycle_X + loss_idt_x
            loss_X.backward(retain_graph=True)
            optimizer_g.step()

            loss_X_meter.add(loss_X.item())

            ######################
            # Y -> X' -> Y^ cycle
            ######################

            optimizer_g.zero_grad()  # set g_x and g_y gradients to zero

            fake_x = netG_y(real_y)         # Y -> X'
            prediction = netD_y(fake_x)
            loss_G_Y = criterionGAN(prediction, True)
            # print(f'loss_G_Y = {round(float(loss_G_Y), 3)}')

            y_hat = netG_x(fake_x)          # Y -> X' -> Y^
            # Forward cycle loss y^ = || G_x(G_y(real_y)) ||
            loss_cycle_Y = criterionCycle(y_hat, real_y) * lambda_Y

            # identity loss
            if lambda_identity > 0:
                # netG_y should be identiy if real_x is fed: ||netG_y(real_x) - real_x||
                idt_y = netG_y(real_x)
                loss_idt_y = criterionIdt(idt_y, real_x) * lambda_X * lambda_identity
            else:
                loss_idt_y = 0.

            loss_Y = loss_G_Y + loss_cycle_Y + loss_idt_y
            loss_Y.backward(retain_graph=True)
            optimizer_g.step()

            loss_Y_meter.add(loss_Y.item())


            ######################
            # netD_x
            ######################

            optimizer_d.zero_grad()

            # loss_real
            pred_real = netD_x(real_y)
            loss_D_x_real = criterionGAN(pred_real, True)
            score_Dx_real_y.add(float(pred_real.data.mean()))

            # loss_fake
            pred_fake = netD_x(fake_y)
            loss_D_x_fake = criterionGAN(pred_fake, False)
            score_Dx_fake_y.add(float(pred_fake.data.mean()))

            # loss and backward
            loss_D_x = (loss_D_x_real + loss_D_x_fake) * 0.5

            loss_D_x.backward()
            optimizer_d.step()


            ######################
            # netD_y
            ######################

            optimizer_d.zero_grad()

            # loss_real
            pred_real = netD_y(real_x)
            loss_D_y_real = criterionGAN(pred_real, True)

            # loss_fake
            pred_fake = netD_y(fake_x)
            loss_D_y_fake = criterionGAN(pred_fake, False)

            # loss and backward
            loss_D_y = (loss_D_y_real + loss_D_y_fake) * 0.5

            loss_D_y.backward()
            optimizer_d.step()

            # save snapshot
            if i % opt.plot_every == 0:
                filename = opt.name + '_snap_%03d_%05d.png' % (epoch, i,)
                test_path = os.path.join(opt.checkpoint_path, filename)
                tv.utils.save_image(fake_y, test_path, normalize=True)
                print(f'{filename} saved.')

                losses['loss_X'] = loss_X_meter.value()[0]
                losses['loss_Y'] = loss_Y_meter.value()[0]
                scores['score_Dx_real_y'] = score_Dx_real_y.value()[0]
                scores['score_Dx_fake_y'] = score_Dx_fake_y.value()[0]
                print(losses)
                print(scores)

            # print(f'iteration {i} finished')

        # save model
        if epoch % opt.save_every == 0 or epoch == opt.max_epochs - 1:
            save_filename = f'{opt.name}_netG_{epoch}.pth'
            save_filepath = os.path.join(opt.model_path, save_filename)
            t.save(netG_x.state_dict(), save_filepath)
            print(f'model saved as {save_filename}')

        # epoch end logs
        epoech_time = int(time.time() - epoch_start_time)

        print_options(opt, epoch_log=True, epoch=epoch, time=epoech_time, losses=losses, scores=scores)
        print()


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
           (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(
            state_dict, getattr(module, key), keys, i + 1)

def test(**kwargs):
    opt = Config()
    opt.phase = 'test'
    opt.preprocess = 'scale_width'
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.no_dropout = True
    opt.mode = 'test'

    device = t.device('cuda') if opt.gpu else t.device('cpu')

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'loaded {dataset_size} images for test.')

    netG_x = ResnetGenerator(opt)
    netG_x.to(device)
    print(netG_x)
    summary(netG_x, (3, 256, 256))

    models = sorted(os.listdir(opt.model_path))
    assert len(models) > 0, 'no models found!'
    latest_model = models[-1]
    model_path = os.path.join(opt.model_path, latest_model)
    print(f'loading trained model {model_path}')

    map_location = lambda storage, loc: storage
    state_dict = t.load(model_path, map_location=map_location)
    netG_x.load_state_dict(state_dict)

    for i, data in enumerate(dataset):
        real_x = data['A'].to(device)

        with t.no_grad():
            # fake_y = netG_x.forward(real_x)
            fake_y = netG_x(real_x)
            filename = opt.name + '_fake_%05d.png'%(i,)
            test_path = os.path.join(opt.test_path, filename)
            tv.utils.save_image(fake_y, test_path, normalize=True)
            print(f'{filename} saved')


def print_options(opt, epoch_log = False, epoch=0, time=0, losses=None, scores=None):
    file_name = os.path.join(opt.save_path, 'options.txt')
    if epoch_log:
        with open(file_name, 'a+') as opt_file:
            print(f'epoch {epoch} finished, cost time {time}s.')
            print(losses)
            print(scores)
            if epoch == 0:
                opt_file.write(f'Each epoch cost about {time}s.\n')
            opt_file.write(f'epoch {epoch} ')
            opt_file.write(str(losses)+' ')
            opt_file.write(str(scores)+'\n')
        return

    var_opt = vars(opt)
    message = f'\nTraning start time: {datetime.datetime.now()} \n\n'
    message += '----------------- Options ---------------\n'
    for key, value in var_opt.items():
        message += '{:>20}: {:<30}\n'.format(str(key), str(value))
    message += '----------------- End -------------------'
    message += '\n'
    print(message)
    # save to the disk
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


if __name__ == '__main__':
    import fire
    fire.Fire()