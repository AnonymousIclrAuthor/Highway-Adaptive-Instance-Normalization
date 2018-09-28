import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from data_loader import *
from model import *
import time
import datetime
import os
from utils.util import *
from torch.backends import cudnn


class Run(object):
    def __init__(self, config):
        if config['DATASET'] == 'CelebA':
            self.celeba_loader = get_loader(config['CELEBA_PATH'], config['ATTR_PATH'], config['SELECTED_ATTRS'],
                                            config['CELEBA_CROP_SIZE'], config['IMG_SIZE'], config['BATCH_SIZE'],
                                            config['DATASET'], config['MODE'], config['NUM_WORKERS'])

        self.config = config
        self.device = torch.device("cuda:%d" % (int(config['GPU1'])) if torch.cuda.is_available() else "cpu")
        self.make_dir()
        self.init_network()
        self.loss = {}

        if config['LOAD_MODEL']:
            self.load_pretrained_model()

    def make_dir(self):
        if not os.path.exists(self.config['MODEL_SAVE_PATH']):
            os.makedirs(self.config['MODEL_SAVE_PATH'])

    def init_network(self):
        """Create a generator and a discriminator."""
        G_opts = self.config['G']
        D_opts = self.config['D']
        if self.config['DATASET'] in ['CelebA']:
            self.G = Generator(G_opts['FIRST_DIM'], G_opts['N_RES_BLOCKS'], G_opts['STYLE_DIM'], 
                               G_opts['MLP_DIM'], self.config['C_DIM'])
            self.D = Discriminator(self.config['IMG_SIZE'], D_opts['FIRST_DIM'], self.config['C_DIM'], D_opts['N_RES_BLOCKS']) 

        G_params = list(self.G.parameters()) # + list(blah)
        
        self.G_optimizer = torch.optim.Adam(G_params, self.config['G_LR'], [self.config['BETA1'], self.config['BETA2']])
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), self.config['D_LR'], [self.config['BETA1'], self.config['BETA2']])
        
        self.G_scheduler = get_scheduler(self.G_optimizer, config)
        self.D_scheduler = get_scheduler(self.D_optimizer, config)

        self.G.apply(weights_init(self.config['INIT']))
        self.D.apply(weights_init('gaussian'))
        # print_network(self.G, 'G')
        # print_network(self.D, 'D')

        self.set_gpu()

    def set_gpu(self):
        def multi_gpu(gpu1, gpu2, model):
            model = nn.DataParallel(model, device_ids=[gpu1, gpu2])
            return model

        gpu1 = int(self.config['GPU1'])
        gpu2 = int(self.config['GPU2'])
        if self.config['DATA_PARALLEL']:
            self.G = multi_gpu(gpu1, gpu2, self.G)
            self.D = multi_gpu(gpu1, gpu2, self.D)

        self.G.to(self.device)
        self.D.to(self.device)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def mask_criterion(self, masks, c):
        # max: 1.0 ~ 
        def get_similarity(x, eps=1e-8):
            x = x.view(x.size(0),x.size(1),-1) # b,c,wh
            l2_norm_x = torch.clamp(torch.sum(x**2, dim=1, keepdim=True)**0.5, min=eps) # b,1,wh
            x = x / l2_norm_x
            x_T = x.permute(0,2,1) # b,wh,c
            return torch.bmm(x_T, x) # b,wh,wh
            # return F.cosine_similarity(x,x)
        def get_dist(m):
            m_col = m.view(m.size(0), -1, 1) # b, wh, 1
            m_row = m.view(m.size(0), 1, -1) # b, 1, wh
            x_col = m_col.expand(m_col.size(0),m_col.size(1),m_col.size(1))
            x_row = m_row.expand(m_row.size(0),m_row.size(2),m_row.size(2))
            return torch.abs(x_row - x_col)

        mean = 0.
        smooth = 0.
        sim = 0.

        similarity = get_similarity(c)

        for i, mask in enumerate(masks):
            dist = get_dist(mask)
            
            sim = sim + torch.mean(similarity * dist)
            smooth = smooth + torch.sum(torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:])) + \
                     torch.sum(torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]))
            mean = mean + torch.mean(mask)
        return (sim, mean, smooth)

    def l1_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def cls_criterion(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset in ['CelebA']: # multihot
            return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)

    def model_save(self, iteration):
        self.G = self.G.cpu()
        self.D = self.D.cpu()

        torch.save(self.G.state_dict(),
            os.path.join(self.config['MODEL_SAVE_PATH'], 'G_exemplar_%s.pth' % (self.config['SAVE_NAME'])))
        torch.save(self.D.state_dict(),
            os.path.join(self.config['MODEL_SAVE_PATH'], 'D_exemplar_%s.pth' % (self.config['SAVE_NAME'])))
        
        self.set_gpu()

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.config['MODEL_SAVE_PATH'], 'G_exemplar_%s.pth' % (self.config['SAVE_NAME']))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.config['MODEL_SAVE_PATH'], 'D_exemplar_%s.pth' % (self.config['SAVE_NAME']))))

    def update_learning_rate(self):
        if self.G_scheduler is not None:
            self.G_scheduler.step()
        if self.D_scheduler is not None:
            self.D_scheduler.step()

    def train_ready(self):
        self.G.train()
        self.D.train()

    def test_ready(self):
        self.G.eval()
        self.D.eval()

    def concat_input(self, c, s):
        s_ = s.expand([s.size(0)]+[s.size(1)]+[c.size(2)]+[c.size(3)])
        return torch.cat([c, s_], dim=1)

    def masked_img(self, x, m):
        return F.interpolate(m, None, 4, 'bilinear', align_corners=False) * x

    def update_G(self, x_A, x_B, label_A, label_B, isTrain=True):

        G = self.G.module if self.config['DATA_PARALLEL'] else self.G

        '''
        ### 1st stage
        '''
        c_A = G.c_encoder(x_A)
        c_B = G.c_encoder(x_B)

        # get mask
        m_A = G.mask(c_A, c_B)
        m_B = G.mask(c_B, c_A)

        # get style
        s_fA = G.s_encoder(self.masked_img(x_A, m_A), isf=True)
        s_bA = G.s_encoder(self.masked_img(x_A, 1-m_A), isf=False)
        s_fB = G.s_encoder(self.masked_img(x_B, m_B), isf=True)
        s_bB = G.s_encoder(self.masked_img(x_B, 1-m_B), isf=False)

        # from A to B 
        x_AB = G.decoder(c_A, m_A, s_fB, s_bA)

        # from B to A
        x_BA = G.decoder(c_B, m_B, s_fA, s_bB)

        src_AB, cls_AB = self.D(x_AB)
        src_BA, cls_BA = self.D(x_BA)

        '''
        ### 2nd stage
        '''
        c_AB = G.c_encoder(x_AB)
        c_BA = G.c_encoder(x_BA)

        m_AB = G.mask(c_AB, c_BA)
        m_BA = G.mask(c_BA, c_AB)

        s_fAB = G.s_encoder(self.masked_img(x_AB, m_AB), isf=True)
        s_bAB = G.s_encoder(self.masked_img(x_AB, 1-m_AB), isf=False)
        s_fBA = G.s_encoder(self.masked_img(x_BA, m_BA), isf=True)
        s_bBA = G.s_encoder(self.masked_img(x_BA, 1-m_BA), isf=False)

        # from AB to A
        x_ABA = G.decoder(c_AB, m_AB, s_fBA, s_bAB)

        # from BA to B
        x_BAB = G.decoder(c_BA, m_BA, s_fAB, s_bBA)

        m_AA = G.mask(c_A, c_A)
        m_BB = G.mask(c_B, c_B)

        # from A to A
        x_AA = G.decoder(c_A, m_AA, s_fA, s_bA) # sA_mask_*s_A
        
        # from B to B
        x_BB = G.decoder(c_B, m_BB, s_fB, s_bB)

        g_loss_fake = - (torch.mean(src_AB) + torch.mean(src_BA))
        g_loss_cls = self.cls_criterion(cls_AB, label_B) + self.cls_criterion(cls_BA, label_A)

        loss_cross_rec = self.l1_criterion(x_ABA, x_A) + self.l1_criterion(x_BAB, x_B)
        loss_ae_rec = self.l1_criterion(x_AA, x_A) + self.l1_criterion(x_BB, x_B)

        loss_cross_s = self.config['LAMBDA_S_B']*(self.l1_criterion(s_fAB, s_fB) + self.l1_criterion(s_fBA, s_fA)) + \
                        self.config['LAMBDA_S_F']*(self.l1_criterion(s_bAB, s_bA) + self.l1_criterion(s_bBA, s_bB))
        loss_cross_c = self.l1_criterion(c_AB, c_A) + self.l1_criterion(c_BA, c_B)

        mask_sim_A, mask_min_A, mask_smooth_A = self.mask_criterion([m_A, m_AB, m_AA], c_A.detach())
        mask_sim_B, mask_min_B, mask_smooth_B = self.mask_criterion([m_B, m_BA, m_BB], c_B.detach())

        g_loss_mask_sim, g_loss_mask_min, g_loss_mask_smooth = (
                                                                    mask_sim_A + mask_sim_B, 
                                                                    mask_min_A + mask_min_B,
                                                                    mask_smooth_A + mask_smooth_B
                                                                )
        style_reg = -self.l1_criterion(s_fA, s_fB)

        # Backward and optimize.
        g_loss = g_loss_fake + \
                 self.config['LAMBDA_X_REC'] * (loss_ae_rec) + \
                 self.config['LAMBDA_X_CYC'] * loss_cross_rec + \
                 self.config['LAMBDA_CLS'] * g_loss_cls + \
                 self.config['LAMBDA_LATENT_REC'] * (loss_cross_c + loss_cross_s) + \
                 self.config['LAMBDA_MASK_MIN'] * (g_loss_mask_min) + \
                 self.config['LAMBDA_MASK_SMOOTH'] * g_loss_mask_smooth + \
                 self.config['LAMBDA_MASK_SIM'] * g_loss_mask_sim

        if isTrain:
            self.G_optimizer.zero_grad()
            g_loss.backward()
            self.G_optimizer.step()

        # Logging.
        self.loss['G/loss_fake'] = g_loss_fake.item()
        self.loss['G/loss_cross_rec'] = loss_cross_rec.item() * self.config['LAMBDA_X_REC']
        self.loss['G/loss_ae_rec'] = loss_ae_rec.item() * self.config['LAMBDA_X_REC']
        self.loss['G/loss_cross_c'] = loss_cross_c.item() * self.config['LAMBDA_LATENT_REC']
        self.loss['G/loss_cross_s'] = loss_cross_s.item() * self.config['LAMBDA_LATENT_REC']
        self.loss['G/D_loss_cls'] = g_loss_cls.item() * self.config['LAMBDA_CLS']
        self.loss['G/loss_mask'] = (g_loss_mask_min.item()) * self.config['LAMBDA_MASK_MIN']
        self.loss['G/loss_mask_smooth'] = g_loss_mask_smooth.item() * self.config['LAMBDA_MASK_SMOOTH']
        self.loss['G/loss_style_reg'] = style_reg.item() * self.config['LAMBDA_STYLE_REG']
        self.loss['G/loss_mask_sim'] = g_loss_mask_sim.item() * self.config['LAMBDA_MASK_SIM']

        return (x_AB, x_BA, x_ABA, x_AA, m_A, m_B, s_fA, s_fB, s_bA, s_bB, c_A, c_B)

    def update_D(self, x_A, x_B, label_A, d_loss=0, update=False):

        # Compute loss with real images
        out_src, out_src_cls = self.D(x_A)
        d_loss_real = - torch.mean(out_src)
        d_loss_cls = self.cls_criterion(out_src_cls, label_A)

        # Compute loss with fake images.
        x_AB = self.G(x_A, x_B)
        out_src, _ = self.D(x_AB.detach())
        d_loss_fake = torch.mean(out_src)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_A.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_A.data + (1 - alpha) * x_AB.data).requires_grad_(True)
        out_src, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)

        d_loss = d_loss + d_loss_real + d_loss_fake + \
                 self.config['LAMBDA_CLS'] * d_loss_cls + \
                 self.config['LAMBDA_GP'] * d_loss_gp

        if update:
            self.D_optimizer.zero_grad()
            d_loss.backward()
            self.D_optimizer.step()

        else:
            return d_loss

        self.loss['D/loss_real'] = d_loss_real.item()
        self.loss['D/loss_fake'] = d_loss_fake.item()
        self.loss['D/loss_cls'] = d_loss_cls.item() * self.config['LAMBDA_CLS']
        self.loss['D/loss_gp'] = d_loss_gp.item() * self.config['LAMBDA_GP']


    def train(self):

        if self.config['DATASET'] == 'CelebA':
            data_loader = self.celeba_loader

        print('# iters: %d' % (len(data_loader)))
        print('# data: %d' % (len(data_loader)*self.config['BATCH_SIZE']))
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        self.train_ready()
        print("Start training ~ Ayo:)!")
        start_time = time.time()

        
        for i in range(self.config['START'], self.config['NUM_ITERS']):

        ### Preprocess input data ###
            # Fetch real images and labels.
            try:
                x_A, label_A = next(data_iter)
                if x_A.size(0) != self.config['BATCH_SIZE']:
                    x_A, label_A = next(data_iter)
                x_B, label_B = next(data_iter)
                if x_B.size(0) != self.config['BATCH_SIZE']:
                    x_B, label_B = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_A, label_A = next(data_iter)
                if x_A.size(0) != self.config['BATCH_SIZE']:
                    x_A, label_A = next(data_iter)
                x_B, label_B = next(data_iter)
                if x_B.size(0) != self.config['BATCH_SIZE']:
                    x_B, label_B = next(data_iter)
            

            x_A = x_A.to(self.device)   # Input images.
            x_B = x_B.to(self.device)   # Exemplar images corresponding with target labels.

            label_A = label_A.to(self.device)     # Labels for computing classification loss.
            label_B = label_B.to(self.device)     # Labels for computing classification loss.

        ### Training ###    
            d_loss = self.update_D(x_A, x_B, label_A)
            self.update_D(x_B, x_A, label_B, d_loss=d_loss, update=True)
            if i % self.config['N_CRITIC'] == 0:
                x_AB, x_BA, x_ABA, x_AA, m_A, m_B, s_fA, s_fB, s_bA, s_bB, c_A, c_B = \
                self.update_G(x_A, x_B, label_A, label_B)

        ### ETC ###
            if i % self.config['PRINT_EVERY'] == 0:

                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                print('=====================================================')
                print("Elapsed [{}], Iter [{}/{}]".format(
                    elapsed, i+1, self.config['NUM_ITERS']))
                print('=====================================================')
                # print('D/loss_real: %.5f' % (self.loss['D/loss_real']))
                # print('D/loss_fake: %.5f' % (self.loss['D/loss_fake']))
                print('D/loss_cls: %.5f' % (self.loss['D/loss_cls']))
                # print('D/loss_gp: %.5f' % (self.loss['D/loss_gp']))
                # print('G/loss_fake: %.5f' % (self.loss['G/loss_fake']))
                # print('G/loss_cross_rec: %.5f' % (self.loss['G/loss_cross_rec']))
                # print('G/loss_ae_rec: %.5f' % (self.loss['G/loss_ae_rec']))
                print('G/D_loss_cls: %.5f' % (self.loss['G/D_loss_cls']))
                print('G/loss_mask: %.5f' % (self.loss['G/loss_mask']))
                print('G/loss_mask_smooth: %.5f' % (self.loss['G/loss_mask_smooth']))
                print('G/loss_style_reg: %.5f' % (self.loss['G/loss_style_reg']))
                print('G/loss_mask_sim: %.10f' % (self.loss['G/loss_mask_sim']))

                x_fA = self.masked_img(x_A, m_A)
                x_fB = self.masked_img(x_B, m_B)
                x_bA = self.masked_img(x_A, 1-m_A)
                x_bB = self.masked_img(x_B, 1-m_B)

                show_output(x_A, x_B, x_AB, x_BA, m_A, m_B, x_fA, x_fB, x_bA, x_bB, label_A, label_B, 
                            self.config['SELECTED_ATTRS'], self.config['NUM_PRINT'])
                
                self.model_save(i+1)

            if i > self.config['NUM_ITERS_DECAY']:
                self.update_learning_rate()


    def test(self):
        print("test start")
        self.load_pretrained_model()

        if self.config['DATASET'] == 'CelebA':
            data_loader = self.celeba_loader

        x_A = 0
        label_A = 0

        with torch.no_grad():
            for i, (x, label) in enumerate(data_loader):
                if i%2 == 0:
                    x_A = x
                    label_A = label
                    continue
                else:
                    x_B = x
                    label_B = label

                x_A = x_A.to(self.device)
                x_B = x_B.to(self.device)

                # x_rec = x_rec.to(self.device)
                label_A = label_A.to(self.device)
                label_B = label_B.to(self.device)

                x_AB, x_BA, x_ABA, x_AA, m_A, m_B, s_fA, s_fB, s_bA, s_bB, c_A, c_B = \
                self.update_G(x_A, x_B, label_A, label_B, isTrain=False)

                x_fA = self.masked_img(x_A, m_A)
                x_fB = self.masked_img(x_B, m_B)
                x_bA = self.masked_img(x_A, 1-m_A)
                x_bB = self.masked_img(x_B, 1-m_B)

                show_output(x_A, x_B, x_AB, x_BA, m_A, m_B, x_fA, x_fB, x_bA, x_bB, label_A, label_B, 
                        self.config['SELECTED_ATTRS'], self.config['NUM_PRINT'])


def main():
    
    # For fast training
    cudnn.benchmark = True

    run = Run(config)
    if config['MODE'] == 'train':
        run.train()
    else:
        run.test()

config = ges_Aonfig('configs/config_celebA.yaml')
main()