import time
import torch
from Loss import cal_si_snr
# from DataLoader import AudioDataset, AudioDataLoader
from DataLoader_M2Met import AudioDataset, AudioDataLoader,collate
import os
import torch.nn as nn


class Solver(object):
    def __init__(self, args, model, use_gpu, optimizer, logger):
        audio_data_trainval = AudioDataset(s1_scp='./data/M2Met/dev.scp',stage='dev')

        audio_data_pretrain = AudioDataset(s1_scp='./data/M2Met/train.scp',stage='train')
        # audio_data_trainval = AudioDataset(s1_ref='./data/audio/trainval_s1_aux.scp',s1_scp='./data/audio/trainval_s1.scp',
        #                 mix_scp='./data/audio/trainval_mix.scp',s2_ref='./data/audio/trainval_s2_aux.scp',s2_scp='./data/audio/trainval_s2.scp')

        # audio_data_pretrain = AudioDataset(s1_ref='./data/audio/pretrain_s1_aux.scp',s1_scp='./data/audio/pretrain_s1.scp',
                        # mix_scp='./data/audio/pretrain_mix.scp',s2_ref='./data/audio/pretrain_s2_aux.scp',s2_scp='./data/audio/pretrain_s2.scp')

        # pretrain data
        self.audio_data_loader = AudioDataLoader(
            audio_data_pretrain,batch_size=args.batch_size, collate_fn=collate ,num_workers=args.num_workers, pin_memory=True,shuffle=True,persistent_workers=True,prefetch_factor=5)
        self.pretrain_len = len(audio_data_pretrain)
        # trainval data
        self.audio_trainval = AudioDataLoader(
            audio_data_trainval, batch_size=args.batch_size,collate_fn=collate, num_workers=args.num_workers, pin_memory=True,persistent_workers=True,prefetch_factor=5)
        self.trainval_len = len(audio_data_trainval)

        self.args = args
        self.model = model
        self.use_gpu = use_gpu
        self.optimizer = optimizer
        self.logger = logger

        self._rest()
        self.logger.info('learning rate:'+str(self.optimizer.state_dict()['param_groups'][0]['lr']))

    def _rest(self):
        self.halving = False
        if str(self.args.continue_from)!='None':
           
            checkpoint_name = str(self.args.continue_from)
            checkpoint = torch.load('./log/'+str(self.args.data_set_name)+'/model/'+checkpoint_name)

            # load model
            model_dict = self.model.state_dict()
            pretrained_model_dict = checkpoint['model']
            pretrained_model_dict = {
                k: v for k, v in pretrained_model_dict.items() if k in model_dict}
            model_dict.update(pretrained_model_dict)
            self.model.load_state_dict(model_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("*** model "+checkpoint_name +
                             " has been successfully loaded! ***")
            # load other params
            self.start_epoch = checkpoint['epoch']
            self.best_val_sisnr = checkpoint['best_val_sisnr']
            self.val_no_impv = checkpoint['val_no_impv']
            

        else:
            self.start_epoch = 0
            self.best_val_sisnr = float('inf')
            self.val_no_impv = 0
            self.pre_val_sisnr = float("inf")
            self.logger.info("*** train from scratch ***")

    def train(self):
        self.logger.info("use SI_SNR and speakerLoss as loss function")
        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.logger.info("------------")
            self.logger.info("Epoch:%d/%d" % (epoch, self.args.num_epochs))
            # train
            # --------------------------------------
            start = time.time()
            self.model.train()
            temp = self._run_one_epoch(self.audio_data_loader, state='train')
            tr_loss_sisnr = temp['si_snr']
            acc = temp['acc']/self.pretrain_len
            # tr_loss_snr = temp['snr']
            tr_loss_sisnr = tr_loss_sisnr/self.pretrain_len
            speaker_loss = temp['speaker_loss']/self.pretrain_len
            # tr_loss_snr = tr_loss_snr/self.pretrain_len
            end = time.time()
            self.logger.info("Train: SI_SNR=%.04f,speaker_loss=%.04f,acc=%0.04f,Time:%d minutes" %
                             (-tr_loss_sisnr, speaker_loss, acc,(end-start)//60))

            # validation
            # --------------------------------------
            start = time.time()
            self.model.eval()
            with torch.no_grad():
                temp = self._run_one_epoch(self.audio_trainval, state='val')
                # temp= {'si_snr':0,'speaker_loss':0}
                val_loss_sisnr = temp['si_snr']
                acc = temp['acc']/self.trainval_len
                # val_loss_snr = temp['snr']
                val_loss_sisnr = val_loss_sisnr/self.trainval_len
                speaker_loss = temp['speaker_loss']/self.trainval_len
                # val_loss_snr/=self.trainval_len
            end = time.time()
            self.logger.info("Val: SI_SNR=%.04f,speaker_loss=%.04f,acc=%.04f,Time:%d minutes" %
                             (-val_loss_sisnr, speaker_loss, acc,(end - start) // 60))
            self.logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            # check whether to adjust learning rate and early stop
            # -------------------------------------
            if val_loss_sisnr >= self.best_val_sisnr:
                self.val_no_impv += 1
                if self.val_no_impv >= 3:
                    self.halving = True
                if self.val_no_impv >= 6:
                    self.logger.info(
                        "No improvement for 6 epoches in val dataset, early stop")
                    break
            else:
                self.val_no_impv = 0

            # half the learning rate
            # -----------------------------------
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr']/2
                self.optimizer.load_state_dict(optim_state)
                self.logger.info("**learning rate is adjusted from [%f] to [%f]"
                                 % (optim_state['param_groups'][0]['lr']*2, optim_state['param_groups'][0]['lr']))
                self.halving = False

            # self.pre_val_sisnr = val_loss_sisnr

            # save the model
            # ----------------------------------
            if val_loss_sisnr < self.best_val_sisnr:
                self.best_val_sisnr = val_loss_sisnr
                checkpoint = {'model': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict(),
                              'epoch': epoch+1,
                              'best_val_sisnr': self.best_val_sisnr,
                              'val_no_impv': self.val_no_impv}
                os.makedirs('./log/'+str(self.args.data_set_name)+'/model/',exist_ok=True)
                torch.save(
                    checkpoint, "./log/"+str(self.args.data_set_name)+"/model/Checkpoint_%04d.pt" % epoch)
                self.logger.info(
                    "***save checkpoint as Checkpoint_%04d.pt***" % epoch)

    def _run_one_epoch(self, audio_data_loader, state='train'):
        epoch_loss = {'si_snr': 0, 'speaker_loss': 0,'acc':0}
        length = len(audio_data_loader)
        for idx,audio in enumerate(audio_data_loader):
            audio_mix = audio['mix'].float()
            audio_s1 = audio['s1'].float()

            audio_s1_aux = audio['ref'].float()

            s1_id = audio['s1_id']
            

            if self.use_gpu:
                audio_mix = audio_mix.cuda()
                audio_s1 = audio_s1.cuda()

                audio_s1_aux = audio_s1_aux.cuda()
                s1_id = s1_id.cuda()
            # print(audio_mix.size())
            # print(audio_s1.size())
            audio_est_s1,s1_emb = self.model([audio_mix,audio_s1_aux])
            # audio_est_s2, s2_emb = self.model([audio_mix, audio_s2_aux])
            ce = nn.CrossEntropyLoss(reduction='sum')
            
            loss = cal_si_snr(audio_s1,audio_est_s1)
   
            # loss += cal_si_snr(audio_s2,audio_est_s2)
            loss = -loss  # return negative number

            ce_loss = ce(s1_emb, s1_id)
            # ce_loss += ce(s2_emb, s2_id)
            temp = s1_emb.detach()
            est_speaker_id = temp.argmax(dim=-1)
            truth_id = s1_id.detach()
            sum = torch.sum(est_speaker_id==truth_id).item()
            
            
            epoch_loss['acc']+=sum
            epoch_loss['si_snr'] += loss.item()
            epoch_loss['speaker_loss'] += ce_loss.item()

            total_loss = loss +0.5*ce_loss

            if state == 'train':
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if idx %100==0:
                    self.logger.info('processed batch:%d/%d'%(idx,length))

        return epoch_loss
