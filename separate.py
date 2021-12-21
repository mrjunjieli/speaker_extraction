import torch
from torch.serialization import save
from AVModel import AVModel
from DataLoader_M2Met import AudioDataset, AudioDataLoader, collate
import soundfile
import numpy as np
import os
from Loss import cal_si_snr
import librosa
import argparse
import torch.nn as nn 

sample_rate = 16000


def inference(mix_path, ref_path, target_path, model_path=None, output_path='./separate', save=False):
    '''
    save: 是否保存分离后的结果
    '''
    os.makedirs(output_path, exist_ok=True)
    model = AVModel()
    model = nn.DataParallel(model)
    checkpoint = torch.load(model_path)
    model_dict = checkpoint['model']

    model.load_state_dict(model_dict)
    model.cuda()
    model.eval()

    mix_data, _ = librosa.load(mix_path, sr=sample_rate)
    mix_data = mix_data.astype(np.float32)
    mix_data = np.expand_dims(mix_data, 0)
    mix_data = np.expand_dims(mix_data, 0)
    mix_data = torch.from_numpy(mix_data)

    target_data, _ = librosa.load(target_path, sr=sample_rate)
    target_data = target_data.astype(np.float32)
    target_data = np.expand_dims(target_data, 0)
    target_data = np.expand_dims(target_data, 0)
    target_data = torch.from_numpy(target_data)

    ref_data, _ = librosa.load(ref_path, sr=sample_rate)
    ref_data = ref_data.astype(np.float32)
    ref_data = np.expand_dims(ref_data, 0)
    ref_data = np.expand_dims(ref_data, 0)

    if ref_data.shape[-1]>3*sample_rate:
        ref_data = ref_data[:,:, 0:3*sample_rate]  # 参考信号只取3s
    else:
        ref_data = np.pad(ref_data,((0,0),(0,0),(0,3*sample_rate-ref_data.shape[-1])),"constant")

    # print(ref_data.shape)
    ref_data = torch.from_numpy(ref_data)

    mix_data = mix_data.cuda()
    target_data = target_data.cuda()
    ref_data = ref_data.cuda()
    # print(mix_data.shape)
    # print(target_data.shape)


    with torch.no_grad():
        est, s1_emb = model([mix_data, ref_data])
        loss = cal_si_snr(target_data, est)
        # AVG_SISNR += loss
    print(mix_path, 'SI_SNR:%.04f' % loss)

    if save:
        est = est[0].cpu().permute(1, 0).numpy()
        est/= np.max(np.abs(est))
        name = mix_path.strip().split('/')[-1].split('.wav')[0]
        soundfile.write(os.path.join(output_path, name +
                        '_est.wav'), est, samplerate=sample_rate)
        print('save', str(os.path.join(output_path, name+'_est.wav')))
    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser('AVConv-TasNet')

    parser.add_argument('--data_scp_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str, default='./separate')

    args = parser.parse_args()
    num = 0
    AVG_SISNR = 0
    with open(args.data_scp_path, 'r') as p:
        for line in p.readlines():
            mix, target, ref = line.strip().split(' ')
            num += 1
            AVG_SISNR+= inference(mix, ref, target, model_path=args.model_path,
                      output_path=args.output_path, save=True)

    print('total SI_SNR:%.04f' % (AVG_SISNR/num))
