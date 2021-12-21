

import random
import os

def generate_speaker(wav_scp_path,output):
    speakerid_list = set()
    with open(wav_scp_path, 'r') as p:
        for line in p.readlines():
            speaker = line.strip().split(' ')[0].split('-')[2]
            speaker = speaker.split('_')[-2:]
            speaker = '_'.join(speaker)
            speakerid_list.add(speaker)
    speakerid_list = sorted(speakerid_list)
    with open(os.path.join(output,'speaker_name_forAll.txt'), 'w') as t:
        for line in speakerid_list:
            t.write(str(line)+'\n')
    print('finish generate speakerid!!!')
    print('total number of speaker is %d'%len(speakerid_list))
    print('-----------')


def split_train_dev(wav_scp_path,output, rate=0.8):
    total_list = []
    with open(wav_scp_path, 'r') as p:
        for line in p.readlines():
            total_list.append(line)
    train_list = []
    train_speaker_list = set()
    dev_list = []
    for line in total_list:
        number = random.uniform(0, 1)
        speaker = line.strip().split(' ')[0].split('-')[2]
        speaker = speaker.split('_')[-2:]
        speaker = '_'.join(speaker)
        if speaker not in train_speaker_list:
            train_list.append(line)
            train_speaker_list.add(speaker)
        else:
            if number<=0.9:
                train_list.append(line)
            else:
                dev_list.append(line)

    # train_list = random.sample(total_list, int(rate*len(total_list)))
    # dev_list = list(set(total_list).difference(set(train_list)))
    # print(len(train_speaker_list))
    with open(os.path.join(output,'train.scp'), 'w') as t:
        for line in sorted(train_list):
            t.write(line)
    with open(os.path.join(output,'dev.scp'), 'w') as d:
        for line in sorted(dev_list):
            d.write(line)

    print('finish split train and dev list !!!')
    print('the number of train list :%d'%len(train_list))
    print('the number of dev list :%d'%len(dev_list))
    print('------------')


def generate_noise(path,output):
    noise_list = [x for x in os.listdir(path) if '.wav'in x]

    with open(os.path.join(output,'noise.scp'),'w') as p:
        for line in noise_list:
            p.write(str(os.path.join(path,line))+'\n')

    print('finish generate noise list !!!')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('M2met')
    parser.add_argument('--wav_scp_path',type=str,help='the path of wav_scp')
    parser.add_argument('--noise_data',type=str,help='the path of noise_data')
    parser.add_argument('--output_dir',type=str,default='./data/M2Met')
    args = parser.parse_args()

    # wav_scp_path = '/CDShare2/M2MeT_codes/espnet/egs2/AliMeeting/asr/dump2/raw/org/Train_Ali_near_nooverlap_onechannel/wav.scp'
    # wav.scp 单通道 nonoverlap 近场 数据   
    # noise_data = '/CDShare2/OpenSLR/28/RIRS_NOISES/pointsource_noises'
    # noise_data数据来源 https://www.openslr.org/28/ 

    os.makedirs(args.output_dir,exist_ok=True)

    #step1 生成speaker名称列表
    generate_speaker(args.wav_scp_path,args.output_dir) 
    #step2 分割训练集 验证集
    split_train_dev(args.wav_scp_path,args.output_dir,rate=0.9)  
    #step3 生成对应噪声列表
    generate_noise(args.noise_data,args.output_dir)  

    print('FINISH DATA PREPARATION!')

