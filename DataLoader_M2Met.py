import librosa
from numpy.core.defchararray import startswith 
import torch
from torch.serialization import load
from torch.utils.data import Dataset,DataLoader
import numpy as np 
import random
# from simulation_room import simulation
import soundfile as sf
import torch.nn as nn
from Loss import cal_si_snr



def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split(" ",1)
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, scp_path))

        scp_dict[key] = value

    return scp_dict


def load_audio(index_dict,index,timeLen=3,sr=16000):
    '''
    load audio data
    '''
    keys = list(index_dict.keys())
    key=''
    if type(index) not in [int, str]:
        raise IndexError('Unsupported index type: {}'.format(type(index)))
    elif type(index) ==int:
        num_uttrs = len(index_dict)
        if(num_uttrs<=index or index <0):
            raise KeyError('Interger index out of range,suppose to get 0 to {:d} \but get {:d}'.format(num_uttrs-1,index))
        key = keys[index]
    else:
        key = index 
    # audio_data,_ = librosa.load(index_dict[key],sr=sr) 
    audio_data,_ = sf.read(index_dict[key])
    audio_data = audio_data.astype(np.float32)

    # audio_data,_ = librosa.load(index_dict[key],sr=sr) 
    # audio_data = np.expand_dims(audio_data,axis=0)
    # audio_data = torch.tensor(audio_data,dtype=torch.float).unsqueeze(0)

    return audio_data 

def get_allspeakername(speakername_file='./data/M2Met/speaker_name_forAll.txt'):
    speakername_list = []
    with open(speakername_file, 'r') as f:
        for item in sorted(f):
            speakername_list.append(item.strip()) #去掉\n
    return speakername_list



class AudioDataset(Dataset):
    '''
    Load audio data
    batch_size=:
    shuffle=:
    num_works=:
    '''
    def __init__(self,s1_scp=None,stage='train',sr=16000):
        super(AudioDataset,self).__init__()
        self.audio = handle_scp(s1_scp)
        self.sr=sr
        self.speakername = get_allspeakername()
        self.speaker_audio = {}
        self.stage = stage
        self.noise = []
        with open('./data/M2Met/noise.scp','r') as p:
            for line in p.readlines():
                self.noise.append(line.strip())

        for sp in self.audio.keys():
            spea = sp.strip().split(' ')[0]
            speaker = spea.split('-')[2]
            speaker = speaker.split('_')[-2:]
            speaker = '_'.join(speaker)
            if speaker not in self.speaker_audio:
                self.speaker_audio[speaker] = [sp]
            else:
                self.speaker_audio[speaker].append(sp)

    def __len__(self):
        return len(self.audio)
    def __getitem__(self,index):
        if self.stage=='train':
            noise_scale = random.uniform(0,0.5)
            noise_path = random.choice(self.noise)
        if self.stage=='dev':
            noise_scale=0.1
            noise_path = self.noise[index%len(self.noise)]
        noise_scale=1
        noise_data,_ = librosa.load(noise_path,sr=self.sr)
        noise_length = len(noise_data)
        noise_data = noise_data[int(random.uniform(0,0.5)*noise_length):int(random.uniform(0.5,1)*noise_length)]
        noise_data = np.expand_dims(noise_data,0)
        
        noise_data *=noise_scale

        channel_num = random.choice([0,1,2,3,4,5,6,7])
        # channel_num=0
        s1_audio = load_audio(self.audio,index,sr=self.sr)
        s1_audio = s1_audio[:,channel_num]
        s1_audio = np.expand_dims(s1_audio,0)

        if s1_audio.shape[1]/self.sr>4:
            s1_audio = s1_audio[:,0:self.sr*4]

        s1_id = self._returnspeakerName(self.audio,index) #[0]name  [1]才是真实的id
        s1_name = s1_id[0].strip().split(' ')[0].split('-')[2].split('_')[-2:]
        s1_name = '_'.join(s1_name)

        if self.stage=='train':
            ref_path = random.choice(self.speaker_audio[s1_name])
        elif self.stage=='dev': 
            ref_path = self.speaker_audio[s1_name][0]

        ref_audio,_ = sf.read(self.audio[ref_path])
        if self.stage=='train':
            ref_audio = ref_audio[:,random.choice([0,1,2,3,4,5,6,7])]
        else:
            ref_audio = ref_audio[:,0]
        ref_audio = np.expand_dims(ref_audio,axis=0)

        if self.stage=='train':
            num = random.choice([1,2,3])
            speaker_list = list(set(self.speaker_audio.keys())-set([s1_name]))
            speakers= random.sample(speaker_list,num)
            end_length = int(s1_audio.shape[1]*random.uniform(0,0.05))
            start_length = int(s1_audio.shape[1]*random.uniform(0,0.05))
        elif self.stage =='dev':
            speaker_list = list(self.speaker_audio.keys())
            speaker_index = speaker_list.index(s1_name)
            speaker_list = list(set(self.speaker_audio.keys())-set([s1_name]))
            num = speaker_index%2 +1
            speakers = []
            while(num>0):
                speakers.append(speaker_list[(speaker_index+num)%len(speaker_list)])
                num-=1

            end_length=0
            start_length=0

        mix_audio = np.zeros((1,start_length+s1_audio.shape[1]+end_length))

        target_audio = np.pad(s1_audio,((0,0),(start_length,end_length)),"constant")
        
        if self.stage=='train':
            target_audio = random.uniform(0.5,1.2) *target_audio

        mix_audio += target_audio
        length = mix_audio.shape[1]

        if noise_data.shape[1]>length:
            mix_audio+= noise_data[:,0:length]
        else:
            mix_audio[:,0:noise_data.shape[1]] += noise_data

        for speaker in speakers:
            if self.stage=='train':
                path = random.choice(self.speaker_audio[speaker])
                start_ = int(length*random.uniform(0,0.2))
            if self.stage=='dev':
                path = self.speaker_audio[speaker][0]
                start_=0
            path = self.audio[path]
            audio,_ = sf.read(path)
            audio = audio[:,channel_num]
            audio = np.expand_dims(audio,axis=0)
            
            left_length = length-start_

            if audio.shape[1]>left_length:
                mix_audio[:,start_:] += audio[:,0:length-start_]
            else:
                mix_audio[:,start_:audio.shape[1]+start_] += audio[:,:]

        return{
            's1':target_audio,
            'mix':mix_audio,
            'ref':ref_audio,
            's1_id':s1_id[1]
        }
    def _returnspeakerName(self,index_dict,index):
        keys = list(index_dict.keys())
        # spea = keys[index].split("_")[0][0:3]
        spea = keys[index].strip().split(' ')[0]
        speaker = spea.split('-')[2]
        speaker = speaker.split('_')[-2:]
        speaker = '_'.join(speaker)
        # print(spea)
        return spea,self.speakername.index(speaker)


def collate(batch):
    chunk = {'mix':[],'s1':[],'ref':[],'s1_id':[]}
    max_length = 0
    for eg in batch:
        if eg['mix'].shape[1]>max_length:
            max_length = eg['mix'].shape[1]
        chunk['mix'].append(eg['mix'])
        chunk['s1'].append(eg['s1'])
        chunk['ref'].append(eg['ref'])
        chunk['s1_id'].append(eg['s1_id'])
    chunk['ref'] = _pad_aux(chunk['ref'])
    chunk['mix'] = _pad_(chunk['mix'],max_length)
    chunk['s1'] = _pad_(chunk['s1'],max_length)


    chunk['mix'] = torch.tensor(chunk['mix']) 
    chunk['s1'] = torch.tensor(chunk['s1'])
    chunk['ref'] = torch.tensor(chunk['ref'])
    chunk['s1_id'] = torch.tensor(chunk['s1_id'])

    return chunk

def _pad_aux(chunk_list,timeLen=3,sr = 16000):
    for idx,chunk_item in enumerate(chunk_list):
        if(chunk_item.shape[1]<timeLen*sr):
            P = timeLen*sr -chunk_item.shape[1]
            chunk_list[idx] = np.pad(chunk_item,((0,0),(0,P)),"constant")
        else:
            chunk_list[idx] = np.expand_dims(chunk_item[0,0:sr*timeLen],axis=0)
    return chunk_list

def _pad_(chunk_list,length):
    for idx,chunk_item in enumerate(chunk_list):
        chunk_list[idx] = np.pad(chunk_item,((0,0),(0,length-chunk_item.shape[1])),"constant")
    return chunk_list


class AudioDataLoader(DataLoader):
    def __init__(self,*args,**kwargs):
        super(AudioDataLoader,self).__init__(*args,**kwargs)




if __name__=='__main__':
    
    audio_data_trainval = AudioDataset(s1_scp='./data/M2Met/dev.scp',stage='train')
    
    dataLoader = AudioDataLoader(audio_data_trainval,batch_size=1,shuffle=True,collate_fn=collate)

    from AVModel import AVModel
    model = AVModel()
    model = nn.DataParallel(model)
    model_path = '/Work20/2020/lijunjie/Memet/log/M2Met/model/Checkpoint_0008.pt'
    checkpoint = torch.load(model_path)
    model_dict = checkpoint['model']

    model.load_state_dict(model_dict)
    model.cuda()
    model.eval()


    for idx,eg in enumerate(dataLoader):

        # print(eg['s1'].shape)
        # print(eg['mix'].shape)
        # print(eg['ref'].shape)
        # print(eg['s1_id'])
        mix = eg['mix'].float().cuda()
        ref = eg['ref'].float().cuda()
        s1 = eg['s1'].float().cuda()

        # loss = cal_si_snr(s1,mix)
        # print('%d:'%idx,loss)
        # est,s1_emb = model([mix,ref])
        # est = est[0].detach().cpu().permute(1, 0).numpy()
        # est/= np.max(np.abs(est))
        # import soundfile

        # soundfile.write('./audio/%d_est.wav'%idx, est, samplerate=16000)

        from scipy.io import wavfile
        s1_data = np.array(eg['s1']*32767)
        wavfile.write('./audio/%d_s1.wav'%idx,16000,s1_data.astype(np.int16))
        mix_data = np.array(eg['mix']*32767)
        wavfile.write('./audio/%d_mix.wav'%idx,16000,mix_data.astype(np.int16))
        ref = np.array(eg['ref']*32767)
        wavfile.write('./audio/%d_ref.wav'%idx,16000,ref.astype(np.int16))
        # if idx>=10:
        #     break
        break
        print('-----------')
        # print(eg['s1_ref'].size())
        # print(eg['s1_id'])

    
