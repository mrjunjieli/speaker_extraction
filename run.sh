#torch >=1.7！！！！！
#音频采样率 都是16000 ！！！！

wav_scp='/CDShare2/M2MeT_codes/espnet/egs2/AliMeeting/asr/dump2/raw/org/Train_Ali_near_nooverlap_onechannel/wav.scp'
noise_data='/CDShare2/OpenSLR/28/RIRS_NOISES/pointsource_noises'

#stage 1  数据准备
# python createDatapath.py --wav_scp_path "${wav_scp}" --noise_data "${noise_data}"


#stage 2 训练
batch_size=20
num_workers=8 #dataloader 线程数量 
num_epochs=500 #默认训练到num_epochs 轮次
lr=0.001 #初始学习率  当在dev上loss连续三次不下降则lr/=2  连续6次则停止训练
data_set_name='M2Met'
continue_from='None' #是否从断点开始训练 
#例如 continue_from='Checkpoint_0000.pt' 只需写文件名称 无需写路径

#支持多gpu训练
# 分离损失函数 SI-SNR   
CUDA_VISIBLE_DEVICES=0,3 python main.py --batch_size ${batch_size} --data_set_name ${data_set_name}\
             --continue_from ${continue_from} --num_workers ${num_workers}


#stage 3 推理
#----------
# model_path='/Work20/2020/lijunjie/AVModel/Conv_TasNet_speakemb/log/M2met/model/Checkpoint_0002.pt' #模型的文件路径
# data_scp_path='./data/M2Met/data_scp' #待测试的数据文件路径 包括所有的待测试文件 以’mix.wav' 'target.wav' 'ref.wav'的格式存储所有音频路径

# python separate.py --data_scp_path "${data_scp_path}" --model_path "${model_path}"
