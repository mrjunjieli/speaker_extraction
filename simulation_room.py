import os
import random
import re

import pyroomacoustics as pra
import numpy as np
import soundfile as sf





def simulation(audio_data,num):
    audio_data = audio_data.squeeze()
    room_length = [3,10.21]
    room_height = [2.3,3.2]
    room_width = [2.6,5.4]

    length = np.round(random.uniform(room_length[0], room_length[1]))
    width = np.round(random.uniform(room_width[0], room_width[1]))
    height = np.round(random.uniform(room_height[0], room_height[1]))


    # 根据房间大小随机生成rt60
    if length >= 3 and length <= 6:
        rt60_tgt = np.round(random.uniform(0.2, 0.5),2)
    elif length > 6 and length <= 10:
        rt60_tgt = np.round(random.uniform(0.3, 0.6),2)
    elif length > 10:
        rt60_tgt = np.round(random.uniform(0.4, 0.7),2)
    
    mic_locations = np.c_[
        [0.5, width / 2 - 0.2, 1.5],
        [0.5, width / 2 - 0.14, 1.5],
        [0.5, width / 2 - 0.1, 1.5],
        [0.5, width / 2 - 0.06, 1.5],
        [0.5, width / 2 + 0.06, 1.5],
        [0.5, width / 2 + 0.1, 1.5],
        [0.5, width / 2 + 0.14, 1.5],
        [0.5, width / 2 + 0.2, 1.5],
    ]
    room_dim = [length, width, height]
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)
    

    source_location = np.array([random.uniform(0.5, length-0.5),
                                    random.uniform(0.5, width - 0.5),
                                    random.uniform(1.5, 2.0)])
    
    c = 345
    dist = np.linalg.norm(source_location - mic_locations[:,0])
    delay = dist / c


    room.add_source(source_location, signal=audio_data, delay=delay)


    room.add_microphone_array(mic_locations)

    # Run the simulation
    room.simulate()

    # orig_max_value = np.max(np.abs(audio_data))
    multichannel_audio_data = room.mic_array.signals[:, 0:len(audio_data)]

    simulated_data = multichannel_audio_data[num]

    # simulated_data = simulated_data / np.max(np.abs(simulated_data)) * orig_max_value
    # simulated_data = simulated_data.astype(np.int16)

    return np.expand_dims(simulated_data,0)


if __name__ == '__main__':
    path = '/CDShare2/M2MeT_codes/espnet/egs2/AliMeeting/asr/dump2/raw/org/Train_Ali_near_nooverlap_onechannel/data/format.32/Train-near-R2108_M3244_M_SPK3383-c1-0038949-0039388.wav'
    data,_ = sf.read(path)
    audio = simulation(data,0)
    print(audio.shape)
