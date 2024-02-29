import numpy as np
import os
import scipy
from tqdm import tqdm

def read_label(path, sub_id, ignore=30):
    label = []
    with open(os.path.join(path, '%d/%d/%d_1.txt' % (sub_id, sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
        return np.array(label[:-ignore])


def read_data(path, sub_id, channels, resample=3000):
    data = scipy.io.loadmat(os.path.join(path, '%d/%d/subject%d.mat' % (sub_id, sub_id, sub_id)))
    data_use = []
    for c in channels:
        data_use.append(
            np.expand_dims(scipy.signal.resample(data[c], resample, axis=-1), 1))
    data_use = np.concatenate(data_use, axis=1)
    return data_use


if __name__ == '__main__':
    channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
                'LOC_A2', 'ROC_A1', 'X1', 'X2']

    data_path = './dataset/ISRUC_S3/ExtractedChannels/'
    label_path = './dataset/ISRUC_S3/RawData/'
    output_path = './dataset/ISRUC_S3/SampleData'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for sub in tqdm(range(1, 11)):
        #print('Read subject', sub)
        label = read_label(label_path, sub)
        data = read_data(data_path, sub, channels)
        assert len(label) == len(data)
        label[label == 5] = 4
        for id in range(len(data)):
            s_data = data[id]
            s_label = label[id]
            saved_name = 'subject%d_%d_' % (sub, id) + str(s_label) + '.mat'
            scipy.io.savemat(
                os.path.join(output_path, saved_name),
                {
                    "data": s_data
                }
            )