from torch.utils.data import Dataset, DataLoader, Sampler
import json
import torch

class Dataset:
    
    def __init__(self, path, manifest_path):
        with open(manifest_path, 'r') as json_file:
            manifest = json.load(json_file)
        self.manifest = manifest
        self.path = path
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, ind):
        audio_filepath = self.manifest[ind]['audio_filepath']
        audio_file = os.path.join(self.path, audio_filepath)
        sampling_rate, signal = wav.read(audio_file)
        
        return {'sample': signal, 'length': len(signal)}
        
def collate_fn(samples):
    
    max_length = max([sample['length'] for sample in samples])
    samples1 = []
    lengths = []
    samplings = []
    for sample in samples:
        to_add_l = max_length-sample['length']
        sample1 = list(sample['sample'])+[0]*to_add_l
        samples1.append(torch.Tensor(sample1).unsqueeze(0))
        lengths.append(sample['length'])
        
    batch = torch.cat(samples1)
    lengths = torch.Tensor(lengths)
    return dict(batch=batch, lengths=lengths)