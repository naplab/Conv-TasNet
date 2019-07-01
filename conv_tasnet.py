import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utility import models, sdr


# single-channel TasNet as DAE

class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3, 
                 kernel=3, num_spk=2, causal=False):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        self.BN = nn.Conv1d(self.enc_dim, self.feature_dim, 1, bias=False)
        
        # TCN encoder
        self.TCN_enc = models.TCN(self.feature_dim, self.enc_dim*self.num_spk, self.layer, 
                                  self.stack, self.feature_dim*4, self.kernel,
                                  causal=self.causal)

        self.receptive_field = self.TCN_enc.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def seg_separate(self, enc_output):
        # segment a segment of encoder output

        batch_size = enc_output.size(0)

        # TCN encoder
        mask = torch.sigmoid(self.TCN_enc(self.BN(enc_output))).view(batch_size, self.enc_dim, self.num_spk, -1)
        feature = mask * enc_output.unsqueeze(2)
        output = torch.cat([feature[:,:,i,:] for i in range(self.num_spk)], 0)  # B*C, N, L

        return output
        
    def forward(self, input):
        
        # padding
        output, rest = self.pad_signal(input)

        batch_size = output.size(0)
        
        # waveform encoder
        enc_output = self.encoder(output)  # B, N, T

        # split the encoder output into segments
        context = (self.receptive_field-1) // 2
        num_segments = enc_output.size(2) // self.receptive_field
        if enc_output.size(2) > num_segments * self.receptive_field:
            num_segments += 1

        all_separated = []
        for seg in range(num_segments):
            if seg == 0:
                # first segment
                this_seg = enc_output[:,:,:self.receptive_field+context]
                this_separate = self.seg_separate(this_seg)[:,:,:self.receptive_field]  # B*C, N, recep
            else:
                # other segments
                start_idx = seg*self.receptive_field-context
                end_idx = (seg+1)*self.receptive_field+context
                this_seg = enc_output[:,:,start_idx:end_idx]
                this_separate = self.seg_separate(this_seg)[:,:,context:context+self.receptive_field]  # B*C, N, recep
            all_separated.append(this_separate)
        output = torch.cat(all_separated, 2)
        
        # waveform decoder
        output = self.decoder(output)  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = [output[batch_size*i:batch_size*(i+1),0,:].contiguous() for i in range(self.num_spk)]  # C, B, L
        
        return output

def test_conv_tasnet():
    x = torch.rand(2, 32000)
    nnet = TasNet()
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()