import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio


class WaveToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, maxMelLen, sampling_rate, device):
        # sr = 44100 etc
        self.maxMelLen = maxMelLen
        self.sampling_rate = sampling_rate
        torchaudio.set_audio_backend("sox_io")

        self.device = device
        #https://github.com/PCerles/audio/blob/3803d0b27a4e13efa760227ef6c71d0f3753aa98/test/test_transforms.py#L262
        #librosa defaults
        n_fft = 2048
        hop_length = 512
        power = 2.0
        n_mels = 128
        n_mfcc = 40
        # htk is false in librosa, no setting in torchaudio -?
        # norm is 1 in librosa, no setting in torchaudio -?
        self.melspect_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            window_fn=torch.hann_window,
            hop_length=hop_length,
            n_mels=n_mels,
            n_fft=n_fft).to(device)

        self.db_transform = torchaudio.transforms.AmplitudeToDB("power",
                                                                80.).to(device)

    def __call__(self, sample):
        x, y = sample
        x = x.to(self.device)
        mel = self.melspect_transform(x.reshape(-1))
        an_x = self.db_transform(mel)
        #librosa version
        #         mel = librosa.feature.melspectrogram(y=x.reshape(-1),
        #                                              sr=self.sampling_rate)
        #         an_x = librosa.power_to_db(mel, ref=np.max)
        #         an_x = an_x.astype("float32")
        #         y = y.astype('float32')
        #         print(an_x.shape)
        # print(an_x.shape)
        an_x = an_x[:, :self.maxMelLen]
        if an_x.shape[1]>self.maxMelLen:
            print('WARNING ignoring data on the spectogram')        
        # 2-d conv
        #         x = an_x.reshape(1, *an_x.shape[:])
        # 1-d conv
        x = an_x.reshape(1, an_x.shape[0], an_x.shape[1])

        return x, y

class DropStripes(object):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes. 
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [1, 2]    # dim 1: freq_bins; dim 2: time_steps

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def __call__(self, input):
        """input: x -> ( channels, time_steps, freq_bins)"""

        x,y= input if isinstance(input, tuple) else (input,None)

        assert x.ndimension() == 3

        # batch_size = input.shape[0]
        total_width = x.shape[self.dim]

        # for n in range(batch_size):
        self.transform_slice(x, total_width)

        out = x,y if isinstance(input, tuple) else x
        return out


    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0


class SpecAugmentation(object):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, 
        freq_stripes_num):
        """Spec augmetation. 
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.freq_dropper = DropStripes(dim=2, drop_width=time_drop_width, 
            stripes_num=time_stripes_num)

        self.time_dropper = DropStripes(dim=1, drop_width=freq_drop_width, 
            stripes_num=freq_stripes_num)

    def __call__(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x

    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device='cpu'):
        # sr = 44100 etc
        # self.maxMelLen = maxMelLen
        # self.sampling_rate = sampling_rate

        self.device = device

    def __call__(self, sample):
        x, y = sample
        x = x.to(self.device)

        # an_x = an_x[:, :self.maxMelLen]
        # 2-d conv
        #         x = an_x.reshape(1, *an_x.shape[:])
        # 1-d conv
        x = x.reshape(1, x.shape[0], x.shape[1])

        return x, y


# #test
# maxMelLen_test = 850
# SAMPLING_RATE_test = 48000
# sample_len_seconds = 10
# # to_tensor works on single sample
# sample_count = 1
# xx_test = torch.ones((sample_count,SAMPLING_RATE_test*sample_len_seconds))
# y_values = torch.ones(sample_count)
#
# toTensor = ToTensor(maxMelLen_test,SAMPLING_RATE_test)
# x_out,y_out=toTensor((xx_test,y_values))
# x_out.shape,y_out.shape


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
  Utility function for computing output of convolutions
  takes a tuple of (h,w) and returns a tuple of (h,w)
  """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation *
                                      (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation *
                                      (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


# mel.shape,an_x.shape,X_train.shape
class singleconv1dModel(nn.Module):
    '''A simple model for testing by overfitting.
    '''

    def __init__(self,
                 out_channels,
                 h_w,
                 kernel_size,
                 fc_1_size,
                 FLAT=False,
                 output_shape=(10,)):
        # h_w: height will be always one since we use 1d convolution
        super(singleconv1dModel, self).__init__()
        self.out_channels = out_channels
        #### CONV
        self.conv1 = nn.Conv1d(
            in_channels=1,  # depth of image == depth of filters
            out_channels=self.out_channels,  # number of filters 
            kernel_size=kernel_size,  # size of the filters/kernels
            padding=1)

        self.conv1_shape = conv_output_shape(h_w,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             pad=1,
                                             dilation=1)
        # conv is 1d
        self.conv1_shape = (1, self.conv1_shape[1])

        self.fc1 = nn.Linear(self.out_channels * self.conv1_shape[0] *
                             self.conv1_shape[1], fc_1_size)  # 100

        self.fc2 = nn.Linear(fc_1_size, output_shape[0])

    def forward(self, x):
        #         x = x.reshape(1,)
        #         print(x.shape) #  50,1,108800 (850*128)
        x = F.relu(self.conv1(x))
        #         x = self.pool(x)
        # x = self.drop(x)
        #         print(x.shape)# 58, 2, 108801
        #         print(self.conv1_shape)
        #         print(x.shape)
        x = x.view(
            -1, self.out_channels * self.conv1_shape[0] * self.conv1_shape[1])
        # batch_norm is missing
        x = F.relu((self.fc1(x)))
        x = (self.fc2(x))

        #         x = self.drop(x)

        #         x = self.fc4(x)
        #         x = torch.sigmoid(x)
        #                 x = F.log_softmax(x,dim=1)
        return x


# test
# input_shape=(1,(938*128))
# output_shape=(10,)
# testModel_ins=adam(out_channels=2,h_w=input_shape,kernel_size=2,output_shape=output_shape)
# # a.conv1.weight
# a_out=testModel_ins(torch.ones((3,1,input_shape[1])))

# a_out_correct=torch.zeros(a_out.shape)
# a_out_correct[0][:]=1
# a_out_correct
# a_out.detach().numpy()

# torch.exp(a_out),a_out


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock5x5(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2),
                               bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn6(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                 classes_num,intermediate_pool_type='max',global_pool_type='max+avg'):
        '''
        Args:
            intermediate_pool_type: can be avg,max,avg+max,
            global_pool_type: can be max,mean,max+mean,None
        '''
        super(Cnn6, self).__init__()
        del sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None
        # # Spectrogram extractor
        # self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
        #     win_length=window_size, window=window, center=center, pad_mode=pad_mode,
        #     freeze_parameters=True)

        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
        #     n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
        #     freeze_parameters=True)

        # # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        #     freq_drop_width=8, freq_stripes_num=2)

        self.intermediate_pool_type=intermediate_pool_type
        self.global_pool_type = global_pool_type
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block0 = ConvBlock5x5(in_channels=1, out_channels=32)
        self.conv_block1 = ConvBlock5x5(in_channels=32, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4_out_c=512
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=self.conv_block4_out_c)
        self.fc1_output_size = 128
        if self.global_pool_type=='':
            self.fc1 = nn.Linear(512*58, self.conv_block4_out_c, bias=True)
        else:
            self.fc1 = nn.Linear(self.conv_block4_out_c, self.fc1_output_size, bias=True)
        self.fc_audioset = nn.Linear(self.fc1_output_size, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size,1, mel_bins,time_steps)
        """

        # x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        # x = x.transpose(1, 3)
        # x = self.bn0(x)
        # x = x.transpose(1, 3)

        # if self.training:
        # x = self.spec_augmenter(x)

        # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)



        x = input
        # swap to (batch_size,1,time_steps,mel_bins)
        x = x.transpose(2, 3)
        # print('input shape',x.shape)
        dropout_prob=0.2
        x = self.conv_block0(x, pool_size=(2, 2), pool_type=self.intermediate_pool_type)
        x = F.dropout(x, p=dropout_prob, training=self.training)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type=self.intermediate_pool_type)
        x = F.dropout(x, p=dropout_prob, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type=self.intermediate_pool_type)
        x = F.dropout(x, p=dropout_prob, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type=self.intermediate_pool_type)
        x = F.dropout(x, p=dropout_prob, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type=self.intermediate_pool_type)
        x = F.dropout(x, p=dropout_prob, training=self.training)

# https://arxiv.org/pdf/1904.03476.pdf
# To help the systems robust to frequencyshift of sound events,
#  we average out the frequency information inthe feature maps of the
#  last convolutional layer. For audio taggingtasks with weakly 
# labelled data, the information over time frames aremaxed out, 
# which is designed to select the predominant informationover 
# time steps for clip-level classification. Finally, a fully 
# connectedlayer is applied to predict the presence of sound 
# events either at theclip-level or frame-level.

# if there is not swap like that x = x.transpose(2, 3)
# input shape torch.Size([32, 1, 128, 938])
# input shape before mean dim=3 torch.Size([32, 512, 8, 58])
# input shape after mean dim=3 torch.Size([32, 512, 8])
# x1 shape after max dim=2 torch.Size([32, 512])
# x2 shape after max dim=2 torch.Size([32, 512])
        # print('input shape before mean dim=3',x.shape)
        x = torch.mean(x, dim=3)
        if self.global_pool_type=='avg+max':
            (x1, _) = torch.max(x, dim=2)
            # print('x1 shape after max dim=2',x1.shape)
            x2 = torch.mean(x, dim=2)
            # print('x2 shape after max dim=2',x2.shape)
            x = x1 + x2
        elif self.global_pool_type=='max':
            (x1, _) = torch.max(x, dim=2)
            x = x1
        elif self.global_pool_type=='avg':
            x2 = torch.mean(x, dim=2)
            x = x2
        elif self.global_pool_type=='':
            x= x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        else:
            raise Exception(f'this global pool type not implemented{self.global_pool_type}')
    
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        # embedding = F.dropout(x, p=0.5, training=self.training)
        # clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output = (self.fc_audioset(x))

        # output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output
