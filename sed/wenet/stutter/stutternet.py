import torch
from torch import nn
import torchaudio as audio
from torch import Tensor
import torch.nn.functional as F

class StutterNet(nn.Module):
  def __init__(self, vocab_size, n_mels=40, 
               dropout=0.0, use_batchnorm=False, scale=1):
    '''Implementation of StutterNet
    from Sheikh et al. StutterNet: 
    "Stuttering Detection Using 
    Time Delay Neural Network" 2021

    Args:
      n_mels (int, optional): number of mel filter banks
      n_classes (int, optional): number of classes in output layer
      use_dropout (bool, optional): whether or not to use dropout in the
        last two linear layers
      use_batchnorm (bool, optional): whether ot not to batchnorm in the
        TDNN layers
      scale (float ,optional): width scale factor
    '''
    super(StutterNet, self).__init__()

    self.n_mels = n_mels
    
    # self.spec = audio.transforms.MelSpectrogram(n_mels=n_mels, sample_rate=16000,
    #                                           n_fft=512, pad=1, f_max=8000, win_length=400,
    #                                           f_min=0, power=2.0, hop_length=160, norm='slaney')
    # self.db = audio.transforms.AmplitudeToDB()
    # self.mfcc = audio.transforms.MFCC(16000, 40)
    self.tdnn_1 = nn.Conv1d(n_mels, int(512*scale), 5, dilation=1)
    self.tdnn_2 = nn.Conv1d(int(512*scale), int(1536*scale), 5, dilation=2)
    self.tdnn_3 = nn.Conv1d(int(1536*scale), int(512*scale), 7, dilation=3)
    self.tdnn_4 = nn.Conv1d(int(512*scale), int(512*scale), 1)
    self.tdnn_5 = nn.Conv1d(int(512*scale), int(1500*scale), 1)
    self.fc_1 = nn.Linear(int(3000*scale), 512)
    self.relu = nn.ReLU()
    self.bn_1 = nn.BatchNorm1d(int(512*scale))
    self.bn_2 = nn.BatchNorm1d(int(1536*scale))
    self.bn_3 = nn.BatchNorm1d(int(512*scale))
    self.bn_4 = nn.BatchNorm1d(int(512*scale))
    self.bn_5 = nn.BatchNorm1d(int(1500*scale))
    
    nn.init.xavier_uniform_(self.fc_1.weight)
    self.dropout_1 = nn.Dropout(dropout)
    self.fc_2 = nn.Linear(512, 256)
    nn.init.xavier_uniform_(self.fc_1.weight)
    self.dropout_2 = nn.Dropout(dropout)

    self.binary_head = nn.Linear(256, 2)
    self.class_head = nn.Linear(256, vocab_size)

    self.sig = nn.Sigmoid()

    self.loss_fluent = torch.nn.BCELoss()
    self.loss_soft = torch.nn.MultiLabelSoftMarginLoss()

  def forward(
      self,
      speech: torch.Tensor,
      speech_lengths: torch.Tensor,
      text: torch.Tensor,
      text_lengths: torch.Tensor,
    ):
    '''forward method'''
    # import pdb; pdb.set_trace()
    x = speech
    batch_size = x.shape[0]

    x = x.permute(0, 2, 1)

    x = self.tdnn_1(x)
    x = self.relu(x)
    x = self.bn_1(x)
    x = self.tdnn_2(x)
    x = self.relu(x)
    x = self.bn_2(x)
    x = self.tdnn_3(x)
    x = self.relu(x)
    x = self.bn_3(x)
    x = self.tdnn_4(x)
    x = self.relu(x)
    x = self.bn_4(x)
    x = self.tdnn_5(x)
    x = self.relu(x)
    x = self.bn_5(x)
        
    mean = torch.mean(x,-1)
    std = torch.std(x,-1)
    x = torch.cat((mean,std),1)
    x = self.fc_1(x)
    x = self.dropout_1(x)
    x = self.fc_2(x)
    x = self.dropout_2(x)

    soft_labels = text
    fluent_labels = soft_labels.any(dim=1).long()
    fluent_labels = F.one_hot(fluent_labels, num_classes=2)

    binary = self.binary_head(x)
    binary = self.sig(binary)

    classes = self.class_head(x)
    # classes = self.sig(classes)

    fluent_loss = self.loss_fluent(binary, fluent_labels.float())
    soft_loss = self.loss_soft(classes, soft_labels)
    total_loss = 0.1 * fluent_loss + 0.9 * soft_loss

    return {"loss" : total_loss}
  def decode(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor
  ):
    x = speech
    batch_size = x.shape[0]

    x = x.permute(0, 2, 1)

    x = self.tdnn_1(x)
    x = self.relu(x)
    x = self.bn_1(x)
    x = self.tdnn_2(x)
    x = self.relu(x)
    x = self.bn_2(x)
    x = self.tdnn_3(x)
    x = self.relu(x)
    x = self.bn_3(x)
    x = self.tdnn_4(x)
    x = self.relu(x)
    x = self.bn_4(x)
    x = self.tdnn_5(x)
    x = self.relu(x)
    x = self.bn_5(x)

    mean = torch.mean(x,-1)
    std = torch.std(x,-1)
    x = torch.cat((mean,std),1)
    x = self.fc_1(x)
    x = self.dropout_1(x)
    x = self.fc_2(x)
    x = self.dropout_2(x)

    classes = self.class_head(x)
    results = torch.sigmoid(classes)
    return results

class ResBlock1d(nn.Module):
  def __init__(self, input_dims, output_dims, depth=2, kernel_size=3,
               use_batchnorm=False, downsample=False, dropout=0.0):
    super(ResBlock1d, self).__init__()

    self.depth = depth
    self.use_batchnorm = use_batchnorm

    scale = 1
    self.up = None
    if (downsample):
      self.down = nn.Conv1d(int(input_dims), int(output_dims), 3, 2, padding=1)
      # self.down = nn.MaxPool1d(1, stride=2)
      scale=2

    self.downsample = downsample

    self.conv_1 = nn.Conv1d(int(input_dims), 
      output_dims, 3, stride=scale, padding=1)
    
    self.convs = nn.ModuleList([nn.Conv1d(output_dims, 
      output_dims, kernel_size, padding='same') for _ in range(depth-1)])
    
    self.bn_1 = nn.BatchNorm1d(output_dims)
    self.bn = None

    if (use_batchnorm):
      self.bn = nn.ModuleList([nn.BatchNorm1d(
          output_dims) for _ in range(depth-1)])
      
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):

    temp = x
    if (self.downsample):
      temp = self.down(x)

    x = self.conv_1(x)
    x = self.bn_1(x)

    if (not self.use_batchnorm):
      for i in range(self.depth-1):
        x = self.convs[i](x)
        x = self.dropout(x)
        if (i != self.depth-2):
          x = self.relu(x)
    else:
      for i in range(self.depth-1):
        x = self.convs[i](x)
        x = self.dropout(x)
        x = self.bn[i](x)
        if (i != self.depth-2):
          x = self.relu(x)
    x = temp + x

    return x

class ResNet1D(nn.Module):
  def __init__(self, n_mels=100,n_classes=12, kernel_size=3,
               dropout=0.0, use_batchnorm=False, scale=1):
    '''Implementation of StutterNet
    from Sheikh et al. StutterNet: 
    "Stuttering Detection Using 
    Time Delay Neural Network" 2021

    Args:
      n_mels (int, optional): number of mel filter banks
      n_classes (int, optional): number of classes in output layer
      use_dropout (bool, optional): whether or not to use dropout in the
        last two linear layers
      use_batchnorm (bool, optional): whether ot not to batchnorm in the
        TDNN layers
      scale (float ,optional): width scale factor
    '''
    super(ResNet1D, self).__init__()

    self.n_mels = n_mels
    
    # self.spec = audio.transforms.MelSpectrogram(n_mels=n_mels, sample_rate=16000,
    #                                            n_fft=512, pad=1, f_max=8000, f_min=0,
    #                                            power=2.0, hop_length=160)
    # self.mfcc = audio.transforms.MFCC(16000, 40)
    # self.db = audio.transforms.AmplitudeToDB()
    self.tdnn_1 = nn.Conv1d(n_mels, int(64*scale), 3, padding=1, bias=False)

    self.res_1_1 = ResBlock1d(int(64*scale), int(64*scale), kernel_size=kernel_size, downsample=True, use_batchnorm=use_batchnorm, dropout=dropout)
    self.res_1_2 = ResBlock1d(int(64*scale), int(64*scale), kernel_size=kernel_size, downsample=False, use_batchnorm=use_batchnorm, dropout=dropout)
    self.res_1_3 = ResBlock1d(int(64*scale), int(64*scale), kernel_size=kernel_size, downsample=False, use_batchnorm=use_batchnorm, dropout=dropout)

    self.res_2_1 = ResBlock1d(int(64*scale), int(128*scale), kernel_size=kernel_size, downsample=True, use_batchnorm=use_batchnorm, dropout=dropout)
    self.res_2_2 = ResBlock1d(int(128*scale), int(128*scale), kernel_size=kernel_size, downsample=False, use_batchnorm=use_batchnorm, dropout=dropout)
    self.res_2_3 = ResBlock1d(int(128*scale), int(128*scale), kernel_size=kernel_size, downsample=False, use_batchnorm=use_batchnorm, dropout=dropout)

    self.res_3_1 = ResBlock1d(int(128*scale), int(256*scale), kernel_size=kernel_size, downsample=True, use_batchnorm=use_batchnorm, dropout=dropout)
    self.res_3_2 = ResBlock1d(int(256*scale), int(256*scale), kernel_size=kernel_size, downsample=False, use_batchnorm=use_batchnorm, dropout=dropout)
    self.res_3_3 = ResBlock1d(int(256*scale), int(256*scale), kernel_size=kernel_size, downsample=False, use_batchnorm=use_batchnorm, dropout=dropout)

    self.res_4_1 = ResBlock1d(int(256*scale), int(512*scale), kernel_size=kernel_size, downsample=True, use_batchnorm=use_batchnorm, dropout=dropout)
    self.res_4_2 = ResBlock1d(int(512*scale), int(512*scale), kernel_size=kernel_size, downsample=False, use_batchnorm=use_batchnorm, dropout=dropout)
    self.res_4_3 = ResBlock1d(int(512*scale), int(512*scale), kernel_size=kernel_size, downsample=False, use_batchnorm=use_batchnorm, dropout=dropout)

    # self.bn = nn.BatchNorm1d(int(512*scale))

    self.relu = nn.ReLU()
    self.fc = nn.Linear(int(1024*scale), n_classes)

  def forward(self, x):
    '''forward method'''
    batch_size = x.shape[0]

    # x = self.spec(x)
    # x = self.mfcc(x)
    # x = self.db(x)
    x = self.tdnn_1(x)

    x = self.res_1_1(x)
    x = self.relu(x)
    x = self.res_1_2(x)
    x = self.relu(x)
    x = self.res_1_3(x)
    x = self.relu(x)

    x = self.res_2_1(x)
    x = self.relu(x)
    x = self.res_2_2(x)
    x = self.relu(x)
    x = self.res_2_3(x)
    x = self.relu(x)

    x = self.res_3_1(x)
    x = self.relu(x)
    x = self.res_3_2(x)
    x = self.relu(x)
    x = self.res_3_3(x)
    x = self.relu(x)

    x = self.res_4_1(x)
    x = self.relu(x)
    x = self.res_4_2(x)
    x = self.relu(x)
    x = self.res_4_3(x)
    x = self.relu(x)

    # x = self.bn(x)
    mean = torch.mean(x,-1)
    std = torch.std(x,-1)
    x = torch.cat((mean,std),1)
    x = self.fc(x)

    return x
  
  from torch import Tensor

'''credit: https://github.com/roman-vygon/BCResNet'''

class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)


class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out

class BCResNet(torch.nn.Module):
    def __init__(self):
        super(BCResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16, 8)
        self.block1_2 = BroadcastedBlock(8)

        self.block2_1 = TransitionBlock(8, 12, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12, dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(12, 16, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(16, 20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(20, 20, 5, groups=20, padding=(0, 2))
        self.conv3 = nn.Conv2d(20, 32, 1, bias=False)
        self.conv4 = nn.Conv2d(32, 12, 1, bias=False)

    def forward(self, x):

        out = self.conv1(x)

        out = self.block1_1(out)
        out = self.block1_2(out)

        out = self.block2_1(out)
        out = self.block2_2(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        out = self.conv2(out)

        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

        out = self.conv4(out)

        return out.reshape((-1, 12))
