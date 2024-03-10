import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, out_channels, hidden_size, kernel_size, num_layers, global_cmvn):
        super(ConvLSTM, self).__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for fluent/dysfluent classification
        self.fc_fluent = nn.Linear(hidden_size, 2)  # Assuming binary classification for fluent/dysfluent
        
        # Fully connected layer for soft prediction of five event types
        self.fc_soft = nn.Linear(hidden_size, vocab_size)  # Assuming 5 event types
        self.global_cmvn = global_cmvn
        self.loss_soft = torch.nn.MultiLabelSoftMarginLoss()
        


    def forward(
        self,
        speech_tensor: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        hidden_state=None
    ):
        x = self.global_cmvn(speech_tensor)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        fluent_output = self.fc_fluent(x.mean(dim=1))
        soft_output = self.fc_soft(x.mean(dim=1))

        soft_labels = text
        fluent_labels = soft_labels.any(dim=1).long()
        fluent_labels = F.one_hot(fluent_labels, num_classes=2)

        # Loss computation
        fluent_loss = weighted_cross_entropy_with_logits(fluent_output, fluent_labels)
        soft_loss = self.loss_soft(soft_output, soft_labels)
        
        # Total loss is a sum of fluent_loss and soft_loss
        # You may want to balance these losses or apply different weights
        total_loss = 0.1 * fluent_loss + 0.9 * soft_loss
        # total_loss = soft_loss
        return {"loss" : total_loss}
    
    def decode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
        """ Decode input speech

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )

        Returns: 
        """
        assert speech.shape[0] == speech_lengths.shape[0]

        x = self.global_cmvn(speech)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        soft_output = self.fc_soft(x.mean(dim=1))
        results = torch.sigmoid(soft_output)
        return results


# Loss functions
def weighted_cross_entropy_with_logits(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels.float())

