import torch
import torch.nn as nn
from transformers import LEDModel

class HierarchicalEncoder(nn.Module):
    def __init__(self, model_name="allenai/led-base-16384"):
        super().__init__()
        self.led = LEDModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.led(input_ids=input_ids,
                           attention_mask=attention_mask)
        return outputs.last_hidden_state
