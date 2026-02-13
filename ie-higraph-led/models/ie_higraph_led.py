import torch
import torch.nn as nn
from transformers import LEDForConditionalGeneration
from models.entity_graph import EntityGraphLayer

class IEHiGraphLED(nn.Module):
    def __init__(self, model_name="allenai/led-base-16384"):
        super().__init__()
        self.led = LEDForConditionalGeneration.from_pretrained(model_name)
        self.entity_graph = EntityGraphLayer(self.led.config.d_model)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.led(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
