from transformers.models.electra import ElectraConfig

class Config(ElectraConfig):
    model_type = "electra"

    def __init__(
            self,
            *args,
            mask_prob: float = 0.25,
            mask_token_id: int = 103,
            masking_strategy: str = 'dynamic_masking',
            temperature: int = 1,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.masking_strategy = masking_strategy
        self.temperature = temperature
