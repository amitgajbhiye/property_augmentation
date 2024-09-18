import logging

import torch
from torch import nn
from torch.nn.functional import normalize
from transformers import BertModel, RobertaModel, DebertaModel

log = logging.getLogger(__name__)

MODEL_CLASS = {
    "bert-base-uncased": (BertModel, 103),
    "bert-large-uncased": (BertModel, 103),
    "roberta-base": (RobertaModel, 50264),
    "roberta-large": (RobertaModel, 50264),
    "deberta-base": (DebertaModel, 50264),
    "deberta-large": (DebertaModel, 50264),
}


class ConceptPropertyModel(nn.Module):
    def __init__(self, model_params):
        super(ConceptPropertyModel, self).__init__()

        # self._concept_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self._property_encoder = BertModel.from_pretrained("bert-base-uncased")

        # self._concept_encoder = BertModel.from_pretrained(
        #     model_params.get("hf_model_path")
        # )
        # self._property_encoder = BertModel.from_pretrained(
        #     model_params.get("hf_model_path")
        # )

        self.hf_checkpoint_name = model_params.get("hf_checkpoint_name")
        self.hidden_dropout_prob = model_params.get("self.hidden_dropout_prob", 0.1)

        self.model_class, self.mask_token_id = MODEL_CLASS.get(self.hf_checkpoint_name)

        self._concept_encoder = self.model_class.from_pretrained(
            model_params.get("hf_model_path"),
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

        self._property_encoder = self.model_class.from_pretrained(
            model_params.get("hf_model_path"),
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

        self.dropout_prob = model_params.get("dropout_prob")
        self.strategy = model_params.get("vector_strategy")

        log.info(f"Model Class : {self.model_class}")
        log.info(f"Mask Token ID : {self.mask_token_id}")

    def forward(
        self,
        concept_input_id,
        concept_attention_mask,
        concept_token_type_id,
        property_input_id,
        property_attention_mask,
        property_token_type_id,
    ):
        concept_output = self._concept_encoder(
            input_ids=concept_input_id,
            attention_mask=concept_attention_mask,
            token_type_ids=concept_token_type_id,
        )

        property_output = self._property_encoder(
            input_ids=property_input_id,
            attention_mask=property_attention_mask,
            token_type_ids=property_token_type_id,
        )

        concept_last_hidden_states, concept_cls = (
            concept_output.get("last_hidden_state"),
            concept_output.get("pooler_output"),
        )

        property_last_hidden_states, property_cls = (
            property_output.get("last_hidden_state"),
            property_output.get("pooler_output"),
        )

        if self.strategy == "mean":
            # The dot product of the average of the last hidden states of the concept and property hidden states.

            v_concept_avg = torch.sum(
                concept_last_hidden_states
                * concept_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(concept_attention_mask, dim=1, keepdim=True)

            # Normalising concept vectors
            v_concept_avg = normalize(v_concept_avg, p=2, dim=1)

            v_property_avg = torch.sum(
                property_last_hidden_states
                * property_attention_mask.unsqueeze(1).transpose(2, 1),
                dim=1,
            ) / torch.sum(property_attention_mask, dim=1, keepdim=True)

            logits = (
                (v_concept_avg * v_property_avg)
                .sum(-1)
                .reshape(v_concept_avg.shape[0], 1)
            )

            return v_concept_avg, v_property_avg, logits

        elif self.strategy == "cls":
            # The dot product of concept property cls vectors.

            # Normalising concept vectors
            concept_cls = normalize(concept_cls, p=2, dim=1)

            logits = (
                (concept_cls * property_cls).sum(-1).reshape(concept_cls.shape[0], 1)
            )

            return concept_cls, property_cls, logits

        elif self.strategy == "mask_token":
            # The dot product of the mask tokens.

            # Index of mask token in concept input ids
            _, concept_mask_token_index = (
                concept_input_id == torch.tensor(self.mask_token_id)
            ).nonzero(as_tuple=True)

            concept_mask_vector = torch.vstack(
                [
                    torch.index_select(v, 0, torch.tensor(idx))
                    for v, idx in zip(
                        concept_last_hidden_states, concept_mask_token_index
                    )
                ]
            )
            # Normalising concept vectors
            concept_mask_vector = normalize(concept_mask_vector, p=2, dim=1)

            # Index of mask token in property input id
            _, property_mask_token_index = (
                property_input_id == torch.tensor(self.mask_token_id)
            ).nonzero(as_tuple=True)

            property_mask_vector = torch.vstack(
                [
                    torch.index_select(v, 0, torch.tensor(idx))
                    for v, idx in zip(
                        property_last_hidden_states, property_mask_token_index
                    )
                ]
            )

            logits = (
                (concept_mask_vector * property_mask_vector)
                .sum(-1)
                .reshape(concept_mask_vector.shape[0], 1)
            )

            # print("concept_mask_token_index")
            # print(concept_mask_token_index)

            # print("property_mask_token_index")
            # print(property_mask_token_index)

            return concept_mask_vector, property_mask_vector, logits
