#!/usr/bin/env python
import torch
import torchvision
import open_clip


class OpenCLIPNetwork:
    def __init__(self, device, canonical_corpus = ("object", "things", "stuff", "texture")):
        """Initialize the OpenCLIP model and cache default text embeddings.

        Parameters
        ----------
        device : Union[torch.device, str]
            Device on which to place the model and tensors (e.g., "cuda", "cpu").
        canonical_corpus : List[str]
            List of canonical corpus phrases. ("negatives")
        """
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = 'laion2b_s34b_b88k'
        self.clip_n_dims = 512
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)

        self.negatives = canonical_corpus
        self.positives = (" ",)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        """Compute positive-vs-hardest-negative probabilities for features.

        For each row in `embed`, computes a softmax over two values: similarity
        to the selected positive phrase and similarity to the hardest (highest
        similarity) negative phrase. A temperature of 10 is used before the
        softmax.

        Parameters
        ----------
        embed : torch.Tensor
            Feature matrix of shape [N, D] (e.g., [32768, 512]). D must match
            the model's embedding dimension. dtype should be floating (fp16/fp32).
        positive_id : int
            Index into `self.positives` selecting which positive phrase to score.

        Returns
        -------
        torch.Tensor
            Tensor of shape [N, 2] containing probabilities for
            [positive, hardest_negative] per row, same dtype as `embed`.
        """
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input, mask=None):
        """Encode a batch of images into OpenCLIP embeddings.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape [B, 3, H, W], float in [0, 1]. Images are resized
            to 224x224, normalized with CLIP mean/std, and cast to float16.
        mask : Optional[torch.Tensor]
            Optional mask forwarded to the underlying model. Shape semantics
            depend on the specific OpenCLIP variant (e.g., per-patch masks).

        Returns
        -------
        torch.Tensor
            Image embeddings of shape [B, D] (D = `self.clip_n_dims`), dtype
            float16, not L2-normalized.
        """
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input, mask=mask)

    def encode_text(self, text_list, device):
        """Encode a list of text prompts into OpenCLIP embeddings.

        Parameters
        ----------
        text_list : Sequence[str]
            Iterable of length T with text prompts.
        device : Union[torch.device, str]
            Device on which to place tokenized text and run encoding.

        Returns
        -------
        torch.Tensor
            Text embeddings of shape [T, D] (D = `self.clip_n_dims`), dtype
            matching the model precision (typically float16), not L2-normalized.
        """
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)
    
    def set_positives(self, text_list):
        """Set positive phrases and cache their L2-normalized embeddings.

        Parameters
        ----------
        text_list : Sequence[str]
            Iterable of length P containing positive phrases.

        Side Effects
        ------------
        Updates `self.positives` and `self.pos_embeds` (shape [P, D], L2-normalized).
        """
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    
    def set_semantics(self, text_list):
        """Set semantic labels and cache their L2-normalized embeddings.

        Parameters
        ----------
        text_list : Sequence[str]
            Iterable of length S containing semantic label phrases.

        Side Effects
        ------------
        Updates `self.semantic_labels` and `self.semantic_embeds`
        (shape [S, D], L2-normalized on the same device as the model).
        """
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.semantic_labels]).to("cuda")
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds /= self.semantic_embeds.norm(dim=-1, keepdim=True)
    
    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        """Assign a semantic label per spatial location from feature maps.

        Parameters
        ----------
        sem_map : torch.Tensor
            Feature tensor of shape [L, H, W, D] (e.g., L=3, D=512). Features
            are compared against `self.semantic_embeds` and `self.neg_embeds`.

        Returns
        -------
        torch.Tensor
            Long tensor of shape [L, H, W] containing predicted class indices
            in [0, S-1] for the S semantic labels. Locations assigned to any
            negative/background class are set to -1.
        """
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred.long()

    def get_max_across(self, sem_map):
        """Compute per-phrase positive relevance across pyramid levels.

        For each level and each phrase in `self.positives`, computes the
        positive probability against the hardest negative at every spatial
        location.

        Parameters
        ----------
        sem_map : torch.Tensor
            Feature tensor of shape [L, H, W, D]. D must match the model's
            embedding dimension.

        Returns
        -------
        torch.Tensor
            Tensor of shape [L, P, H, W] where P = len(self.positives),
            containing probabilities in [0, 1] for the positive class at each
            location and level.
        """
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1)

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j)
                pos_prob = probs[..., 0:1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)
        
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map