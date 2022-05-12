import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]
    
def sample_negatives(targets, n_negatives = 10):
        
        y = targets.clone()
        
        if n_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        num = tsz
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
#         Большое значение - time в targets
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if n_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, n_negatives).flatten()
#                 тайм коды - от нуля до максимального = количеству временных срезов в таргете

                neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, n_negatives * num))
#                 Выбираем негативные индексы
#                 Получается 
    
                neg_idxs[neg_idxs == tszs] += 1

        if n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
#                 Прибавляем, чтобы было сэмплирование из того же аудио сэмпла, что и сам таргет кусочек
        else:
            neg_idxs = cross_neg_idxs

        negs = y[neg_idxs.view(-1)]
#         Сэмплируем сами негативные примеры
        
        negs = negs.view(bsz, num, n_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs
    
    
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

class ConstrativeLoss(nn.Module):

    def __init__(self, logit_temp: float = 1.0, 
                 cut = True, reduction = 'mean'):
        """
        Compute the contrastive loss with respect to the model outputs and sampled negatives from quantizer codebooks.
        Args:
            logit_temp: Temperature normalization applied in loss.
            reduce: Reduce loss via sum reduction (Default true)
        """
        super().__init__()
        self.logit_temp = logit_temp
        self.cut =  cut
        self.reduction = reduction

    def forward(
        self,
        logits: torch.tensor,
        targets: torch.tensor,
        negatives: torch.tensor,
        attention_mask = None
    ) -> [torch.tensor, torch.tensor, torch.tensor]:
        """
        Args:
            logits: Model activations
            targets: The true target quantized representations
            negatives: Sampled negatives from the quantizer codebooks. Sampled from all other timesteps.
            feature_loss: Feature penalty (L2 Norm)
        Returns:
            output loss values, acc_score
        """

        # Calculate similarity between logits and all targets, returning FxBxT
        similarity_scores = self._calculate_similarity(logits, negatives, targets)

        # Create targets of size B*T
        similarity_targets = logits.new_zeros(similarity_scores.size(1) * similarity_scores.size(2), dtype=torch.long)

        # Transpose similarity scores to (T*B)xF for loss
        similarity_scores = similarity_scores.transpose(0, 2)
        similarity_scores = similarity_scores.reshape(-1, similarity_scores.size(-1))

        attent = attention_mask[:, :-1].transpose(1, 0)
        attent = attent.reshape(-1)
        
        loss = torch.mean(F.cross_entropy(similarity_scores, similarity_targets, reduction='none')*attent)
#         loss = F.cross_entropy(similarity_scores, similarity_targets, reduction=self.reduction)

        acc_score = np.mean((torch.argmax(similarity_scores, dim = 1)*attent).cpu().numpy() == 0)
        return loss, acc_score
#         else:
#             return loss

    def _calculate_similarity(self, logits, negatives, targets):
#         neg_is_pos = (targets == negatives).all(-1)
#         print(neg_is_pos)
        targets = targets.unsqueeze(0)
        targets = torch.cat([targets, negatives], dim=0) 
        if self.cut:
            logits = logits[:, :-1, :]
            targets = targets[:, :, :-1, :]
        logits = torch.cosine_similarity(logits.float(), targets.float(), dim=-1).type_as(logits)
        logits /= self.logit_temp
#         if neg_is_pos.any():
#             logits[1:][neg_is_pos] = float("-inf")
        return logits