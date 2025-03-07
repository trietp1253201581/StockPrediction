import torch
import torch.nn.functional as F
import math
import numpy as np

# Định nghĩa embedding cho các giá trị cổ phiếu với d_model = 6
# Mỗi dòng tương ứng với giá trị: [1.0, -0.5, 12.0, 3.2]
# Ta định nghĩa các embedding sao cho mỗi embedding có 6 chiều:
embeddings = torch.tensor([
    [1.0, 0.0, 0.5, 0.5, 1.0, 0.0],    # embed(1.0)
    [0.0, 1.0, 0.5, -0.5, 0.0, 1.0],     # embed(-0.5)
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],      # embed(12.0)
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]       # embed(3.2)
], dtype=torch.float)

# Thêm chiều batch: (seq_len, batch_size, d_model)
x = embeddings.unsqueeze(1)  # shape: (4, 1, 6)
print(x)

# Số head: 3, vì d_model = 6 và d_k = d_model / num_heads = 6/3 = 2.
num_heads = 3
d_model = 6
d_k = d_model // num_heads  # = 2
sqrt_dk = math.sqrt(d_k)    # sqrt(2) ~ 1.414

# Tách embeddings thành 3 head:
head1 = x[:, :, 0:2]  # (4, 1, 2)
head2 = x[:, :, 2:4]  # (4, 1, 2)
head3 = x[:, :, 4:6]  # (4, 1, 2)

print("Head1:\n", head1.squeeze(1))
print("Head2:\n", head2.squeeze(1))
print("Head3:\n", head3.squeeze(1))

# --- Tính toán cho Head 1 ---
# Query tại vị trí 1 (index 0) của Head1: Q1 = [1.0, 0.0]
Q1_head1 = head1[0, 0, :]  # tensor([1.0, 0.0])
scores_head1 = []
for j in range(head1.size(0)):
    Kj = head1[j, 0, :]
    score = torch.dot(Q1_head1, Kj) / sqrt_dk
    scores_head1.append(score.item())
scores_head1 = np.array(scores_head1)
print("Head 1 Scores:", scores_head1)

exp_scores_head1 = np.exp(scores_head1)
attn_weights_head1 = exp_scores_head1 / exp_scores_head1.sum()
print("Head 1 Attention Weights:", attn_weights_head1)

output_head1 = sum(attn_weights_head1[j] * head1[j, 0, :].numpy() for j in range(head1.size(0)))
print("Head 1 Output:", output_head1)

# --- Tính toán cho Head 2 ---
# Query tại vị trí 1 của Head2: Q1 = head2[0,0,:] = [0.5, 0.5]
Q1_head2 = head2[0, 0, :]
scores_head2 = []
for j in range(head2.size(0)):
    Kj = head2[j, 0, :]
    score = torch.dot(Q1_head2, Kj) / sqrt_dk
    scores_head2.append(score.item())
scores_head2 = np.array(scores_head2)
print("Head 2 Scores:", scores_head2)

exp_scores_head2 = np.exp(scores_head2)
attn_weights_head2 = exp_scores_head2 / exp_scores_head2.sum()
print("Head 2 Attention Weights:", attn_weights_head2)

output_head2 = sum(attn_weights_head2[j] * head2[j, 0, :].numpy() for j in range(head2.size(0)))
print("Head 2 Output:", output_head2)

# --- Tính toán cho Head 3 ---
# Query tại vị trí 1 của Head3: Q1 = head3[0,0,:] = [1.0, 0.0]
Q1_head3 = head3[0, 0, :]
scores_head3 = []
for j in range(head3.size(0)):
    Kj = head3[j, 0, :]
    score = torch.dot(Q1_head3, Kj) / sqrt_dk
    scores_head3.append(score.item())
scores_head3 = np.array(scores_head3)
print("Head 3 Scores:", scores_head3)

exp_scores_head3 = np.exp(scores_head3)
attn_weights_head3 = exp_scores_head3 / exp_scores_head3.sum()
print("Head 3 Attention Weights:", attn_weights_head3)

output_head3 = sum(attn_weights_head3[j] * head3[j, 0, :].numpy() for j in range(head3.size(0)))
print("Head 3 Output:", output_head3)

# --- Kết hợp các head bằng cách nối lại ---
multihead_output = np.concatenate([output_head1, output_head2, output_head3])
print("Multi-Head Attention Output:", multihead_output)
