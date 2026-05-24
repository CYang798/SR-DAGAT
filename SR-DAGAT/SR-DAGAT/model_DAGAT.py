import datetime, math, numpy as np, torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor, mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn import JumpingKnowledge
from torch.nn import ModuleList, Linear


# --- directed_norm 和 get_norm_adj 函数保持不变 ---
def get_norm_adj(adj, norm):
    if norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


def directed_norm(adj):
    adj_coo = adj.coo()
    row, col, value = adj_coo[0], adj_coo[1], adj_coo[2]
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(adj.size(0), adj.size(1)))
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)
    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)
    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj


# --- 核心改动：修改 DAGCNConv，使用注意力融合 ---
class DAGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha=None):  # alpha 不再使用
        super(DAGCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)

        # --- 新增：用于计算注意力的MLP ---
        # 输入是拼接的前向和后向信息，输出是两种信息的权重
        self.attention_mlp = nn.Sequential(
            Linear(output_dim * 2, output_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(output_dim // 2, 2)
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        num_nodes = x.shape[0]

        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        adj_norm = get_norm_adj(adj, norm="dir")
        adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
        adj_t_norm = get_norm_adj(adj_t, norm="dir")

        # 计算前向和后向传播的节点表示
        x_forward = self.lin_src_to_dst(adj_norm @ x)
        x_backward = self.lin_dst_to_src(adj_t_norm @ x)

        # --- 核心改动：使用注意力机制融合 ---
        # 拼接两种表示
        combined_repr = torch.cat([x_forward, x_backward], dim=-1)
        # 计算注意力权重
        attention_weights = F.softmax(self.attention_mlp(combined_repr), dim=-1)

        # 加权求和
        output = attention_weights[:, 0].unsqueeze(1) * x_forward + attention_weights[:, 1].unsqueeze(1) * x_backward
        return output


# --- 修改 get_conv，不再传递 alpha ---
def get_conv(conv_type, input_dim, output_dim, alpha=None):
    if conv_type == "dir-gcn":
        return DAGCNConv(input_dim, output_dim)  # alpha不再需要
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")


# --- 修改 GNN，不再需要 alpha 相关参数 ---
class GNN(torch.nn.Module):
    def __init__(
            self, num_features, hidden_dim, num_layers=2, dropout=0,
            conv_type="dir-gcn", jumping_knowledge=None, normalize=False,
            alpha=None, learn_alpha=None  # alpha 参数已无用，但保留以兼容旧接口
    ):
        super(GNN, self).__init__()
        output_dim = hidden_dim

        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim))

        self.num_layers, self.dropout, self.normalize = num_layers, dropout, normalize
        self.jumping_knowledge = jumping_knowledge

        if self.jumping_knowledge and self.jumping_knowledge != '':
            input_dim = hidden_dim * num_layers if self.jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, hidden_dim)
            self.jump = JumpingKnowledge(mode=self.jumping_knowledge, channels=hidden_dim, num_layers=num_layers)
        else:
            self.lin, self.jump = None, None

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]
        if self.jumping_knowledge and self.jumping_knowledge != '':
            x = self.jump(xs)
            x = self.lin(x)
        return x


# 定义会话图模型类
class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        # --- 新增：位置嵌入相关 ---
        self.max_len = opt.max_len
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)

        # GNN 初始化不再需要 alpha
        self.gnn = GNN(
            num_features=self.hidden_size, hidden_dim=self.hidden_size,
            num_layers=opt.step, dropout=0.0, conv_type="dir-gcn",
            jumping_knowledge=opt.jk
        )

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    # 这里的 forward 保持原样，逻辑简单
    def forward(self, items, edge_index):
        x = self.embedding(items)
        hidden = self.gnn(x, edge_index)
        return hidden


# GPU/CPU 转换函数保持不变
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# --- 核心改动：在外部 forward 函数中加入位置信息 ---
def forward(model, i, data):
    alias_inputs, edge_index, items, mask, targets = data.get_slice(i)

    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    edge_index = trans_to_cuda(edge_index)
    mask = trans_to_cuda(torch.Tensor(mask).long())

    # 1. GNN 在唯一物品图上进行传播
    graph_node_hidden = model(items, edge_index)
    # 2. 将图节点表示映射回会话序列
    seq_hidden_no_pos = graph_node_hidden[alias_inputs]

    # 3. --- 新增：为序列表示加入位置嵌入 ---
    positions = torch.arange(alias_inputs.shape[1], device=alias_inputs.device).long()
    pos_embedding = model.position_embedding(positions).unsqueeze(0)
    # 通过 mask 确保不给 padding 加位置信息
    seq_hidden = seq_hidden_no_pos + pos_embedding * mask.unsqueeze(2).float()

    return targets, model.compute_scores(seq_hidden, mask)


# train_test 函数保持不变
def train_test(model, train_data, test_data):
    # ... (与方案一完全相同) ...
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr