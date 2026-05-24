import numpy as np
import torch  # 导入torch用于处理tensor


# 数据掩码处理函数
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]  # 计算每个会话序列的长度
    len_max = max(us_lens)  # 找到最大序列长度
    # 填充序列至最大长度，使用 item_tail（通常为 0）填充
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # 创建掩码：有效位置为 1，填充位置为 0
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max  # 返回填充后的序列、掩码和最大长度


# 分割验证集的函数
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set  # 解包训练集的输入和目标
    n_samples = len(train_set_x)  # 获取训练样本数量
    sidx = np.arange(n_samples, dtype='int32')  # 创建样本索引数组
    np.random.shuffle(sidx)  # 随机打乱索引
    n_train = int(np.round(n_samples * (1. - valid_portion)))  # 计算训练集样本数
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]  # 提取验证集输入
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]  # 提取验证集目标
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]  # 提取训练集输入
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]  # 提取训练集目标
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)  # 返回训练集和验证集


# 定义数据处理类
class Data():
    # 初始化函数，处理输入数据
    def __init__(self, data, shuffle=False):
        inputs = data[0]  # 获取输入序列
        inputs, mask, len_max = data_masks(inputs, [0])  # 调用 data_masks 填充序列并生成掩码
        self.inputs = np.asarray(inputs)  # 将输入序列转换为 NumPy 数组
        self.mask = np.asarray(mask)  # 将掩码转换为 NumPy 数组
        self.len_max = len_max  # 保存最大序列长度
        self.targets = np.asarray(data[1])  # 将目标转换为 NumPy 数组
        self.length = len(inputs)  # 保存样本数量
        self.shuffle = shuffle  # 保存是否打乱数据的标志

    # 生成批次的函数
    def generate_batch(self, batch_size):
        if self.shuffle:  # 如果需要打乱数据
            shuffled_arg = np.arange(self.length)  # 创建索引数组
            np.random.shuffle(shuffled_arg)  # 随机打乱索引
            self.inputs = self.inputs[shuffled_arg]  # 打乱输入序列
            self.mask = self.mask[shuffled_arg]  # 打乱掩码
            self.targets = self.targets[shuffled_arg]  # 打乱目标
        n_batch = int(self.length / batch_size)  # 计算批次数量
        if self.length % batch_size != 0:  # 如果样本数不能整除批次大小
            n_batch += 1  # 增加一个批次
        slices = np.split(np.arange(n_batch * batch_size), n_batch)  # 按批次大小分割索引
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]  # 调整最后一个批次的大小
        return slices  # 返回批次索引列表

    # 获取单个批次数据的函数
    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]  # 获取当前批次的输入、掩码和目标

        # 针对每个会话构建 edge_index 和唯一的节点列表
        # unique_items_map 是一个批次中所有唯一且非零的物品ID列表，将作为 GNN 的节点。
        # new_alias_inputs 是原始序列中每个物品在这个 unique_items_map 中的索引。
        # final_batched_edge_index 是所有会话合并后的边列表，索引也对应 unique_items_map。

        all_unique_items_in_batch = []
        for seq in inputs:
            all_unique_items_in_batch.extend([item for item in seq if item != 0])

        # 获取批次中所有唯一的物品ID及其在排序后的唯一列表中的索引
        unique_items_map, _ = np.unique(all_unique_items_in_batch, return_inverse=True)
        unique_items_map = unique_items_map.tolist()  # 转换为列表

        # 构建新的 alias_inputs，使其指向 unique_items_map 中的索引
        new_alias_inputs = []
        for u_input in inputs:
            current_session_alias = []
            for item_id in u_input:
                if item_id == 0:
                    current_session_alias.append(0)  # 填充值保持为0
                else:
                    idx = unique_items_map.index(item_id)  # 使用 list.index 更直接
                    current_session_alias.append(idx)
            new_alias_inputs.append(current_session_alias)

        # 构建合并后的 edge_index
        batched_edge_index = []
        for u_input in inputs:
            # 构建当前会话内部的物品ID到其在 unique_items_map 中索引的映射
            session_id_to_global_idx = {item_id: unique_items_map.index(item_id)
                                        for item_id in np.unique(u_input) if item_id != 0}

            for k in range(len(u_input) - 1):
                if u_input[k + 1] == 0:
                    break
                src_item_id = u_input[k]
                dst_item_id = u_input[k + 1]

                # 确保 src_item_id 和 dst_item_id 都在当前会话的有效物品中
                if src_item_id in session_id_to_global_idx and dst_item_id in session_id_to_global_idx:
                    src_global_idx = session_id_to_global_idx[src_item_id]
                    dst_global_idx = session_id_to_global_idx[dst_item_id]
                    batched_edge_index.append([src_global_idx, dst_global_idx])

        if batched_edge_index:
            final_batched_edge_index = torch.tensor(batched_edge_index).t().contiguous()
        else:
            final_batched_edge_index = torch.empty((2, 0), dtype=torch.long)  # 空的边列表

        # 返回构建好的数据
        return new_alias_inputs, final_batched_edge_index, unique_items_map, mask, targets