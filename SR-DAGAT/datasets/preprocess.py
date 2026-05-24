# 导入必要的Python模块
import argparse  # 用于解析命令行参数
import time  # 用于处理时间相关操作
import csv  # 用于读取和处理CSV文件
import pickle  # 用于序列化和保存数据到文件
import operator  # 提供操作符函数，如排序时的键提取
import datetime  # 用于处理日期和时间
import os  # 用于操作系统相关功能，如创建目录

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加一个参数 '--dataset'，默认值为 'sample'，帮助信息说明支持的数据集名称
parser.add_argument('--dataset', default='yoochoose', help='dataset name: diginetica/yoochoose/sample')
# 解析命令行参数并存储在 opt 中
opt = parser.parse_args()
# 打印解析后的参数对象，方便调试
print(opt)

# 设置默认数据集文件名为 'sample_train-item-views.csv'
dataset = 'sample_train-item-views.csv'
# 如果命令行参数指定数据集为 'diginetica'，则使用对应的文件
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
# 如果命令行参数指定数据集为 'yoochoose'，则使用对应的文件
elif opt.dataset == 'yoochoose':
    dataset = 'yoochoose-clicks.dat'

# 打印程序开始时间，格式为当前时间的字符串表示
print("-- Starting @ %ss" % datetime.datetime.now())
# 以只读模式打开数据集文件
with open(dataset, "r") as f:
    # 根据数据集类型选择 CSV 文件的分隔符并创建 DictReader 对象
    if opt.dataset == 'yoochoose':
    # yoochoose数据集文件没有列名（header），需要手动指定
    # 字段名分别为: session_id, timestamp, item_id, category
        fieldnames = ['session_id', 'timestamp', 'item_id', 'category']
        reader = csv.DictReader(f, delimiter=',', fieldnames=fieldnames)
    else:
        reader = csv.DictReader(f, delimiter=';')  # 其他数据集使用分号分隔
    # if opt.dataset == 'yoochoose':
    #     reader = csv.DictReader(f, delimiter=',')  # yoochoose 使用逗号分隔
    # else:
    #     reader = csv.DictReader(f, delimiter=';')  # 其他数据集使用分号分隔
    sess_clicks = {}  # 创建字典，用于存储每个会话的点击序列
    sess_date = {}  # 创建字典，用于存储每个会话的日期
    ctr = 0  # 初始化计数器，用于统计处理的行数
    curid = -1  # 初始化当前会话ID，设为-1表示尚未开始处理
    curdate = None  # 初始化当前日期，设为 None 表示尚未赋值
    # 逐行读取 CSV 文件中的数据
    for data in reader:
        sessid = data['session_id']  # 获取当前行的会话ID
        # 如果已有日期且当前会话ID与之前不同，说明切换到了新会话
        if curdate and not curid == sessid:
            date = ''  # 初始化日期变量
            # 根据数据集类型将日期字符串转换为时间戳
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))  # yoochoose 的时间格式
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))  # 其他数据集的时间格式
            sess_date[curid] = date  # 将上一个会话的日期存储到字典中
        curid = sessid  # 更新当前会话ID为当前行的会话ID
        # 根据数据集类型获取物品ID
        if opt.dataset == 'yoochoose':
            item = data['item_id']  # yoochoose 数据集直接使用 item_id
        else:
            item = data['itemId'], int(data['timeframe'])  # 其他数据集将 item_id 和 timeframe 组成元组
        curdate = ''  # 重置当前日期变量
        # 根据数据集类型获取日期
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']  # yoochoose 使用 timestamp 字段
        else:
            curdate = data['eventdate']  # 其他数据集使用 eventdate 字段

        # 将物品添加到会话的点击序列中
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]  # 如果会话已存在，追加物品
        else:
            sess_clicks[sessid] = [item]  # 如果会话不存在，创建新列表并添加物品
        ctr += 1  # 计数器加一，表示处理了一行数据
    date = ''  # 初始化日期变量，用于处理最后一个会话
    # 将最后一个会话的日期转换为时间戳
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))  # yoochoose 的时间格式
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))  # 其他数据集的时间格式
        # 对于非 yoochoose 数据集（例如 diginetica），按 timeframe 对点击序列排序
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))  # 按元组的第二个元素排序
            sess_clicks[i] = [c[0] for c in sorted_clicks]  # 只保留物品ID，丢弃 timeframe
    sess_date[curid] = date  # 存储最后一个会话的日期
# 打印数据读取完成的时间
print("-- Reading data @ %ss" % datetime.datetime.now())

# 过滤掉长度为1的会话
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:  # 如果会话的点击序列长度为1
        del sess_clicks[s]  # 删除该会话的点击序列
        del sess_date[s]  # 删除该会话的日期

# 统计每个物品出现的次数
iid_counts = {}  # 创建字典，用于存储物品ID及其出现次数
for s in sess_clicks:
    seq = sess_clicks[s]  # 获取会话的点击序列
    for iid in seq:  # 遍历序列中的每个物品
        if iid in iid_counts:
            iid_counts[iid] += 1  # 如果物品已存在，计数加一
        else:
            iid_counts[iid] = 1  # 如果物品不存在，初始化计数为1

# 对物品出现次数进行排序
sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))  # 按出现次数排序

length = len(sess_clicks)  # 记录当前会话总数
# 过滤掉物品出现次数少于5次的会话
for s in list(sess_clicks):
    curseq = sess_clicks[s]  # 获取当前会话的点击序列
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))  # 保留出现次数>=5的物品
    if len(filseq) < 2:  # 如果过滤后序列长度小于2
        del sess_clicks[s]  # 删除该会话的点击序列
        del sess_date[s]  # 删除该会话的日期
    else:
        sess_clicks[s] = filseq  # 更新会话的点击序列为过滤后的序列

# 根据日期分割测试集
dates = list(sess_date.items())  # 将会话日期字典转换为列表，格式为 [(session_id, date), ...]
maxdate = dates[0][1]  # 初始化最大日期为第一个会话的日期

# 找到所有会话中的最大日期
for _, date in dates:
    if maxdate < date:
        maxdate = date  # 更新最大日期

# 设置分割日期，yoochoose 为1天前，diginetica 为7天前
splitdate = 0  # 初始化分割日期
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # 减去1天的秒数 (86400秒)
else:
    splitdate = maxdate - 86400 * 7  # 减去7天的秒数

# 打印分割日期
print('Splitting date', splitdate)  # 输出分割日期，便于调试
# 分割训练集和测试集
tra_sess = filter(lambda x: x[1] < splitdate, dates)  # 训练集：日期早于分割日期的会话
tes_sess = filter(lambda x: x[1] > splitdate, dates)  # 测试集：日期晚于分割日期的会话

# 对会话按日期排序
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))  # 训练集按日期升序排序
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))  # 测试集按日期升序排序
print(len(tra_sess))  # 打印训练集会话数量
print(len(tes_sess))  # 打印测试集会话数量
print(tra_sess[:3])  # 打印训练集前3个会话，便于检查
print(tes_sess[:3])  # 打印测试集前3个会话，便于检查
# 打印分割完成的时间
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# 创建物品ID映射字典，用于将原始物品ID重新编号
item_dict = {}
# 定义函数，将训练集会话转换为序列并重新编号物品ID
def obtian_tra():
    train_ids = []  # 存储训练集会话ID
    train_seqs = []  # 存储训练集点击序列
    train_dates = []  # 存储训练集日期
    item_ctr = 1  # 物品ID计数器，从1开始
    for s, date in tra_sess:  # 遍历训练集会话
        seq = sess_clicks[s]  # 获取会话的点击序列
        outseq = []  # 初始化输出序列
        for i in seq:  # 遍历序列中的每个物品
            if i in item_dict:
                outseq += [item_dict[i]]  # 如果物品已有编号，使用已有编号
            else:
                outseq += [item_ctr]  # 否则分配新编号
                item_dict[i] = item_ctr  # 将新编号存入字典
                item_ctr += 1  # 计数器加一
        if len(outseq) < 2:  # 如果序列长度小于2，跳过（实际上不会发生）
            continue
        train_ids += [s]  # 添加会话ID
        train_dates += [date]  # 添加日期
        train_seqs += [outseq]  # 添加重新编号后的序列
    print(item_ctr)  # 打印物品总数（编号的最大值加1）
    return train_ids, train_dates, train_seqs  # 返回训练集数据

# 定义函数，将测试集会话转换为序列，仅保留训练集中出现的物品
def obtian_tes():
    test_ids = []  # 存储测试集会话ID
    test_seqs = []  # 存储测试集点击序列
    test_dates = []  # 存储测试集日期
    for s, date in tes_sess:  # 遍历测试集会话
        seq = sess_clicks[s]  # 获取会话的点击序列
        outseq = []  # 初始化输出序列
        for i in seq:  # 遍历序列中的每个物品
            if i in item_dict:  # 如果物品在训练集中出现
                outseq += [item_dict[i]]  # 添加对应的编号
        if len(outseq) < 2:  # 如果序列长度小于2，跳过
            continue
        test_ids += [s]  # 添加会话ID
        test_dates += [date]  # 添加日期
        test_seqs += [outseq]  # 添加重新编号后的序列
    return test_ids, test_dates, test_seqs  # 返回测试集数据

# 获取训练集和测试集数据
tra_ids, tra_dates, tra_seqs = obtian_tra()  # 获取训练集
tes_ids, tes_dates, tes_seqs = obtian_tes()  # 获取测试集

# 定义函数，处理序列数据，生成输入序列和目标
def process_seqs(iseqs, idates):
    out_seqs = []  # 存储输入序列
    out_dates = []  # 存储日期
    labs = []  # 存储目标标签
    ids = []  # 存储会话索引
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):  # 遍历序列和日期
        for i in range(1, len(seq)):  # 从1开始，生成所有可能的子序列
            tar = seq[-i]  # 目标物品为序列倒数第i个
            labs += [tar]  # 添加目标
            out_seqs += [seq[:-i]]  # 添加除目标外的子序列作为输入
            out_dates += [date]  # 添加对应的日期
            ids += [id]  # 添加会话索引
    return out_seqs, out_dates, labs, ids  # 返回处理后的数据

# 处理训练集和测试集序列
tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)  # 处理训练集
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)  # 处理测试集
tra = (tr_seqs, tr_labs)  # 训练集数据，包含输入序列和目标
tes = (te_seqs, te_labs)  # 测试集数据，包含输入序列和目标
print(len(tr_seqs))  # 打印训练集序列数量
print(len(te_seqs))  # 打印测试集序列数量
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])  # 打印训练集前3个序列、日期和目标
Massage = print(te_seqs[:3], te_dates[:3], te_labs[:3])  # 打印测试集前3个序列、日期和目标
all = 0  # 初始化总序列长度

# 计算所有序列的总长度
for seq in tra_seqs:
    all += len(seq)  # 累加训练集序列长度
for seq in tes_seqs:
    all += len(seq)  # 累加测试测试集序列长度
# 计算并打印平均序列长度
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
# 根据数据集类型保存数据
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):  # 如果目录不存在
        os.makedirs('diginetica')  # 创建目录
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))  # 保存训练集
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))  # 保存测试集
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))  # 保存所有训练序列
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):  # 如果目录不存在
        os.makedirs('yoochoose1_4')  # 创建目录
    if not os.path.exists('yoochoose1_64'):  # 如果目录不存在
        os.makedirs('yoochoose1_64')  # 创建目录
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))  # 保存测试集到 1/4 目录
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))  # 保存测试集到 1/64 目录

    # 计算训练集的 1/4 和 1/64 分割点
    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))  # 打印 1/4 训练集大小
    print(len(tr_seqs[-split64:]))  # 打印 1/64 训练集大小

    # 分割训练集为 1/4 和 1/64 两部分
    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))  # 保存 1/4 训练集
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))  # 保存 1/4 训练序列

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))  # 保存 1/64 训练集
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))  # 保存 1/64 训练序列

else:
    if not os.path.exists('sample'):  # 如果目录不存在
        os.makedirs('sample')  # 创建目录
    pickle.dump(tra, open('sample/train.txt', 'wb'))  # 保存训练集
    pickle.dump(tes, open('sample/test.txt', 'wb'))  # 保存测试集
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))  # 保存所有训练序列

# 打印完成信息
print('Done.')