def RN(objects):
    relations = []
    for i in range(len(objects)):
        for j in range(len(objects)):
            relations.append(g_theta(objects[i], objects[j]))
    summed = sum(relations)
    return f_phi(summed)


# 假设我们有 n 个对象，每个对象是一个 d 维向量
objects = [o1, o2, ..., on]  # o_i ∈ ℝ^d

# 1. 定义亲和力函数（可以是注意力机制，也可以是简单相似度）
def affinity(o_i, o_j):
    # 简单例子：点积 + 非线性
    return relu(dot(o_i, o_j))  # or use neural network: MLP(concat(o_i, o_j))

# 2. 计算所有对象对之间的亲和力分数
affinity_scores = []
pairs = []
for i in range(len(objects)):
    for j in range(len(objects)):
        if i == j:
            continue  # 可选：排除自身配对
        score = affinity(objects[i], objects[j])
        affinity_scores.append(score)
        pairs.append((objects[i], objects[j]))

# 3. 选择亲和力最高的 top_k 个对象对
top_k = 50  # 超参数
top_indices = argsort(affinity_scores)[-top_k:]  # 取分数最高的 top_k 对

selected_pairs = [pairs[i] for i in top_indices]

# 4. 将这些对象对送入关系网络 g_θ，进行关系推理
relations = [g_theta(o_i, o_j) for (o_i, o_j) in selected_pairs]

# 5. 聚合并输出（如求和后过 f_φ）
output = f_phi(sum(relations))