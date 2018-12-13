import numpy as np
import matplotlib.pyplot as plt
import math

number_of_bandits = 10  # 老虎机个数
number_of_arms=10  # 老虎机臂数
number_of_pulls=10000  # 老虎机被选中探索次数
epsilon = 0.3  # exploration比例
min_temp = 0.1 # 衰减epsilon的最小阈值
decay_rate = 0.999  # exploration程度衰减系数


def pick_arm(q_values, counts, strategy, success, failure):
    """exploitation & exploration"""
    global epsilon

    if strategy == 'random':
        # 随机选择
        return np.random.randint(0, len(q_values))

    if strategy == 'greedy':
        # 贪心，不做探索，只选当前收益最大的老虎机
        best_arms_value = np.max(q_values)
        best_arms = np.argwhere(q_values==best_arms_value).flatten()
        return best_arms[np.random.randint(0,len(best_arms))]

    if strategy == 'egreedy' or strategy == 'egreedy_decay':
        # 当对用户比较了解时，可能不再需要太高比例的探索，因此用decay_rate来减弱探索程度
        if  strategy == "egreedy_decay":
            epsilon = max(epsilon * decay_rate, min_temp)

        # 以1-epsilon概率选取当前收益最大的臂，以epsilon的概率随机选取一个臂
        if np.random.random() > epsilon:
            best_arms_value = np.max(q_values)  # 得到收益最大的臂
            best_arms = np.argwhere(q_values == best_arms_value).flatten()
            return best_arms[np.random.randint(0,len(best_arms))]
        else:
            return np.random.randint(0,len(q_values))

    if strategy == 'ucb':
        # Upper Confidence Bound
        # 均值越大，标准差越小，被选中的概率会越来越大 $\bar{x_j}(t)+\sqrt{\frac{2\ln{t}}{T_{j,t}}}$
        total_counts = np.sum(counts)  # 每个臂的使用次数之和

        # 第一项：当前得到的多臂老虎机收益率 第二项：避免部分老虎机过多探索，0.001平滑项
        q_values_ucb = q_values + np.sqrt(np.reciprocal(counts + 0.001) * 2 * math.log(total_counts + 1.0))
        best_arms_value = np.max(q_values_ucb)
        best_arms = np.argwhere(q_values_ucb == best_arms_value).flatten()
        return best_arms[np.random.randint(0,len(best_arms))]

    if strategy == 'thompson':
        # 汤普森采样，每个臂维护一个beta分布，每次用现有的beta分布产生一个随机数，选择随机数最大的臂
        sample_means = np.zeros(len(counts))
        for i in range(len(counts)):
            sample_means[i]=np.random.beta(success[i]+1,failure[i]+1)
        return np.argmax(sample_means)


fig = plt.figure()
ax = fig.add_subplot(111)

for st in ['greedy', 'random', 'egreedy', 'egreedy_decay', 'ucb', 'thompson']:
    best_arm_counts = np.zeros((number_of_bandits,number_of_pulls))

    for i in range(number_of_bandits):
        arm_means = np.random.rand(number_of_arms)
        best_arm = np.argmax(arm_means)

        q_values = np.zeros(number_of_arms)
        counts = np.zeros(number_of_arms)
        success = np.zeros(number_of_arms)
        failure = np.zeros(number_of_arms)

        for j in range(number_of_pulls):
            a = pick_arm(q_values,counts,st,success,failure)

            reward = np.random.binomial(1,arm_means[a])
            counts[a] += 1.0
            q_values[a] += (reward-q_values[a])/counts[a]

            success[a] += reward
            failure[a] += (1-reward)
            best_arm_counts[i][j] = counts[best_arm]*100.0/(j+1)
        epsilon = 0.3

    ys = np.mean(best_arm_counts, axis=0)
    xs = range(len(ys))
    ax.plot(xs, ys, label=st)

plt.xlabel('Steps')
plt.ylabel('Optimal pulls')

plt.tight_layout()
plt.legend()
plt.ylim((0, 110))
plt.show()
