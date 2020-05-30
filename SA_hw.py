import numpy as np
import matplotlib.pyplot as plt
import timeit  # 用來計時的package

nodes = 8
initial_t = 1000  # 初始溫度
final_t = 1  # 最後溫度
iteration = 300


def evaluate_fitness(current_solution):  # Evaluate the fitness
    total_dist = []
    for i in range(len(current_solution) - 1):  # 共8個距離，current_solution有9個點（起點與終點相同）
        print("city=", current_solution[i], "next city=", current_solution[i + 1])  # 幫助確認點跟點之間是否索引正確
        dist = distance_matrix[current_solution[i]][current_solution[i + 1]]  # 去distance matrix抓出兩城市間的距離
        total_dist.append(dist)
    print("total dist", sum(total_dist))  # 幫助檢查總距離
    return sum(total_dist)  # 回傳總距離


def cooling_schedule():  # Because I want to cool down the temperature in a smoother way,
    t = initial_t * np.power((final_t / initial_t), (i/iteration))  # so I choose the cooling function which will
    return t                                    # drop the temperature down slower than the geometrical one.


def select_new_solution(current_solution):  # find neighborhood
    loc_1 = np.random.randint(low=1, high=7)
    loc_2 = np.random.randint(low=1, high=7)    # 在位置1-7之間選擇2個位置交換(位置 0 & 8 是固定的node 1)
    if loc_1 == loc_2:  # avoid to choose the same location
        loc_1 = np.random.randint(low=1, high=7)
    current_solution[loc_1], current_solution[loc_2] = current_solution[loc_2], current_solution[loc_1]  # swap
    return current_solution


#  Build 8 nodes TSP Problem
#  generate 8 nodes symmetric distance matrix
distance_matrix = np.zeros((nodes, nodes), dtype=int)  # generate a 8x8 matrix for distance
for i in range(nodes):
    for j in range(i, nodes):
        distance_matrix[i][j] = distance_matrix[j][i] = np.random.randint(low=10, high=91)  # 對稱放入隨機距離
        if i == j:                          # city i to city j is not reasonable if i = j
            distance_matrix[i][j] = 999     # make the distance impossible
print(distance_matrix)

#   generate initial solution: [0,1,2,3,4,5,6,7,0]
initial_solution = []
for i in range(nodes):   # 產生[0,1,2,3,4,5,6,7]
    initial_solution.insert(i, i)  # 在第i個位置放入i
initial_solution.append(0)  # 放入0使頭尾都是 0，即回到城市1


#   SA Progress
best_solution = []  # for the best tour
convergence_history = []   # for the shortest tour distance


current_t = initial_t
current_solution = initial_solution.copy()  # 一開始的目前解就是最初解
current_fitness = evaluate_fitness(initial_solution)  # 計算最初解的fitness並令其為目前fitness
shortest_dist = current_fitness  # 一開始的最短距離就是initial fitness
convergence_history.append(current_fitness)  # 最初fitness插入convergence history


while current_t > final_t:
    i = 0
    while i < iteration:
        new_solution = select_new_solution(current_solution)  # 求出新的路徑解(tour in list)
        new_fitness = evaluate_fitness(new_solution)  # 求新路徑解之距離總和(value)
        delta = new_fitness - current_fitness   # 計算新解及目前解的差
        if delta < 0:   # For minimization, if delta < 0, new fitness replace the current one
            current_fitness = new_fitness
        else:  # but still allow some worse possible if the probability is bigger than random number
            if np.exp(- delta / current_t) > np.random.uniform():  # generate r~U(0,1)
                current_fitness = new_fitness
        if current_fitness < shortest_dist:  # collect the best information
            shortest_dist = current_fitness  # 記住最小距離(value)
            best_solution = new_solution.copy()  # 記住最小路徑解
        convergence_history.append(shortest_dist)   # 記錄收斂過程
        i = i + 1  # 下一代開始
        current_t = cooling_schedule()  # 降溫



print("best solution:", best_solution, "shortest distance:", shortest_dist)
print("shortest distance record:", convergence_history)  # convergence history
print("time:", timeit.timeit())  # 計算運算時間

# plot the convergence history
plt.plot(np.array(convergence_history))
plt.ylabel("shortest distance")
plt.xlabel("iteration")


print("##################")
#   Exhaustive solution 窮舉法
all_possible = []  # for all possible (5040個，8個點7種距離之排列組合)
while len(all_possible) < 5040:  # 除了node 1（位置0） 以外，7個node排列組合為7!=5040
    temp = np.random.permutation([1, 2, 3, 4, 5, 6, 7]).tolist()  # because the result in numpy is array,
    temp.insert(0, 0)                                               # change it to list
    temp.insert(8, 0)   # insert 0(which means node 1) at the front and back
    if temp not in all_possible:   # make sure the tour does not overlap
        all_possible.append(temp)


all_possible_dist = []  # for all possible distance
for i in range(len(all_possible)):  # 共5040個
    total_dist = []
    for j in range(nodes):
        dist = distance_matrix[all_possible[i][j]][all_possible[i][j+1]]  # 去distance matrix抓出兩城市間的距離
        total_dist.append(dist)
    all_possible_dist.append(sum(total_dist))  # 記住所有解的距離

shortest_dist = min(all_possible_dist)  # 找出最短解
shortest_index = all_possible_dist[shortest_dist]  # 找出最短解的位置
shortest_tour = all_possible[shortest_index]  # 根據相對位置索引出最短路徑之一（tour）
print("shortest tour:", shortest_tour, "shortest distance:", shortest_dist)
print("time:", timeit.timeit())  # 計時

plt.show()