from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# Створення графа
G = nx.Graph()

cities = [
    ("Київ"),
    ("Харків"),
    ("Львів"),
    ("Одеса"),
    ("Дніпро"),
    ("Херсон"),
    ("Запоріжжя"),
    ("Івано-Франківськ"),
    ("Чернівці"),
    ("Тернопіль"),
    ("Вінниця"),
    ("Хмельницький"),
    ("Рівне"),
    ("Житомир"),
    ("Черкаси"),
    ("Полтава"),
    ("Суми"),
    ("Чернігів"),
    ("Кропивницький"),
    ("Миколаїв"),
    ("Луцьк"),
    ("Донецьк"),
    ("Луганськ"),
]

G.add_nodes_from(cities)

edges = [
    ("Київ", "Харків", {"distance_km": 477}),
    ("Київ", "Львів", {"distance_km": 545}),
    ("Київ", "Одеса", {"distance_km": 472}),
    ("Київ", "Дніпро", {"distance_km": 478}),
    ("Харків", "Львів", {"distance_km": 888}),
    ("Харків", "Одеса", {"distance_km": 734}),
    ("Харків", "Дніпро", {"distance_km": 222}),
    ("Львів", "Одеса", {"distance_km": 762}),
    ("Львів", "Дніпро", {"distance_km": 878}),
    ("Одеса", "Дніпро", {"distance_km": 504}),
    ("Херсон", "Одеса", {"distance_km": 290}),
    ("Херсон", "Запоріжжя", {"distance_km": 270}),
    ("Запоріжжя", "Дніпро", {"distance_km": 89}),
    ("Івано-Франківськ", "Львів", {"distance_km": 138}),
    ("Івано-Франківськ", "Чернівці", {"distance_km": 155}),
    ("Чернівці", "Львів", {"distance_km": 209}),
    ("Чернівці", "Тернопіль", {"distance_km": 105}),
    ("Вінниця", "Хмельницький", {"distance_km": 102}),
    ("Вінниця", "Житомир", {"distance_km": 116}),
    ("Житомир", "Київ", {"distance_km": 130}),
    ("Черкаси", "Київ", {"distance_km": 189}),
    ("Черкаси", "Кропивницький", {"distance_km": 141}),
    ("Полтава", "Дніпро", {"distance_km": 147}),
    ("Полтава", "Харків", {"distance_km": 144}),
    ("Суми", "Харків", {"distance_km": 156}),
    ("Чернігів", "Київ", {"distance_km": 149}),
    ("Миколаїв", "Одеса", {"distance_km": 132}),
    ("Миколаїв", "Херсон", {"distance_km": 109}),
    ("Харків", "Донецьк", {"distance_km": 335}),
    ("Дніпро", "Донецьк", {"distance_km": 252}),
    ("Луцьк", "Рівне", {"distance_km": 97}),
    ("Донецьк", "Луганськ", {"distance_km": 148}),
    ("Луганськ", "Харків", {"distance_km": 333}),
    ("Вінниця", "Рівне", {"distance_km": 311}),
    ("Київ", "Рівне", {"distance_km": 318}),
]

G.add_edges_from(edges)

# Додамо ваги до ребер як відстань у кілометрах
for edge in G.edges():
    distance = G.edges[edge]["distance_km"]
    G.edges[edge]["weight"] = distance

# Аналіз основних характеристик
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degrees = dict(G.degree())

print(f"Кількість вершин: {num_nodes}")
print(f"Кількість ребер: {num_edges}")
print(f"Ступінь вершин: {degrees}")

# Функція для знаходження шляхів за допомогою DFS
def dfs_paths(graph, vertex, visited=None):
    if visited is None:
        visited = set()
    visited.add(vertex)
    print(vertex, end=' ')  # Відвідуємо вершину
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            dfs_paths(graph, neighbor, visited)


# Функція для знаходження шляхів за допомогою BFS
def bfs_paths(graph, start):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next_node in set(graph[vertex]) - set(path):
            queue.append((next_node, path + [next_node]))
    return path

# Виклик функцій для знаходження шляхів
start_city = "Хмельницький"

# Виведення результатів
print("\nDFS шлях:")
dfs_paths(G, start_city)
print("\nBFS шлях:")
bfs_path = bfs_paths(G, start_city)
print(bfs_path)

# Алгоритм Дейкстри
def dijkstra(graph, start, end):
     # Ініціалізація відстаней та множини невідвіданих вершин
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    unvisited = list(graph.nodes())

    while unvisited:
        # Знаходження вершини з найменшою відстанню серед невідвіданих
        current_vertex = min(unvisited, key=lambda vertex: distances[vertex])

        # Якщо поточна відстань є нескінченністю, то ми завершили роботу
        if distances[current_vertex] == float('infinity'):
            break

        for neighbor in graph.neighbors(current_vertex):
            distance = distances[current_vertex] + graph.edges[current_vertex, neighbor]["distance_km"]

            # Якщо нова відстань коротша, то оновлюємо найкоротший шлях
            if distance < distances[neighbor]:
                distances[neighbor] = distance

        # Видаляємо поточну вершину з множини невідвіданих
        unvisited.remove(current_vertex)

    return distances

start_vertex  = "Хмельницький"
end_vertex  = "Тернопіль"

shortest_distances = dijkstra(G, start_vertex, end_vertex)

# Виведення результату
print(f"Найкоротший маршрут з {start_vertex} до {end_vertex}:")
print(shortest_distances[end_vertex], "км")

# Візуалізація графа
plt.figure(figsize=(45, 25))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=7, edge_color="gray")
edge_labels = nx.get_edge_attributes(G, "distance_km")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Граф ключових автомобільних доріг України")
plt.show()

