# goit-algo-hw-06
В якості графа вибрано прототип головних автомобільних шляхів між обласними центрами України.
![image](https://github.com/Serhii-Khodyka/goit-algo-hw-06/assets/155672103/cde1afd4-2586-4090-9b79-1ed1e6ab2fac)

Характеристики графа:
Кількість вершин: 23
Кількість ребер: 35
Ступінь вершин: {'Київ': 8, 'Харків': 8, 'Львів': 6, 'Одеса': 6, 'Дніпро': 7, 'Херсон': 3, 'Запоріжжя': 2, 'Івано-Франківськ': 2, 'Чернівці': 3, 'Тернопіль': 1, 'Вінниця': 3, 'Хмельницький': 1, 'Рівне': 3, 'Житомир': 2, 'Черкаси': 2, 'Полтава': 2, 'Суми': 1, 'Чернігів': 1, 'Кропивницький': 1, 'Миколаїв': 2, 'Луцьк': 1, 'Донецьк': 3, 'Луганськ': 2}

Різниця в побудові шляху з Хмельницького двома алгоритмами DFS та BFS, що DFS відпрацював глибше у графі і досліджує всі можливі гілки, а BFS горизонтально і побудував коротший шлях 
DFS шлях:
Хмельницький Вінниця Житомир Київ Харків Львів Одеса Дніпро Запоріжжя Херсон Миколаїв Полтава Донецьк Луганськ Івано-Франківськ Чернівці Тернопіль Суми Черкаси Кропивницький Чернігів Рівне Луцьк
BFS шлях:
['Хмельницький', 'Вінниця', 'Рівне', 'Київ', 'Харків', 'Луганськ', 'Донецьк', 'Дніпро', 'Запоріжжя', 'Херсон', 'Миколаїв', 'Одеса', 'Львів', 'Івано-Франківськ', 'Чернівці', 'Тернопіль']

Реалізація алгоритма Дейкстри
Найкоротший шлях з Хмельницький до Тернопіль:
['Хмельницький', 'Вінниця', 'Житомир', 'Київ', 'Львів', 'Чернівці', 'Тернопіль']
Найкоротша відстань: 1207 км
