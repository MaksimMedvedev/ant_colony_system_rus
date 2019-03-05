import numpy as np
import random


class Colony:
    def __init__(self, distances, alpha = 0.6, beta = 2, rho = 0.3, q0 = 0.6):
        """

        Здесь, помимо указанных аргументов, также определяются:
        tour_length: общая длина маршрута для одного муравья
        min_nn_heuristic: длина маршрута, определенного по принципу выбора соседнего города с наименьшей дистанцией до
        tau0: начальное количество феромона на каждом пути
        sigma: параметр, определяющий изменение феромона при глобальном обновлении феромонов (по смыслу равен rho)
        pheromones_on_arcs: распределение феромонов по дорогам

        :param distances: матрица расстояний между пунктами
        :param ants_count: количество муравьев в колонии
        :param alpha: коэффициент влияния феромонов на выбор пути
        :param beta: коэффициент влияния расстояния между городами на выбор пути
        :param rho: скорость испарения феромона
        :param q0: параметр, определяющий влияние случайности при выборе следующего города муравьем (чем q0 больше, тем
        менее случаен выбор)
        :param iters_count: количество запусков поиска колонией оптимального пути
        """

        np.fill_diagonal(distances, 0)
        self.distances = np.array(distances)
        self._cities = list(range(len(distances)))
        self.ants_count = 2 * len(self._cities)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.iters_count = 2 * len(self._cities)
        
        self.tour_length = 0
        self.min_nn_heuristic = self.__nearest_neighbor_heuristic()
        self.tau0 = 1 / (len(distances)*self.min_nn_heuristic) if len(distances)*self.min_nn_heuristic != 0 else 0
        self.sigma = self.rho
        self.pheromones_on_arcs = np.full_like(self.distances, self.tau0, dtype = np.double)
        np.fill_diagonal(self.pheromones_on_arcs, 0)

    @property
    def cities(self):
        return self._cities.copy()

    def ants_full_search(self):
        self.global_min_tour = []
        self.global_min_tour_length = 0
        self.first_time = True
        # поиск маршрута муравьями в количестве ants_count, локальное обновление феромонов каждым муравьем после поиска
        for iteration in range(self.iters_count):
            self.__searching_iteration()
            self.first_time = False
            # глобальное обновление феромонов "глобально лучшим" из муравьев
            self.__global_pheromones_update()
        return self.global_min_tour_length 

    def __searching_iteration(self):
        """

        Одна из iters_count итераций поиска маршрута муравьями в количестве ants_count.

        Для каждого муравья сначала случайным образом выбирается начальный город, после этого муравей в __find_tour()
        ищет какой-то путь на основе правил алгоритма ACS.

        После нахождения маршрута при необходимости в __update_optimum() обновляется глобально минимальный маршрут,
        затем в __local_pheromones_update() происходит обновление феромонов каждым муравьем относительно того пути,
        который он прошел.

        Поскольку граф полносвязный, считается, что в список непосещенных городов входят все города, кроме
        первого, из которого муравей начинает движение

        :return: None
        """

        for ant in range(self.ants_count):
            self.tour_current = []
            self.tour_length = 0
            self.non_visited_cities = self.cities
            # случайным образом выбираем стартовую точку муравья и добавляем ее в текущий путь, считая ее уже посещенной
            # (вносим в табу)
            self.current_city = random.choice(self.cities)
            self.non_visited_cities.remove(self.current_city)
            self.tour_current.append(self.current_city)

            # муравей ищет какой-то путь
            self.__find_tour()
            # обновляем глобальный оптимум
            self.global_min_tour_length, self.global_min_tour = self.__update_optimum(self.global_min_tour_length,
                                                                                      self.global_min_tour)
            # локальное обновление феромонов
            if not self.first_time:
                self.__local_pheromones_update()
            
    def __find_tour(self):
        """

        Пока муравью еще есть куда идти (список непосещенных городов), в __choose_next_city() выбирается следующий город,
        после чего в __update_path_and_tabu() обновляется список непосещенных, а также добавляется новая точка маршрута
        в общий маршрут

        :return: None
        """
        while len(self.non_visited_cities) != 0:
            next_city = self.__choose_next_city()
            self.__update_path_and_tabu(self.current_city, next_city)
        self.__update_path_and_tabu(self.current_city, self.tour_current[0])

    def __choose_next_city(self):
        """
        Берется значение феромона на дороге между текущим городом current_city и потенциальным кандидатом candidate,
        затем рассчитываются параметры для вычисления вероятности: n^alpha, tau^beta.
        Вероятность - вероятность того, что муравей посетит определенный город. Чем выше, тем больше шанс у муравья
        пойти в определенную точку.
        Вероятности для всех потенциальных кандидатов хранятся в probabilities, значение их рассчитывается как
        (pheromones_on_arc ^ alpha) * visibility / denumerator, где:
            denumerator сумма всех числителей дробей;
            visibility = (1 / distance) ^ beta, distance - расстояние между текущим городом и кандидатом на посещение.

        Одновременно с этим рассчитывается параметр tau * (n^beta).

        По алгоритму ACS, генерируется случайное q и если оно меньше, чем заданное q0, то мы выбираем город
        с наибольшим значением выражения tau * (n^beta), n = visibility, иначе - случайным образом с учетом
        вероятностей, подсчитанных ранее

        :return: next_city: город, который муравей посетит следующим (индекс города в матрице смежности)
        """
        probabilities = np.zeros(len(self.non_visited_cities))
        attract = np.full(len(self.non_visited_cities), np.nan)
        denumerator = 0
        # если посетить можно единственный город, то выбирать ничего не нужно, возвращается он
        if len(self.non_visited_cities) == 1:
            return self.non_visited_cities[0]
        for candidate in range(len(self.non_visited_cities)):
            pheromones_on_arc = self.pheromones_on_arcs[self.current_city][self.non_visited_cities[candidate]]
            visibility = (1 / self.distances[self.current_city][self.non_visited_cities[candidate]]) ** self.beta
            # attract = tau * (n^beta)
            attract[candidate] = pheromones_on_arc * visibility
            probabilities[candidate] = (pheromones_on_arc ** self.alpha) * visibility
            denumerator += probabilities[candidate]
        # получаем итоговые вероятности для каждого из потенциальных городов для посещения
        probabilities /= denumerator
        # критерий выбора города - случайное равномерно распределенное число
        q = random.uniform(0, 1)
        if q < self.q0:
            next_city = self.non_visited_cities[int(np.argwhere(attract == np.nanmax(attract))[0][0])]
        else:
            next_city = random.choices(self.non_visited_cities, probabilities)[0]
        return next_city

    def __update_path_and_tabu(self, current_city, next_city):
        """

        К маршруту добавляется следующий посещаемый город, затем обновляется общая длина маршрута, из списка непосещенных
        удаляется новый город. Муравей считается перешедшим в новый город из старого

        :param current_city: город, в котором сейчас находится муравей
        :param next_city: город, в который муравей перейдет, который был выбран в __choose_next_city()
        :return: None
        """
        # добавляем к маршруту найденный следующий город
        self.tour_current.append(next_city)
        # обновляем общую длину пути
        self.tour_length += self.distances[current_city][next_city]
        # удаление посещенного города из списка непосещенных (реализация табу), если есть еще откуда удалять
        if len(self.non_visited_cities) != 0:
            self.non_visited_cities.remove(next_city)
            # город, в который перешли, становится текущим
            self.current_city = next_city

    def __local_pheromones_update(self):
        """

        При локальном обновлении каждый муравей обновляет феромоны на полном маршруте, который он прошел в рамках своей
        итерации.

        Правило обновления: new_tau = (1 - rho) * old_tau + rho * tau_delta, tau_delta по алгоритму берется равным tau0

        :return: None
        """
        tau_delta = self.tau0
        for first_city, next_city in zip(self.tour_current[:-1], self.tour_current[1:]):
            old_tau = self.pheromones_on_arcs[first_city][next_city]
            new_tau = (1 - self.rho) * old_tau + self.rho * tau_delta
            self.pheromones_on_arcs[first_city][next_city] = new_tau

    def __global_pheromones_update(self):
        """
        При глобальном обновлении феромоны на всех возможных путях обновляет муравей с кратчайшим на текущий момент
        маршрутом.

        Правило обновления: new_tau = (1 - sigma) * old_tau + sigma * tau_delta,
        tau_delta по алгоритму берется 1 / Lk для дорог, которые входят в наикратчайший маршрут,
        Lk - длина кратчайшего на текущий момент маршрута.
        Для дорог, не входящих в кратчайший маршрут, tau_delta = 0.

        :return: None
        """
        tau_delta = 1 / self.global_min_tour_length if self.global_min_tour_length != 0 else 0
        self.pheromones_on_arcs *= (1 - self.sigma)
        for first_city, next_city in zip(self.global_min_tour[:-1], self.global_min_tour[1:]):
            self.pheromones_on_arcs[first_city][next_city] += self.sigma * tau_delta
        
    def __update_optimum(self, curr_optimal_length, curr_optimal_tour):
        """

        Обновление оптимального на текущий момент маршрута.

        Если какой-то муравей на какой-то итерации нашел маршрут короче, чем текущий оптимум, обновляем оптимум.
        В конце алгоритма результат - самый оптимальный маршрут, найденный на iters_count итераций ants_count муравьями.

        :param curr_optimal_length: текущая длина оптимального маршрута
        :param curr_optimal_tour: города, входящие в текущий оптимальный маршрут
        :return: (curr_optimal_length, curr_optimal_tour): обновленные (либо прежние) длина и маршрут
        """
        if curr_optimal_length == 0 or self.tour_length < curr_optimal_length:
            return (self.tour_length, self.tour_current)
        else:
            return (curr_optimal_length, curr_optimal_tour)
    
    def __nearest_neighbor_heuristic(self):
        """

        Поиск грубой оценки оптимального маршрута методом NNH.
        Согласно методу, муравей, начиная из определенного (здесь - самого первого) города, в качестве следующего
        выбирает город, до которого ему ближе всего относительно текущего.

        :return: min_length: найденная грубая оценка оптимального маршрута
        """
        possible_cities = list(self._cities)
        min_length = 0
        current_city = 0
        possible_cities.remove(current_city)
        while len(possible_cities) != 0:
            min_arc = np.min(self.distances[current_city].take(possible_cities))
            min_length += min_arc
            next_city_index = int(np.where(self.distances[current_city].take(possible_cities) == min_arc)[0][0])
            next_city = possible_cities[next_city_index]
            possible_cities.remove(next_city)
            current_city = next_city
        min_length += self.distances[current_city][0]
        return min_length

def input_matrix():
    """

    Ввод матрицы смежности
    Вводится первая строчка, после чего на основании ее длины будет подсчитано, сколько еще строк нужно.
    Если в рамках любой строки, следующей за первой, было введено больше значений, чем столбцов в первой строке,
    то "конец" введенного будет "обрезан"

    :return: distances: матрица смежности
    """
    distances = []
    try:
        distances.append([int(num) for num in input().split()])
        for num in range(len(distances[0]) - 1):
            distances.append([int(i) for i in input().split()[:len(distances[0])]])
    except:
        raise ValueError('Inappropriate symbol has been met (not int)')
    return distances

if __name__ == '__main__':
    distances = input_matrix()
    x = Colony(distances)
    print(x.ants_full_search())
