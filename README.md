# Ant Colony System for Travelling Salesman Problem
Алгоритм ACS для поиска решения задачи коммивояжера.

Используется классический алгоритм, описанный в 1996 году Дориго и Гамбарделла.

## Описание
На вход алгоритм получает матрицу смежности полносвязного графа, где координаты строк и столбцов - "индексы" соответствующих городов. Отсчет идет от нуля.

Задаваемые параметры алгоритма:
* **distances**: матрица расстояний между пунктами
* **alpha**: коэффициент влияния феромонов на выбор пути
* **beta**: коэффициент влияния расстояния между городами на выбор пути
* **rho**: скорость испарения феромона
* **q0**: параметр, определяющий влияние случайности при выборе следующего города муравьем (чем q0 больше, тем менее случаен выбор)

Задаваемые параметры были подобраны путем тестирования алгоритма на 9 городах

Рассчитываемые параметры:
* **ants_count**: количество муравьев в колонии
* **iters_count**: количество запусков поиска колонией оптимального пути (итераций)
* **min_nn_heuristic**: грубая оценка оптимального маршрута, полученная методом Nearest Neighbor Heuristic (в алгоритме обозначена как *Lk*)
* **tau0**: начальное количество феромона на каждой ветви графа
* **sigma**: параметр, определяющий изменение феромона при глобальном обновлении феромонов (по смыслу равен *rho*, в алгоритме называется *alpha*)
* **pheromones_on_arcs**: распределение феромонов по дорогам (число феромонов на каждой дороге обозначается *tau*)
* **tour_current**: полный маршрут, пройденный муравьем
* **tour_length**: длина маршрута, пройденного муравьем, от начала до конца
* **non_visited_cities**: список непосещенных разрешенных к посещению городов относительно текущего
* **global_min_tour**: глобально минимальный маршрут
* **global_min_tour_length**: длина глобально минимального маршрута
            
### Алгоритм поиска
На каждой итерации из *iters_count* муравьи в количестве *ants_count* ищут маршрут исходя из соображения:
1. Для каждого муравья при его начале движения случайным образом выбирается первый город
2. Для городов из списка разрешенных непосещенных:
  * Рассчитывается вероятность перехода в город
  * Выбирается случайное равномерно распределенное число *q*
  * Если *q < q0*, то выбирается город, для которого значение "привлекательности" максимально
  * Иначе город выбирается случайно, но с учетом вероятностей, рассчитанных ранее
3. Муравей обновляет собственный список разрешенных непосещенных городов, обновляет текущий маршрут и его длину
4. Если список непосещенных городов не пуст, то возврат к пункту 2, иначе пункт 5
5. Муравей, построив маршрут и подсчитав его длину, обновляет феромоны на этом маршруте по формуле *(1 - rho) * tau + rho * delta_tau*, здесь *delta_tau*, согласно алгоритму, равно *tau0*
6. При необходимости обновляются глобально минимальный маршрут и его длина

Когда все муравьи закончили путешествия, перед окончанием итерации феромоны обновляются глобально для всех возможных дорог
Обновление по формуле *(1 - sigma) * tau + sigma * delta_tau*, где *delta_tau = 1/Lk*, если обновляется феромон на дороге, входящей в глобально минимальный маршрут, и *delta_tau = 0*, если дорога в минимальный маршрут не входит

После завершения всех итераций возвращается найденный наименьший маршрут (его длина).

## Запуск

+ import ACS
+ alg = ACS.Colony(distances) # distances можно задать самостоятельно как квадратную матрицу, либо вызвать ACS.distances для ручного ввода
+ solution = alg.ants_full_search()
