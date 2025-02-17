import numpy as np
from typing import List, Tuple
import logging
from functools import lru_cache
import hashlib
from collections import OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MemoizedDict(OrderedDict):
    """Custom cache with size limit"""
    def __init__(self, maxsize=1000):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if len(self) >= self.maxsize:
            self.popitem(last=False)
        super().__setitem__(key, value)

class SeatingOptimizer:
    def __init__(self, num_students: int, rows: int, cols: int):
        self.num_students = int(num_students)
        self.rows = int(rows)
        self.cols = int(cols)

        # GA parameters
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1

        # Caching systems
        self.fitness_cache = MemoizedDict(maxsize=10000)
        self.neighbor_cache = MemoizedDict(maxsize=1000)
        self.position_scores = {}
        self.layout_cache = MemoizedDict(maxsize=5000)

    def _hash_layout(self, layout: np.ndarray) -> str:
        """Create hash of layout for caching"""
        return hashlib.md5(layout.tobytes()).hexdigest()

    @lru_cache(maxsize=1024)
    def _calculate_position_score(self, row: int, col: int) -> float:
        """Cache score calculations for individual positions"""
        return float(self.rows - row)  # Front row preference

    def calculate_fitness(self, layout: np.ndarray) -> float:
        """Enhanced fitness calculation with caching"""
        layout_hash = self._hash_layout(layout)

        if layout_hash in self.fitness_cache:
            return self.fitness_cache[layout_hash]

        score = 0.0
        occupied_positions = set()

        # Use cached position scores
        for i in range(self.rows):
            for j in range(self.cols):
                if layout[i, j] > 0:
                    pos_key = (i, j)
                    if pos_key not in self.position_scores:
                        self.position_scores[pos_key] = self._calculate_position_score(i, j)
                    score += self.position_scores[pos_key]
                    occupied_positions.add(pos_key)

                    # Calculate spacing score
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if (0 <= i + di < self.rows and 
                                0 <= j + dj < self.cols and 
                                (di != 0 or dj != 0)):
                                if layout[i + di, j + dj] == 0:
                                    score += 1.0

        self.fitness_cache[layout_hash] = score
        return score

    @lru_cache(maxsize=512)
    def _get_valid_positions(self) -> tuple:
        """Cache valid position combinations"""
        return tuple((i, j) for i in range(self.rows) 
                    for j in range(self.cols))

    def create_initial_layout(self) -> np.ndarray:
        """Optimized initial layout creation"""
        try:
            layout = np.zeros((self.rows, self.cols), dtype=int)

            # Ensure we have enough space for students
            if self.num_students > self.rows * self.cols:
                raise ValueError("Too many students for room size")

            # Get available positions
            available_positions = [(i, j) for i in range(self.rows) 
                                 for j in range(self.cols)]

            # Randomly place students
            selected_positions = np.random.choice(
                len(available_positions),
                self.num_students,
                replace=False
            )

            for student_id, pos_idx in enumerate(selected_positions, 1):
                i, j = available_positions[pos_idx]
                layout[i, j] = student_id

            return layout

        except Exception as e:
            logger.error(f"Failed to create initial layout: {e}")
            return None

    def _generate_neighbor(self, layout: np.ndarray) -> np.ndarray:
        """Generate neighbor with improved caching"""
        layout_hash = self._hash_layout(layout)
        mod_type = np.random.choice(['swap', 'shift', 'rotate'])

        cache_key = (layout_hash, mod_type)
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key].copy()

        neighbor = layout.copy()

        if mod_type == 'swap':
            occupied = [(i, j) for i in range(self.rows) 
                       for j in range(self.cols) if layout[i, j] > 0]
            if len(occupied) >= 2:
                pos1, pos2 = np.random.choice(len(occupied), 2, replace=False)
                i1, j1 = occupied[pos1]
                i2, j2 = occupied[pos2]
                neighbor[i1, j1], neighbor[i2, j2] = neighbor[i2, j2], neighbor[i1, j1]

        elif mod_type == 'shift':
            if np.random.random() < 0.5:
                row = np.random.randint(self.rows)
                direction = 1 if np.random.random() < 0.5 else -1
                neighbor[row] = np.roll(neighbor[row], direction)
            else:
                col = np.random.randint(self.cols)
                direction = 1 if np.random.random() < 0.5 else -1
                neighbor[:, col] = np.roll(neighbor[:, col], direction)

        else:  # rotate
            if self.rows >= 2 and self.cols >= 2:
                i = np.random.randint(self.rows - 1)
                j = np.random.randint(self.cols - 1)
                section = neighbor[i:i+2, j:j+2].copy()
                neighbor[i:i+2, j:j+2] = np.rot90(section)

        self.neighbor_cache[cache_key] = neighbor
        return neighbor.copy()

    @lru_cache(maxsize=128)
    def _get_tournament_indices(self, k: int, size: int) -> tuple:
        """Cache tournament selection indices"""
        return tuple(np.random.choice(size, k, replace=False))

    def _tournament_select(self, population: List[np.ndarray], 
                         fitness_scores: List[float]) -> np.ndarray:
        """Optimized tournament selection"""
        k = 3
        selected = self._get_tournament_indices(k, len(population))
        tournament_fitness = [fitness_scores[i] for i in selected]
        winner_idx = selected[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Optimized crossover with caching"""
        parents_hash = self._hash_layout(parent1) + self._hash_layout(parent2)
        if parents_hash in self.layout_cache:
            return self.layout_cache[parents_hash].copy()

        child = np.zeros((self.rows, self.cols), dtype=int)
        students = set(range(1, self.num_students + 1))
        used_students = set()

        mask = np.random.rand(self.rows, self.cols) < 0.5
        for i in range(self.rows):
            for j in range(self.cols):
                if mask[i, j] and parent1[i, j] > 0:
                    if parent1[i, j] not in used_students:
                        child[i, j] = parent1[i, j]
                        used_students.add(parent1[i, j])

        remaining_students = students - used_students
        remaining_positions = [(i, j) for i in range(self.rows) 
                             for j in range(self.cols) if child[i, j] == 0]

        np.random.shuffle(remaining_positions)
        for student, (i, j) in zip(remaining_students, remaining_positions):
            child[i, j] = student

        self.layout_cache[parents_hash] = child
        return child.copy()

    def _mutate(self, layout: np.ndarray) -> np.ndarray:
        """Optimized mutation"""
        layout_hash = self._hash_layout(layout)
        if layout_hash in self.layout_cache:
            return self.layout_cache[layout_hash].copy()

        new_layout = layout.copy()
        occupied = [(i, j) for i in range(self.rows) 
                   for j in range(self.cols) if layout[i, j] > 0]

        if len(occupied) >= 2:
            pos1, pos2 = np.random.choice(len(occupied), 2, replace=False)
            i1, j1 = occupied[pos1]
            i2, j2 = occupied[pos2]
            new_layout[i1, j1], new_layout[i2, j2] = \
                new_layout[i2, j2], new_layout[i1, j1]

        self.layout_cache[layout_hash] = new_layout
        return new_layout.copy()

    def simulated_annealing(self, initial_layout: np.ndarray, initial_temp: float = 10.0, 
                        cooling_rate: float = 0.95, iterations: int = 100) -> Tuple[np.ndarray, float]:
        """Simulated annealing with caching"""
        current_layout = initial_layout.copy()
        current_fitness = self.calculate_fitness(current_layout)

        best_layout = current_layout.copy()
        best_fitness = current_fitness

        temp = initial_temp

        for iteration in range(iterations):
            neighbor = self._generate_neighbor(current_layout)
            neighbor_fitness = self.calculate_fitness(neighbor)

            delta = neighbor_fitness - current_fitness
            acceptance_prob = np.exp(delta / temp) if temp > 0 else 0

            if delta > 0 or np.random.random() < acceptance_prob:
                current_layout = neighbor.copy()
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness:
                    best_layout = current_layout.copy()
                    best_fitness = current_fitness

            temp *= cooling_rate

            if iteration % 10 == 0:
                print(f"SA Iteration {iteration}: Temp = {temp:.2f}, Best Fitness = {best_fitness:.2f}")

        return best_layout, best_fitness

    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run hybrid optimization with GA and SA"""
        try:
            best_layout = None
            best_fitness = float('-inf')

            # Create and validate initial population
            population = []
            attempts = 0
            max_attempts = self.population_size * 2

            while len(population) < self.population_size and attempts < max_attempts:
                layout = self.create_initial_layout()
                if layout is not None and layout.shape == (self.rows, self.cols):
                    population.append(layout)
                attempts += 1

            if not population:
                raise ValueError("Could not create valid initial population")

            # Ensure we have a valid starting point
            best_layout = population[0]
            best_fitness = self.calculate_fitness(best_layout)

            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = []
                valid_layouts = []

                for layout in population:
                    try:
                        fitness = self.calculate_fitness(layout)
                        fitness_scores.append(fitness)
                        valid_layouts.append(layout)
                    except Exception as e:
                        logger.warning(f"Skipping invalid layout: {e}")

                if not valid_layouts:
                    logger.error("No valid layouts in population")
                    break

                population = valid_layouts

                # Find best solution
                best_idx = np.argmax(fitness_scores)
                if fitness_scores[best_idx] > best_fitness:
                    best_fitness = fitness_scores[best_idx]
                    best_layout = population[best_idx].copy()

                # Apply SA periodically
                if generation % 10 == 0:
                    try:
                        sa_layout, sa_fitness = self.simulated_annealing(
                            best_layout,
                            initial_temp=10.0 * (1 - generation/self.generations),
                            iterations=50
                        )
                        if sa_fitness > best_fitness:
                            best_layout = sa_layout
                            best_fitness = sa_fitness
                    except Exception as e:
                        logger.warning(f"SA optimization failed: {e}")

                # Create new population
                new_population = [best_layout]  # Elitism

                while len(new_population) < self.population_size:
                    try:
                        parent1 = self._tournament_select(population, fitness_scores)
                        parent2 = self._tournament_select(population, fitness_scores)
                        child = self._crossover(parent1, parent2)

                        if np.random.random() < self.mutation_rate:
                            child = self._mutate(child)

                        if child is not None and child.shape == (self.rows, self.cols):
                            new_population.append(child)
                    except Exception as e:
                        logger.warning(f"Failed to create new individual: {e}")

                population = new_population

                if generation % 10 == 0:
                    print(f"Generation {generation}: Best Fitness = {best_fitness}")

            # Validate final solution
            if best_layout is None or best_layout.shape != (self.rows, self.cols):
                raise ValueError("Failed to find valid solution")

            # Final SA refinement
            try:
                final_layout, final_fitness = self.simulated_annealing(
                    best_layout,
                    initial_temp=5.0,
                    cooling_rate=0.98,
                    iterations=200
                )

                if final_fitness > best_fitness:
                    best_layout = final_layout
                    best_fitness = final_fitness
            except Exception as e:
                logger.warning(f"Final SA refinement failed: {e}")

            return best_layout, best_fitness

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
