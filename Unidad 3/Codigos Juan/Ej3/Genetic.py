import random
import numpy as np

ELITE_RATE = 0.05
MUTATION_RATE = 0.12
MUTATION_SIGMA = 0.25
LOCAL_SEARCH_STEPS = 4
WEIGHT_LIMIT = 5.0
SELECTION_PRESSURE = 3

def updateNetwork(population):
    # ===================== ESTA FUNCION RECIBE UNA POBLACION A LA QUE SE DEBEN APLICAR MECANISMOS DE SELECCION, =================
    # ===================== CRUCE Y MUTACION. LA ACTUALIZACION DE LA POBLACION SE APLICA EN LA MISMA VARIABLE ====================
    if not population:
        return

    population_size = len(population)
    ranked_indexes = sorted(range(population_size), key=lambda i: population[i].score, reverse=True)
    elite_count = max(1, int(population_size * ELITE_RATE))
    selected_pairs = select_fittest(population)

    new_genomes = [population[index].get_genome().copy() for index in ranked_indexes[:elite_count]]

    while len(new_genomes) < population_size:
        parent1_index, parent2_index = selected_pairs[len(new_genomes) % len(selected_pairs)]
        child_genome = evolve(population[parent1_index], population[parent2_index])
        new_genomes.append(child_genome)

    best_score = population[ranked_indexes[0]].score
    average_score = sum(dino.score for dino in population) / population_size
    print(f"Mejor score: {best_score} | Promedio: {average_score:.2f}")

    for dino, genome in zip(population, new_genomes):
        dino.set_genome(genome)

    # =============================================================================================================================

def select_fittest(population):
    # ===================== FUNCION DE SELECCION =====================
    scores = np.array([max(0, dino.score) for dino in population], dtype=float)
    max_score = np.max(scores)

    if max_score == 0:
        probabilities = np.ones(len(population)) / len(population)
    else:
        normalized_scores = scores / max_score
        fitness = np.power(normalized_scores, SELECTION_PRESSURE) + 0.01
        probabilities = fitness / np.sum(fitness)

    pairs = []
    for _ in range(len(population)):
        parent_indexes = np.random.choice(len(population), size=2, replace=True, p=probabilities)
        pairs.append((int(parent_indexes[0]), int(parent_indexes[1])))

    return pairs

    # ================================================================

def evolve(element1, element2):
    # ===================== FUNCION DE CRUCE Y MUTACION =====================
    genome1 = element1.get_genome()
    genome2 = element2.get_genome()

    alpha = np.random.rand(genome1.size)
    child = alpha * genome1 + (1 - alpha) * genome2

    mutation_mask = np.random.rand(child.size) < MUTATION_RATE
    child[mutation_mask] += np.random.normal(0, MUTATION_SIGMA, np.sum(mutation_mask))

    anchor = genome1 if element1.score >= element2.score else genome2
    child = local_search(child, anchor)

    return np.clip(child, -WEIGHT_LIMIT, WEIGHT_LIMIT)

    # ===============================================================

def local_search(genome, anchor):
    best = np.clip(genome.copy(), -WEIGHT_LIMIT, WEIGHT_LIMIT)
    best_distance = np.linalg.norm(best - anchor)
    sigma = MUTATION_SIGMA

    for _ in range(LOCAL_SEARCH_STEPS):
        candidate = best + np.random.normal(0, sigma, best.shape)
        candidate = np.clip(candidate, -WEIGHT_LIMIT, WEIGHT_LIMIT)
        candidate_distance = np.linalg.norm(candidate - anchor)

        if candidate_distance <= best_distance or random.random() < 0.15:
            best = candidate
            best_distance = candidate_distance

        sigma *= 0.5

    return best
