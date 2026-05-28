import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.initialize()

    def initialize(self):
        # ======================== INITIALIZE NETWORK WEIGTHS AND BIASES =============================
        self.input_size = 9
        self.hidden_size = 10
        self.output_size = 3

        self.w1 = np.random.uniform(-1.0, 1.0, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(-1.0, 1.0, self.hidden_size)
        self.w2 = np.random.uniform(-1.0, 1.0, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(-1.0, 1.0, self.output_size)

        # ============================================================================================

    def think(self, dino_rect, obstacle_rect, score=0, game_speed=20):
        # ======================== PROCESS INFORMATION SENSED TO ACT =============================
        distance = obstacle_rect.x - dino_rect.x
        is_flying = 1.0 if obstacle_rect.y < 260 else 0.0
        inputs = np.array([
            obstacle_rect.x / 1100,
            obstacle_rect.y / 600,
            obstacle_rect.width / 100,
            obstacle_rect.height / 100,
            distance / 1100,
            is_flying,
            dino_rect.y / 600,
            game_speed / 50,
            min(score, 1000) / 1000
        ], dtype=float)

        hidden = self.sigmoid(np.dot(inputs, self.w1) + self.b1)
        result = self.sigmoid(np.dot(hidden, self.w2) + self.b2)

        # ========================================================================================
        return self.act(result)

    def act(self, output):
        # ======================== USE THE ACTIVATION FUNCTION TO ACT =============================
        action = int(np.argmax(output))

        # =========================================================================================
        if (action == 0):
            return "JUMP"
        elif (action == 1):
            return "DUCK"
        elif (action == 2):
            return "RUN"

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_genome(self):
        return np.concatenate([
            self.w1.ravel(),
            self.b1.ravel(),
            self.w2.ravel(),
            self.b2.ravel()
        ])

    def set_genome(self, genome):
        genome = np.asarray(genome, dtype=float)
        cursor = 0

        w1_size = self.input_size * self.hidden_size
        self.w1 = genome[cursor:cursor + w1_size].reshape(self.input_size, self.hidden_size)
        cursor += w1_size

        self.b1 = genome[cursor:cursor + self.hidden_size]
        cursor += self.hidden_size

        w2_size = self.hidden_size * self.output_size
        self.w2 = genome[cursor:cursor + w2_size].reshape(self.hidden_size, self.output_size)
        cursor += w2_size

        self.b2 = genome[cursor:cursor + self.output_size]
