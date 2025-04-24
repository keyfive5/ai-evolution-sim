import pygame
import random

#add Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#Create Window
GRID_SIZE = 20
TILE_SIZE = 32
WINDOW_WIDTH = GRID_SIZE * TILE_SIZE
WINDOW_HEIGHT = GRID_SIZE * TILE_SIZE

#Start game and time
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()

# World map grid
world = [[random.choice(['empty', 'food', 'water']) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

colors = {
    'empty': (200, 200, 200),
    'food': (0, 255, 0),
    'water': (0, 0, 255),
}

def draw_world():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            pygame.draw.rect(
                screen, colors[world[y][x]],
                pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            )
class Agent:
    def __init__(self):
        self.x = random.randint(0, GRID_SIZE-1)
        self.y = random.randint(0, GRID_SIZE-1)
        self.color = (255, 0, 0)
        self.brain = AgentBrain()

    def draw(self):
        pygame.draw.rect(screen, self.color,
            pygame.Rect(self.x*TILE_SIZE, self.y*TILE_SIZE, TILE_SIZE, TILE_SIZE))

    def sense(self):
        # Get 3x3 grid centered on the agent
        view = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = max(0, min(GRID_SIZE-1, self.x + dx))
                ny = max(0, min(GRID_SIZE-1, self.y + dy))
                tile = world[ny][nx]
                view.append(self.tile_to_number(tile))
        return torch.tensor(view, dtype=torch.float32)

    def tile_to_number(self, tile):
        return {
            'empty': 0,
            'food': 1,
            'water': 2
        }[tile]

    def move(self):
        input_vec = self.sense()
        output = self.brain(input_vec)
        #add randomness to output for now
        action_probs = F.softmax(output, dim=0).detach().numpy()
        action = random.choices([0, 1, 2, 3], weights=action_probs)[0]


        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        self.x = max(0, min(GRID_SIZE-1, self.x + dx))
        self.y = max(0, min(GRID_SIZE-1, self.y + dy))


#define agent brain (simple neural net)
class AgentBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)  # 3x3 view = 9 tiles
        self.fc2 = nn.Linear(16, 4)  # output = up, down, left, right

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits (no softmax)


agent = Agent()
running = True

while running:
    screen.fill((0, 0, 0))
    draw_world()
    agent.move()
    agent.draw()
    pygame.display.flip()
    clock.tick(5)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
