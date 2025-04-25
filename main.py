import pygame
import random

pygame.font.init()
font = pygame.font.SysFont(None, 24)

#add Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from opensimplex import OpenSimplex

POPULATION        = 20
GENERATION_STEPS  = 100
MUTATION_RATE     = 0.1
PROPHET_INTERVAL = 5   # every 5 generations
PROPHET_COUNT    = 2   # how many prophets to drop in


#Create Window
GRID_SIZE = 20
TILE_SIZE = 32
WINDOW_WIDTH = GRID_SIZE * TILE_SIZE
WINDOW_HEIGHT = GRID_SIZE * TILE_SIZE

UPDATE_PER_FRAME = 1   # start with 1 simulation step per draw

# tuning knobs
EVAPORATION = 0.90   # how quickly trails fade each step
DIFFUSION   = 0.10     # how much spreads to neighbors
DEPOSIT     = 0.2    # how much each broadcast addsupd


# generate a smooth heightmap via OpenSimplex noise
simplex = OpenSimplex(seed=random.randrange(100000))
world = []
scale = 20.0  # tweak for larger/smaller biomes
for y in range(GRID_SIZE):
    row = []
    for x in range(GRID_SIZE):
        # noise2 returns −1…1
        h = simplex.noise2(x/scale, y/scale)
        if h < -0.2:
            row.append('water')
        elif h < 0.0:
            row.append('empty')   # beach/desert
        elif h < 0.5:
            row.append('food')    # grass/plains
        else:
            row.append('water')   # mountains are “dry” water for now
    world.append(row)

#Start game and time
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()


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
        self.score = 0
        self.message = 0  # 0 or 1 for now


    def draw(self):
        pygame.draw.rect(screen, self.color,
            pygame.Rect(self.x*TILE_SIZE, self.y*TILE_SIZE, TILE_SIZE, TILE_SIZE))

    def sense(self):
        # Get 3×3 grid centered on the agent
        view = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = max(0, min(GRID_SIZE-1, self.x + dx))
                ny = max(0, min(GRID_SIZE-1, self.y + dy))
                tile = world[ny][nx]
                view.append(self.tile_to_number(tile))

        # Sum up neighbor messages from the message board
        msg_sum = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = max(0, min(GRID_SIZE-1, self.x + dx))
                ny = max(0, min(GRID_SIZE-1, self.y + dy))
                msg_sum += message_board[ny][nx]

        # Return a length-10 tensor: 9 tile values + message sum
        return torch.tensor(view + [msg_sum], dtype=torch.float32)

                

    def tile_to_number(self, tile):
        return {
            'empty': 0,
            'food': 1,
            'water': 2
        }[tile]

    def move(self):
        input_vec = self.sense()
        output = self.brain(input_vec)
        action_probs = F.softmax(output, dim=0).detach().numpy()
        action = random.choices([0,1,2,3], weights=action_probs)[0]

        dx, dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
        self.x = max(0, min(GRID_SIZE-1, self.x + dx))
        self.y = max(0, min(GRID_SIZE-1, self.y + dy))

        # —— scoring —— 
        tile = world[self.y][self.x]
        if   tile == 'food':  self.score += 10
        elif tile == 'water': self.score -= 5
        else:                  self.score -= 1

        # —— broadcast message based on current tile —— 
        self.message = 1 if tile == 'food' else 0




#define agent brain (simple neural net)
class AgentBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 4)  # output = up, down, left, right

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits (no softmax)

class Prophet(Agent):
    def __init__(self):
        super().__init__()
        self.color = (255, 0, 255)   # magenta, not blue
    def move(self):
        dx, dy = random.choice([(0,-1),(0,1),(-1,0),(1,0)])
        self.x = max(0, min(GRID_SIZE-1, self.x + dx))
        self.y = max(0, min(GRID_SIZE-1, self.y + dy))
        # prophets just broadcast—they no longer terraform
        self.message = 1

    def draw(self):
        rect = pygame.Rect(self.x*TILE_SIZE, self.y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, self.color, rect)
        pygame.draw.rect(screen, (0,0,0), rect, width=2)  # black outline
        if self.message==1:
            # yellow dot in center
            pygame.draw.circle(screen, (255,255,0), rect.center, TILE_SIZE//4)



#replaced agent with agents
agents = [Agent() for _ in range(20)]  # Create 20 agents
# after agents = [...]
message_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
pheromone = [[0.0]*GRID_SIZE for _ in range(GRID_SIZE)]
step_counter = 0
generation   = 1
avg_scores = []
max_scores = []
running = True

while running:
    # ── Simulation updates ──
    for _ in range(UPDATE_PER_FRAME):
       # ─── Evaporate & diffuse pheromones ───
        new_pher = [[0.0]*GRID_SIZE for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                # evaporate
                val = pheromone[y][x] * EVAPORATION
                # simple 8-neighborhood diffusion
                neigh = 0
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dx==0 and dy==0: continue
                        ny, nx = (y+dy)%GRID_SIZE, (x+dx)%GRID_SIZE
                        neigh += pheromone[ny][nx]
                val += (neigh/8) * DIFFUSION
                new_pher[y][x] = val
        pheromone = new_pher

        # ─── Move agents & deposit pheromones ───
        # 1) clear message_board
        message_board = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
        # 2) move + deposit
        for agent in agents:
            agent.move()
            message_board[agent.y][agent.x] = agent.message
            if agent.message == 1:
                pheromone[agent.y][agent.x] += DEPOSIT

        # 3) Evolution check
        step_counter += 1
        if step_counter >= GENERATION_STEPS:
            # --- sort by score ---
            agents.sort(key=lambda a: a.score, reverse=True)
            survivors = agents[:POPULATION//2]

            # --- reproduce + mutate ---
            new_agents = []
            for parent in survivors:
                new_agents.append(parent)
                child = Agent()
                child.brain.load_state_dict(parent.brain.state_dict())
                for p in child.brain.parameters():
                    p.data += MUTATION_RATE * torch.randn_like(p)
                new_agents.append(child)

            # ── PROPHET INJECTION (exactly here) ──
            if generation % PROPHET_INTERVAL == 0:
                for _ in range(PROPHET_COUNT):
                    p = Prophet()
                    p.x = random.randrange(GRID_SIZE)
                    p.y = random.randrange(GRID_SIZE)
                    new_agents.append(p)

            # --- record stats ---
            scores = [a.score for a in agents]
            avg_scores.append(sum(scores)/len(scores))
            max_scores.append(max(scores))

            # --- reset population, but keep prophets ---
            prophets = [a for a in new_agents if isinstance(a, Prophet)]
            others  = [a for a in new_agents if not isinstance(a, Prophet)]

            # fill up to POPULATION: prophets first, then top others
            agents = prophets + others[: POPULATION - len(prophets) ]

            # in case you accidentally overfill:
            agents = agents[:POPULATION]

            # now reset each agent’s score & position
            for a in agents:
                a.score = 0
                a.x = random.randrange(GRID_SIZE)
                a.y = random.randrange(GRID_SIZE)


            step_counter = 0
            generation += 1
            print(f"=== Generation {generation} ===")

    # ── Rendering ──
    screen.fill((0,0,0))
    draw_world()
    # ── INSERT HEATMAP OVERLAY HERE ──
    heat_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            v = pheromone[y][x]
            if v <= 0: 
                continue
            # map v (0…) to 0…1 (clamp), then to 0…255
            t = max(0.0, min(1.0, v * 2.0))  
            iv = int(t * 255)
            # now pick a color ramp from blue→green→red
            if t < 0.5:
                # blue→green
                r = 0
                g = int(t * 2 * 255)
                b = 255 - g
            else:
                # green→red
                r = int((t - 0.5) * 2 * 255)
                g = 255 - r
                b = 0
            a = iv // 3   # make it a bit more transparent
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            heat_surf.fill((r, g, b, a), rect)
    screen.blit(heat_surf, (0, 0))
    for agent in agents:
        agent.draw()

    # Stats + speed display
    screen.blit(font.render(f"Gen: {generation}", True, (255,255,255)), (10,10))
    if avg_scores:
        screen.blit(font.render(f"Avg: {avg_scores[-1]:.1f}", True, (255,255,255)), (10,30))
        screen.blit(font.render(f"Max: {max_scores[-1]:.1f}", True, (255,255,255)), (10,50))
    screen.blit(font.render(f"Speed: {UPDATE_PER_FRAME}×", True, (255,255,255)), (10,70))

    pygame.display.flip()
    clock.tick(5)    # ← set to 5 FPS so “1×” is slow

    # ── Event handling ──
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                UPDATE_PER_FRAME = min(10, UPDATE_PER_FRAME+1)
            elif event.key == pygame.K_DOWN:
                UPDATE_PER_FRAME = max(1, UPDATE_PER_FRAME-1)
        elif event.type == pygame.QUIT:
            running = False


