from ursina import (
    Ursina, Button, Entity, camera, window, color,
    held_keys, mouse, Text
)
from ursina.prefabs.first_person_controller import FirstPersonController
import random, time

# ── Game States ──
MENU, WORLD_VIEW, FIRST_PERSON = range(3)

class UniverseGame:
    def __init__(self, app):
        self.app      = app
        window.title      = 'Genesis'
        window.borderless = False
        self.state        = MENU
        self.paused       = False

        # ── MENU UI ──
        self.create_btn = Button(
            text     = 'CREATE UNIVERSE',
            color    = color.azure.tint(-.2),
            scale    = (.4, .1),
            position = (0, 0)
        )
        self.create_btn.on_click = self.start_universe

        # ── placeholders ──
        self.world_parent = Entity()
        self.player       = None

        # ── hook app callbacks ──
        app.update = self.update
        app.input  = self.input

        # ── Pause overlay ──
        self.pause_text = Text(
            'PAUSED',
            origin=(0,0),
            scale=2,
            enabled=False,
            color=color.yellow
        )

    def start_universe(self):
        self.create_btn.disable()
        self.state = WORLD_VIEW

        GRID = 30
        self.tile_colors = {
            'empty': color.light_gray,
            'food' : color.lime,
            'water': color.azure
        }
        # build the floor
        self.world = [
            [random.choice(list(self.tile_colors)) for x in range(GRID)]
            for y in range(GRID)
        ]
        for y in range(GRID):
            for x in range(GRID):
                Entity(
                    parent   = self.world_parent,
                    model    = 'cube',
                    position = (x, 0, y),
                    scale    = (1, .1, 1),
                    color    = self.tile_colors[self.world[y][x]],
                    collider = 'box'
                )

        # overhead camera
        camera.position   = (GRID/2, 30, GRID/2)
        camera.rotation_x = 90
        camera.fov        = 60
        mouse.locked      = False   # free cursor in map mode

        # reincarnate button
        self.incarnate_btn = Button(
            text     = 'REINCARNATE',
            color    = color.orange,
            scale    = (.2, .05),
            position = (0, -.45)
        )
        self.incarnate_btn.on_click = self.enter_first_person

    def enter_first_person(self):
        if self.state != WORLD_VIEW: return
        self.state = FIRST_PERSON
        self.incarnate_btn.disable()

        # spawn FPS controller
        self.player = FirstPersonController()
        self.player.position = (len(self.world)//2, 2, len(self.world)//2)
        mouse.locked = True       # capture mouse for look

    def update(self):
        # don't do anything while paused
        if self.paused:
            return

        # ── map-mode pan ──
        if self.state == WORLD_VIEW:
            speed = 20 * time.dt
            if held_keys['a'] or held_keys['left arrow']:
                camera.x -= speed
            if held_keys['d'] or held_keys['right arrow']:
                camera.x += speed
            if held_keys['w'] or held_keys['up arrow']:
                camera.z -= speed
            if held_keys['s'] or held_keys['down arrow']:
                camera.z += speed

        # ── pass input to FPS controller ──
        elif self.state == FIRST_PERSON and self.player:
            # FirstPersonController has its own update, so nothing extra here
            pass

    def input(self, key):
        # ── Pause toggle ──
        if key.lower() == 'p':
            self.paused = not self.paused
            self.pause_text.enabled = self.paused
            # unlock mouse when paused
            mouse.locked = not self.paused

        # ── ESC to exit first-person ──
        if key == 'escape' and self.state == FIRST_PERSON:
            self.player.disable()
            mouse.locked = False
            camera.position   = (len(self.world)/2, 30, len(self.world)/2)
            camera.rotation_x = 90
            self.incarnate_btn.enable()
            self.state = WORLD_VIEW

        # ── Forward input to FPS controller ──
        if self.state == FIRST_PERSON and self.player:
            self.player.input(key)


if __name__ == '__main__':
    app  = Ursina()
    game = UniverseGame(app)
    app.run()
