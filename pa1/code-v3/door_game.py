import pygame
import numpy as np
import sys
from typing import Optional

# =========================================================
# ===============   ENVIRONMENT (Poisson)   ===============
# =========================================================

class PoissonDoorsEnv:
    """
    This creates a Poisson environment. There are K doors and each has an associated mean.
    In each step you pick an arm i. Damage to a door is drawn from its corresponding
    Poisson Distribution. Initial health of each door is H0 and decreases by damage in each step.
    Game end when any door's health < 0.
    """
    def __init__(self, mus, H0=100, rng=None):
        self.mus = np.array(mus, dtype=float)
        self.K = len(mus)
        self.H0 = H0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.health = np.full(self.K, self.H0, dtype=int)
        self.total_hits = np.zeros(self.K, dtype=int)
        self.total_loss = np.zeros(self.K, dtype=int)
        return self.health.copy()

    def step(self, arm: int):
        reward = int(self.rng.poisson(self.mus[arm]))
        self.health[arm] -= reward
        self.total_hits[arm] += 1
        self.total_loss[arm] += reward
        done = np.any(self.health < 0)
        return reward, done, {
            "reward": reward,
            "health": self.health.copy(),
            "hits": self.total_hits.copy(),
            "loss": self.total_loss.copy()
        }

# =========================================================
# =====================   POLICIES   ======================
# =========================================================

class Policy:
    """
    Base Policy interface.
    - Implement select_arm(self, t) -> int in [0, K-1].
    - Optionally override update(...) for custom learning.
    """
    def __init__(self, K, rng: Optional[np.random.Generator] = None):
        self.K = K
        self.counts = np.zeros(K, dtype=int)
        self.values = np.zeros(K, dtype=float)
        self.current_health = np.ones(K, dtype=int)
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_arm(self, t: int) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: int, current_health: np.ndarray):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.current_health = current_health.copy()

# --------------- TASK 2 CHANGES HERE ----------------
class StudentPolicy(Policy):
    """
    Implement your own algorithm here.
    In Code Mode, only StudentPolicy runs.
    Replace select_arm with your strategy.
    Currently it has a simple implementation of the epsilon greedy strategy.
    """
    def __init__(self, K, rng=None, eps: float = 0.1):
        super().__init__(K, rng)
        self.eps = eps

    def select_arm(self, t: int) -> int:
        untried = np.where(self.counts == 0)[0]
        if untried.size > 0:
            return int(untried[0])

        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.K))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: int, current_health: np.ndarray):
        super().update(arm, reward, current_health)

# =========================================================
# ==================   HAMMER ANIMATION   =================
# =========================================================

class HammerAnimation:
    def __init__(self):
        self.active = False
        self.t = 0
        self.duration = 14
        self.start = (0, 0)
        self.end = (0, 0)
        self.pivot_angle = -35
        self.angle_swing = 70
        self.scale = 1.0

    def trigger(self, target_rect: pygame.Rect):
        cx = target_rect.centerx
        cy = target_rect.y + 10
        self.end = (cx, cy)
        self.start = (cx - 140, cy - 120)
        self.t = 0
        self.active = True

    @staticmethod
    def ease_out_quad(x: float) -> float:
        return 1 - (1 - x) * (1 - x)

    def update_and_draw(self, surface: pygame.Surface):
        if not self.active:
            return

        if self.t >= self.duration:
            self.active = False
            return

        self.t += 1
        p = self.ease_out_quad(self.t / self.duration)
        x = self.start[0] + (self.end[0] - self.start[0]) * p
        y = self.start[1] + (self.end[1] - self.start[1]) * p
        angle = self.pivot_angle + self.angle_swing * p

        hammer_surf = pygame.Surface((90, 90), pygame.SRCALPHA)

        pygame.draw.rect(hammer_surf, (120, 80, 40), pygame.Rect(40, 10, 10, 60), border_radius=4)

        pygame.draw.rect(hammer_surf, (180, 180, 180), pygame.Rect(20, 60, 50, 18), border_radius=4)
        pygame.draw.circle(hammer_surf, (160, 160, 160), (23, 69), 9)

        rotated = pygame.transform.rotozoom(hammer_surf, angle, self.scale)
        rect = rotated.get_rect(center=(x, y))
        surface.blit(rotated, rect.topleft)

# =========================================================
# ==================   PYGAME SIMULATOR   =================
# =========================================================

class DungeonGame:
    """
    Controls:
      - C: Code mode; M: Manual mode
      - Space: Play/Pause  N: Single-step (when paused)
      - R: Reset  P: Reset with sample Poisson params
      - Click a door in Manual mode to hit it.
    In Code Mode, only StudentPolicy runs.
    """
    def __init__(self, env, policy=None, code_mode=False, env_name="Poisson"):
        pygame.init()
        self.env = env
        self.policy = policy
        self.code_mode = code_mode
        self.env_name = env_name

        self.width, self.height = 1100, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dungeon Doors")
        self.clock = pygame.time.Clock()

        self.font_header = pygame.font.SysFont("Arial", 28)
        self.font_info   = pygame.font.SysFont("Arial", 18)
        self.font_small  = pygame.font.SysFont("Arial", 16)

        self.total_hits = 0
        self.done = False
        self.step_count = 0

        self.hammer = HammerAnimation()
        self.frame_count = 0
        self.code_step_interval = 22

        self.paused = True
        self.single_step = False

        self._rebuild_layout()

    def _rebuild_layout(self):
        base_font = 24
        min_font = 12
        font_size = max(min_font, base_font - (self.env.K - 3) * 2)
        self.font_stats = pygame.font.SysFont("Arial", font_size)

        max_width = 120
        min_width = 60
        margin = 80
        available = self.width - 2 * margin

        self.door_width = max(min_width, min(max_width, available // (2 * self.env.K)))
        spacing = (available - self.door_width * self.env.K) // max(1, self.env.K - 1)
        self.door_height = 200

        start_x = margin
        self.doors = [
            pygame.Rect(start_x + i * (self.door_width + spacing),
                        self.height // 2 - self.door_height // 2,
                        self.door_width, self.door_height)
            for i in range(self.env.K)
        ]

    def reset(self, env, policy=None, code_mode=None, env_name=None):
        self.env = env
        if policy is not None:
            self.policy = policy
        if code_mode is not None:
            self.code_mode = code_mode
        if env_name is not None:
            self.env_name = env_name

        self.env.reset()
        self.total_hits = 0
        self.done = False
        self.step_count = 0

        self.hammer = HammerAnimation()
        self.frame_count = 0

        if self.code_mode:
            self.paused = True
            self.single_step = False

        self._rebuild_layout()

    def draw(self):
        self.screen.fill((30, 30, 30))

        hdr_mode = "Code" if self.code_mode else "Manual"
        status = "Paused" if (self.code_mode and self.paused) else ("Playing" if self.code_mode else "—")
        info_line = self.font_info.render(
            f"Env: {self.env_name} | Mode: {hdr_mode} | Status: {status} | Policy: StudentPolicy | H0={self.env.H0} | Doors={self.env.K}",
            True, (200, 200, 200)
        )
        self.screen.blit(info_line, (20, 10))

        if self.code_mode:
            hint = self.font_small.render("Controls: [Space]=Play/Pause  [N]=Step  [M]=Manual  [R]=Reset  [P]=Sample Env", True, (150, 150, 150))
            self.screen.blit(hint, (20, 32))

        text = self.font_header.render(f"Total Hits: {self.total_hits}", True, (255, 255, 255))
        self.screen.blit(text, (self.width // 2 - text.get_width() // 2, 56))

        for i, door in enumerate(self.doors):
            pct = max(0.0, min(1.0, self.env.health[i] / self.env.H0))
            visible_h = int(self.door_height * pct)
            visible_rect = pygame.Rect(door.x, door.y + (self.door_height - visible_h),
                                       self.door_width, visible_h)
            pygame.draw.rect(self.screen, (150, 0, 0), visible_rect)

            health_text = self.font_stats.render(f"{int(self.env.health[i])}", True, (255, 255, 255))
            self.screen.blit(health_text, (door.centerx - health_text.get_width() // 2, door.y - 30))

            hits = self.env.total_hits[i]
            avg_loss = self.env.total_loss[i] / hits if hits > 0 else 0
            hit_text = self.font_stats.render(f"Hits: {hits}", True, (200, 200, 200))
            loss_text = self.font_stats.render(f"Avg loss: {avg_loss:.2f}", True, (200, 200, 200))
            self.screen.blit(hit_text, (door.centerx - hit_text.get_width() // 2, door.bottom + 10))
            self.screen.blit(loss_text, (door.centerx - loss_text.get_width() // 2, door.bottom + 36))

        self.hammer.update_and_draw(self.screen)

        if self.done:
            overlay = pygame.Surface((self.width, self.height))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))
            go_text = self.font_header.render("GAME OVER - Press R to Reset", True, (255, 50, 50))
            self.screen.blit(go_text, (self.width // 2 - go_text.get_width() // 2,
                                       self.height // 2 - go_text.get_height() // 2))

        pygame.display.flip()

    def _do_step_on_arm(self, arm: int):
        reward, done, info = self.env.step(arm)
        self.total_hits += 1
        if self.policy:
            self.policy.update(arm, reward, info["health"])
        self.hammer.trigger(self.doors[arm])
        if done:
            self.done = True

    def _ensure_policy(self):
        if self.policy is None:
            self.policy = StudentPolicy(self.env.K)

    def run(self):
        running = True
        while running:
            self.frame_count += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        mus = getattr(self.env, "mus")
                        self.reset(PoissonDoorsEnv(mus, H0=self.env.H0),
                                   policy=None, env_name="Poisson")

                    elif event.key == pygame.K_p:
                        mus = [1, 2, 3, 1]
                        self.reset(PoissonDoorsEnv(mus, H0=100),
                                   policy=None, env_name="Poisson")

                    elif event.key == pygame.K_m:
                        self.code_mode = False
                        self.paused = False
                        self.single_step = False
                    elif event.key == pygame.K_c:
                        self.code_mode = True
                        self.paused = True
                        self.single_step = False

                    elif event.key == pygame.K_SPACE and self.code_mode:
                        self.paused = not self.paused
                        self.single_step = False
                    elif event.key == pygame.K_n and self.code_mode:
                        if self.paused:
                            self.single_step = True

                elif event.type == pygame.MOUSEBUTTONDOWN and not self.done and not self.code_mode:
                    pos = pygame.mouse.get_pos()
                    for i, door in enumerate(self.doors):
                        if door.collidepoint(pos):
                            self._do_step_on_arm(i)
                            break

            if self.code_mode and not self.done and not self.hammer.active:
                self._ensure_policy()
                if self.paused:
                    if self.single_step:
                        arm = self.policy.select_arm(self.step_count + 1)
                        self._do_step_on_arm(arm)
                        self.step_count += 1
                        self.single_step = False
                else:
                    if self.frame_count % self.code_step_interval == 0:
                        arm = self.policy.select_arm(self.step_count + 1)
                        self._do_step_on_arm(arm)
                        self.step_count += 1

            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

# =========================================================
# ========================= MAIN ==========================
# =========================================================

if __name__ == "__main__":
    rng = np.random.default_rng(1729)
    mus = [1.8, 1.4, 2.3, 2.2, 0.8]
    env = PoissonDoorsEnv(mus, H0=100, rng=rng)
    policy = None  # Starts in Manual Mode
    game = DungeonGame(env, policy=policy, code_mode=False, env_name="Poisson")
    game.run()
