import os
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import wandb
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from hearthstone.engine.entities import HandCard, StoreItem, Unit
from hearthstone.env.hs_env import HearthstoneEnv


class SupportsPPOLoad(Protocol):
    @classmethod
    def load(cls, path: str, device: str = "auto") -> MaskablePPO: ...


class SupportsEnvMethod(Protocol):
    def env_method(
        self,
        method_name: str,
        *method_args: object,
        indices: int | list[int] | None = None,
        **method_kwargs: object,
    ) -> list[object]: ...


class WandbApi(Protocol):
    run: object | None

    def save(
        self,
        glob_str: str,
        base_path: str | None = None,
        policy: str = "live",
    ) -> bool | list[str]: ...


class GameLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1) -> None:
        super(GameLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        # Create clean env for tests
        self.eval_env = HearthstoneEnv()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._run_simulation()
        return True

    def _run_simulation(self) -> None:
        obs, _ = self.eval_env.reset()
        done = False
        truncated = False

        log_lines = [f"# Game Simulation at Step {self.num_timesteps}\n"]
        # clear init type model for action_masks
        model = cast(MaskablePPO, self.model)

        while not done and not truncated:
            # get masks
            masks = self.eval_env.action_masks()
            masks_arr = np.asarray(masks, dtype=bool)

            # call predict for MaskablePPO
            action, _ = model.predict(obs, action_masks=masks_arr, deterministic=True)

            # logs
            player = self.eval_env.game.players[self.eval_env.my_player_id]
            action_str = self._decode_action(int(action))

            log_lines.append(
                f"## Turn {self.eval_env.game.turn_count} | HP: {player.health} | Gold: {player.gold}"
            )
            log_lines.append(f"**Shop**: {self._format_shop(player.store)}")
            log_lines.append(f"**Board**: {self._format_board(player.board)}")
            log_lines.append(f"**Hand**: {self._format_hand(player.hand)}")
            log_lines.append(f"> **ACTION**: `{action_str}`")
            log_lines.append("---\n")

            obs, _, done, truncated, _ = self.eval_env.step(int(action))

        # Results
        p0 = self.eval_env.game.players[0]
        result = "WIN" if p0.health > 0 else "LOSS/DRAW"
        log_lines.append(f"# GAME OVER: {result}. Final HP: {p0.health}")

        # Save logs
        filename = f"game_log_{self.num_timesteps}.md"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        wandb_api = cast(WandbApi, wandb)
        if wandb_api.run is not None:
            wandb_api.save(path)

    def _format_board(self, board: Sequence[Unit]) -> str:
        return (
            " | ".join([f"{u.card_id}({u.cur_atk}/{u.cur_hp})" for u in board])
            if board
            else "Empty"
        )

    def _format_hand(self, hand: Sequence[HandCard]) -> str:
        return (
            ", ".join(
                [c.unit.card_id if c.unit else (c.spell.card_id if c.spell else "") for c in hand]
            )
            if hand
            else "Empty"
        )

    def _format_shop(self, shop: Sequence[StoreItem]) -> str:
        return (
            ", ".join(
                [c.unit.card_id if c.unit else (c.spell.card_id if c.spell else "") for c in shop]
            )
            if shop
            else "Empty"
        )

    def _decode_action(self, action: int, is_targeting: bool = False) -> str:
        if action == 0:
            return "END_TURN"
        if action == 1:
            return "ROLL"
        if 2 <= action <= 8:
            if is_targeting:
                return f"TARGET_BOARD {action - 2}"
            else:
                return f"BUY {action - 2}"
        if 9 <= action <= 15:
            return f"SELL {action - 9}"
        if 16 <= action <= 25:
            return f"PLAY {action - 16}"
        if 26 <= action <= 31:
            return f"SWAP {action - 26}"
        return f"UNKNOWN {action}"


class SelfPlayCallback(BaseCallback):
    """
    Every `update_freq` steps save current model and
    set it like opponent for every next model
    """

    def __init__(self, update_freq: int, model_save_path: str, verbose: int = 1) -> None:
        super(SelfPlayCallback, self).__init__(verbose)
        self.update_freq = update_freq
        self.model_save_path = model_save_path
        self.opponent_path = os.path.join(model_save_path, "opponent_temp.zip")

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            if self.verbose > 0:
                print(f"[SELF-PLAY] Updating opponent model at step {self.num_timesteps}")

            # 1. Save current agent
            self.model.save(self.opponent_path)

            # 2. Upload its copy (on CPU, for optimizing memory)
            # we use custom_objects for load in every py ver.
            ppo_loader = cast(SupportsPPOLoad, MaskablePPO)
            opponent = ppo_loader.load(self.opponent_path, device="cpu")

            # 3. Make "new brain" for all parallel envs
            # training_env - its usually DummyVecEnv, have method env_method
            vec_env = cast(SupportsEnvMethod, self.training_env)
            vec_env.env_method("set_opponent", opponent)

        return True


class BoardPowerCallback(BaseCallback):
    """
    Логирует среднюю силу стола агента в WandB.
    Сила стола = _calculate_board_power() из HearthstoneEnv.
    Чем лучше агент покупает и расставляет карты, тем выше метрика.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self._power_history: list[float] = []

    def _on_step(self) -> bool:
        # Собираем board_power из всех параллельных сред
        vec_env = cast(SupportsEnvMethod, self.training_env)
        try:
            powers = vec_env.env_method("get_board_power")
            for p in powers:
                if isinstance(p, (int, float)) and p > 0:
                    self._power_history.append(float(p))
        except Exception:
            pass

        if self.n_calls % self.log_freq == 0 and self._power_history:
            avg_power = np.mean(self._power_history)
            max_power = np.max(self._power_history)
            self.logger.record("custom/avg_board_power", avg_power)
            self.logger.record("custom/max_board_power", max_power)
            self._power_history.clear()

        return True


class CurriculumCallback(BaseCallback):
    """Phased training curriculum.

    Phase 1 (0 to ghost_start_step): 100% smart bot (fast warm-up,
        populates ghost pool with competent boards).
    Phase 2 (ghost_start_step+): Enable ghost self-play
        (80% ghost pool / 20% smart bot by default).

    If pool_preloaded=True, skip phase 1 and enable ghosts immediately.
    """

    def __init__(
        self,
        ghost_start_step: int = 400_000,
        pool_preloaded: bool = False,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.ghost_start_step = ghost_start_step
        self.pool_preloaded = pool_preloaded
        self._ghost_enabled = False
        self._first_step = True

    def _on_step(self) -> bool:
        # If pool was loaded from disk, enable ghost immediately
        if self._first_step and self.pool_preloaded:
            self._first_step = False
            self._ghost_enabled = True
            if self.verbose > 0:
                print("[CURRICULUM] Ghost pool pre-loaded, "
                      "enabling ghost mode from step 0")
            vec_env = cast(SupportsEnvMethod, self.training_env)
            vec_env.env_method("enable_ghost_mode")
            return True
        self._first_step = False

        if (
            not self._ghost_enabled
            and self.num_timesteps >= self.ghost_start_step
        ):
            self._ghost_enabled = True
            if self.verbose > 0:
                print(
                    f"[CURRICULUM] Enabling ghost self-play "
                    f"at step {self.num_timesteps}"
                )
            vec_env = cast(SupportsEnvMethod, self.training_env)
            vec_env.env_method("enable_ghost_mode")
        return True


class EntropyDecayCallback(BaseCallback):
    """Linearly decay ent_coef from initial value to final value."""

    def __init__(
        self,
        ent_coef_start: float = 0.04,
        ent_coef_end: float = 0.01,
        decay_fraction: float = 0.75,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.decay_fraction = decay_fraction

    def _on_step(self) -> bool:
        progress = 1.0 - self.model._current_progress_remaining  # 0→1
        if progress < self.decay_fraction:
            t = progress / self.decay_fraction  # 0→1 within decay window
            ent_coef = self.ent_coef_start + t * (self.ent_coef_end - self.ent_coef_start)
        else:
            ent_coef = self.ent_coef_end
        self.model.ent_coef = ent_coef
        return True
