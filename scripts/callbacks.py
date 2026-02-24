import os
from typing import Protocol, cast

import numpy as np
import wandb
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from src.hearthstone.engine.entities import HandCard, StoreItem, Unit
from src.hearthstone.env.hs_env import HearthstoneEnv


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

    def _format_board(self, board: list[Unit]) -> str:
        return (
            " | ".join([f"{u.card_id}({u.cur_atk}/{u.cur_hp})" for u in board])
            if board
            else "Empty"
        )

    def _format_hand(self, hand: list[HandCard]) -> str:
        return (
            ", ".join(
                [c.unit.card_id if c.unit else (c.spell.card_id if c.spell else "") for c in hand]
            )
            if hand
            else "Empty"
        )

    def _format_shop(self, shop: list[StoreItem]) -> str:
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
                print(f"🔄 Self-Play: Updating opponent model at step {self.num_timesteps}")

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
