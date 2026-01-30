# Architecture: Hierarchical Battle Knowledge Distillation

## 1. Concept
A three-stage RL framework designed to solve the sparse reward problem in auto-battlers. By distilling the complex "physics" of a battle into a lightweight predictor, the agent gains "foresight"—the ability to mentally simulate combat outcomes and assess risks without running heavy simulations during inference.

---

## 2. Core Components

### I. Battle Analyzer (The Teacher / Offline)
* **Role:** Ground Truth Generator.
* **Input:** Full simulation logs (sequences of 3k-10k tokens) capturing every attack, trigger, and death event.
* **Architecture:** Large Transformer Encoder.
* **Function:** Compresses the entire timeline of a battle into a dense **Battle Embedding**. It "understands" *why* a fight was won or lost.
* **Training:** Trained offline on millions of generated battle logs.

### II. Battle Predictor (The Student / World Model)
* **Role:** Real-time Intuition Provider.
* **Input:** Static Board State (before combat starts).
* **Architecture:** Lightweight Transformer (Fast Inference).
* **Function:**
    * **Latent Distillation:** Predicts the Teacher's *Battle Embedding* based solely on the starting board.
    * **High-Volume Training:** Since the input is light, it can be trained for many epochs to master the "physics" of the game.
* **Output:** Provides the Agent with immediate feedback on board strength, acting as a differentiable proxy for the combat phase.

### III. Strategist (The Agent / Policy)
* **Role:** Long-Term Planner.
* **Input:** Economy state, Shop slots, Hand, and the **Predictor's output vectors**.
* **Architecture:** MARL-GPT (Multi-Agent Transformer).
* **Function:**
    * **Decision Making:** Plans high-level actions (Buy/Sell/Level) using the Predictor's foresight.
    * **Pattern Recognition:** Identifies long-term dependencies (e.g., "Buying this weak unit now enables a triple later") inspired by Decision Transformer architectures.

---

## 3. Micro-Optimizations & Advanced Features

### A. Spatial Encoding (Danger Zones / AlphaStar Logic)
* **Concept:** Explicitly modeling position-based threats (Cleave/AoE damage).
* **Implementation:** A 1D-CNN or spatial attention layer that creates a "mini-map" of the board.
* **Function:**
    * Calculates "Danger Coordinates" (e.g., next to a Taunt unit).
    * Injects a "Danger Flag" into the state of vulnerable units.
    * Helps the agent learn positioning rules (e.g., "Don't put your Carry next to the Tank") without needing millions of failed trials.

### B. Distributional Value Heads
* **Concept:** Predicting risk profiles instead of scalar averages.
* **Implementation:** The model outputs a probability distribution (histogram) of possible damage taken/dealt.
* **Benefit:** Allows the agent to distinguish between "Safe Wins" (low variance) and "Coin Flips" (high variance), enabling safer play when HP is low.

### C. Unit-Specific Attention Heads (MVP Detection)
* **Concept:** Granular credit assignment for every unit on the board.
* **Implementation:** Multi-head attention where specific heads track individual unit metrics (Survival Time, Damage Dealt, Value Generated).
* **Benefit:**
    * Identifies **MVP Units** (high impact) -> "Protect this unit".
    * Identifies **Weak Links** (low impact) -> "Sell this unit".

### D. Global Game State Transformer
* **Concept:** Static evaluation of the macro-state.
* **Implementation:** A dedicated transformer block that evaluates the player's overall position (Health + Economy + Tier) *independently* of the specific combat permutations.
* **Benefit:** Provides a stable baseline value ("How good is my situation in general?") to anchor the variance of combat predictions.