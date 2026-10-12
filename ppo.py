"""Continuous-action PPO adapted from CleanRL (MIT; see CLEANRL_LICENSE).

Source: https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_continuous_action.py
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, strftime

import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions.normal import Normal

from racing import RacingConfig, RacingEnv, policy_state


@dataclass
class Args:
    circuit: str = "simple"
    lookahead_distances: tuple[float, ...] = (0, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150)
    critic_warmup_iterations: int = 0
    reset_critic: bool = False
    total_timesteps: int = 10_000_000_000
    num_envs: int = 8192
    num_steps: int = 128
    max_steps: int = 2400
    learning_rate: float = 3e-4
    update_epochs: int = 4
    num_minibatches: int = 32
    gamma: float = 0.9995
    gae_lambda: float = 0.95
    reward_scale: float = 1.0
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.01
    anneal_lr: bool = False
    norm_adv: bool = True
    clip_vloss: bool = True
    seed: int = 1
    device: str = "auto"
    init_checkpoint: str | None = None
    checkpoint_interval: int = 5
    track: bool = True
    wandb_project: str = "racing-rl"
    wandb_entity: str | None = None


def layer_init(layer, std=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0)
    return layer


class Logger:
    def __init__(self, args, config):
        self.run = None
        self.run_id = f"ppo-{args.circuit}-{args.seed}-{strftime('%Y%m%d-%H%M%S')}"
        self.batch_size = args.num_envs * args.num_steps
        self.max_speed = config.vehicle.speed_normalization
        self.episode_returns = np.zeros(args.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
        self.episode_speed_sums = np.zeros(args.num_envs, dtype=np.float32)
        self.best_lap_time = float("inf")
        if args.track:
            import wandb

            self.run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=self.run_id,
                config=asdict(args) | {
                    "lookahead_distances": config.lookahead_distances,
                    "hidden_sizes": [128, 128],
                    "vehicle": asdict(config.vehicle),
                    "timeout_bootstrapping": True,
                    "reward": {"type": "dense_signed_lap_progress", "progress_scale": 1},
                },
                mode="online",
            )
            self.run_id = self.run.id
        self.start_time = perf_counter()

    def iteration_start(self):
        self.finished_returns = []
        self.finished_progress = []
        self.finished_lengths = []
        self.finished_speeds = []
        self.finished_lap_times = []
        self.clipfracs = []

    def step_start(self, observations):
        self.episode_speed_sums += observations[:, -4]

    def step_end(self, rewards, done, info):
        self.episode_returns += rewards
        self.episode_lengths += 1
        lap = info["lap_events"] != 0
        self.finished_returns.extend(self.episode_returns[done].tolist())
        self.finished_progress.extend(info["final_progress"][done].tolist())
        self.finished_lengths.extend(self.episode_lengths[done].tolist())
        self.finished_speeds.extend((self.episode_speed_sums[done] * self.max_speed / self.episode_lengths[done]).tolist())
        self.finished_lap_times.extend(info["lap_times"][lap].tolist())
        self.episode_returns[done] = 0
        self.episode_lengths[done] = 0
        self.episode_speed_sums[done] = 0

    @torch.no_grad()
    def update(self, policy_loss, value_loss, entropy, approx_kl, ratio, clip_coef):
        self.policy_loss = policy_loss.detach()
        self.value_loss = value_loss.detach()
        self.entropy = entropy.detach()
        self.approx_kl = approx_kl.detach()
        self.clipfracs.append(((ratio - 1).abs() > clip_coef).float().mean().item())

    def iteration_end(self, iteration, learning_rate, critic_warmup):
        global_step = iteration * self.batch_size
        metrics = {
            "charts/learning_rate": learning_rate,
            "charts/critic_warmup": int(critic_warmup),
            "charts/SPS": int(global_step / (perf_counter() - self.start_time)),
            "losses/policy_loss": self.policy_loss.item(),
            "losses/value_loss": self.value_loss.item(),
            "losses/entropy": self.entropy.mean().item(),
            "losses/approx_kl": self.approx_kl.item(),
            "losses/clipfrac": float(np.mean(self.clipfracs)),
            "episodes/count": len(self.finished_returns),
            "episodes/lap_count": len(self.finished_lap_times),
        }
        if self.finished_returns:
            metrics["episodes/return_mean"] = float(np.mean(self.finished_returns))
            metrics["episodes/progress_mean"] = float(np.mean(self.finished_progress))
            metrics["episodes/length_mean"] = float(np.mean(self.finished_lengths))
            metrics["episodes/speed_mps_mean"] = float(np.mean(self.finished_speeds))
        if self.finished_lap_times:
            self.best_lap_time = min(self.best_lap_time, min(self.finished_lap_times))
            metrics["episodes/lap_time_s_mean"] = float(np.mean(self.finished_lap_times))
            metrics["episodes/lap_time_s_best"] = self.best_lap_time
        print(f"step={global_step} SPS={metrics['charts/SPS']} episodes={len(self.finished_returns)}", flush=True)
        if self.run:
            self.run.log(metrics, step=global_step)

    def close(self):
        if self.run:
            self.run.finish()


class Agent(nn.Module):
    def __init__(self, obs_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 1), 1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_size, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 2), 0.01),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, 2), np.log(0.3)))
        with torch.no_grad():
            self.actor_mean[-1].bias[1] = 0.5

    def get_value(self, obs):
        return self.critic(obs)

    def get_distribution(self, obs):
        mean = self.actor_mean(obs)
        logstd = self.actor_logstd.expand_as(mean)
        return Normal(mean, logstd.exp())

    def load_checkpoint(self, checkpoint, config, reset_critic=False):
        state = policy_state(checkpoint, config)
        if reset_critic:
            state = {key: value for key, value in state.items() if not key.startswith("critic.")}
            missing, unexpected = self.load_state_dict(state, strict=False)
            assert not unexpected and set(missing) == set(f"critic.{key}" for key in self.critic.state_dict())
        else:
            self.load_state_dict(state)

    def get_action_and_value(self, obs, action=None):
        dist = self.get_distribution(obs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(obs)


def train(args: Args):
    batch_size = args.num_envs * args.num_steps
    assert args.total_timesteps >= batch_size
    assert args.num_minibatches > 0 and batch_size % args.num_minibatches == 0
    assert args.update_epochs > 0
    assert args.reward_scale > 0
    assert args.checkpoint_interval > 0
    assert args.critic_warmup_iterations >= 0
    
    minibatch_size = batch_size // args.num_minibatches
    iterations = args.total_timesteps // batch_size

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device == "auto" else device)

    config = RacingConfig.from_track(args.circuit, lookahead_distances=args.lookahead_distances, samples_per_span=1, step_dt=0.05, max_steps=args.max_steps)
    
    env = RacingEnv(config, args.num_envs, args.seed)
    obs_size = env.obs.shape[1]

    agent = Agent(obs_size).to(device)
    
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        agent.load_checkpoint(checkpoint, config, reset_critic=args.reset_critic)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_size), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, 2), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    action_buffer = torch.from_numpy(env.actions)
    observation_buffer = torch.from_numpy(env.obs)
    reward_buffer = torch.from_numpy(env.rewards)
    next_obs = torch.from_numpy(env.reset()).to(device)
    next_done = torch.zeros(args.num_envs, device=device)
    logger = None

    try:
        logger = Logger(args, config)
        for iteration in range(1, iterations + 1):
            critic_warmup = iteration <= args.critic_warmup_iterations
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = args.learning_rate * (1 - (iteration - 1) / iterations)

            logger.iteration_start()
            for step in range(args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                action_buffer.copy_(action.cpu())
                logger.step_start(env.obs)

                _, _, _, _, info = env.step()

                rewards[step] = reward_buffer.to(device) * args.reward_scale
                timeouts = (env.truncated != 0) & (env.terminated == 0)

                if np.any(timeouts):
                    with torch.no_grad():
                        mask = torch.as_tensor(timeouts, dtype=torch.bool, device=device)
                        timeout_obs = torch.as_tensor(info["final_observations"][timeouts], device=device)
                        rewards[step, mask] += args.gamma * agent.get_value(timeout_obs).flatten()
                next_obs = observation_buffer.to(device)

                done = (env.terminated | env.truncated).astype(bool)
                next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

                logger.step_end(env.rewards, done, info)

            with torch.no_grad():
                next_value = agent.get_value(next_obs).flatten()
                advantages = torch.zeros_like(rewards)
                last_gae = torch.zeros(args.num_envs, device=device)
                for step in reversed(range(args.num_steps)):
                    next_nonterminal = 1 - (next_done if step == args.num_steps - 1 else dones[step + 1])
                    next_values = next_value if step == args.num_steps - 1 else values[step + 1]
                    delta = rewards[step] + args.gamma * next_values * next_nonterminal - values[step]
                    last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                    advantages[step] = last_gae
                returns = advantages + values

            b_obs = obs.reshape(-1, obs_size)
            b_actions = actions.reshape(-1, 2)
            b_logprobs = logprobs.flatten()
            b_advantages = advantages.flatten()
            b_returns = returns.flatten()
            b_values = values.flatten()

            for _ in range(args.update_epochs):
                indices = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    mb = indices[start:start + minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss = torch.max(
                        -mb_advantages * ratio,
                        -mb_advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()

                    newvalue = newvalue.flatten()
                    if args.clip_vloss:
                        unclipped = (newvalue - b_returns[mb]).square()
                        clipped_value = b_values[mb] + (newvalue - b_values[mb]).clamp(-args.clip_coef, args.clip_coef)
                        clipped = (clipped_value - b_returns[mb]).square()
                        value_loss = 0.5 * torch.max(unclipped, clipped).mean()
                    else:
                        value_loss = 0.5 * (newvalue - b_returns[mb]).square().mean()

                    loss = args.vf_coef * value_loss if critic_warmup else pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    logger.update(pg_loss, value_loss, entropy, approx_kl, ratio, args.clip_coef)

                if not critic_warmup and args.target_kl is not None and approx_kl > args.target_kl:
                    break

            global_step = iteration * batch_size
            logger.iteration_end(iteration, optimizer.param_groups[0]["lr"], critic_warmup)
            checkpoint_name = "policy.pt" if args.circuit == "red-bull-ring" else f"policy-{args.circuit}.pt"
            checkpoint = Path(__file__).parent / "build" / checkpoint_name
            payload = {
                "agent": agent.state_dict(),
                "vehicle": asdict(config.vehicle),
                "lookahead_distances": tuple(config.lookahead_distances),
                "global_step": global_step,
            }
            torch.save(payload, checkpoint.with_suffix(".tmp"))
            checkpoint.with_suffix(".tmp").replace(checkpoint)
            if iteration % args.checkpoint_interval == 0:
                torch.save(payload, checkpoint.with_name(f"{checkpoint.stem}-{logger.run_id}-{global_step}.pt"))
    finally:
        env.close()
        if logger is not None:
            logger.close()


def main():
    train(tyro.cli(Args))


if __name__ == "__main__":
    main()
