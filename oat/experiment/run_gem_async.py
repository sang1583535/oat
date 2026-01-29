# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Entry script of using OAT to RL-tune LLM agents on GEM environments.
Uses async generation where each environment generates independently.
"""

import asyncio
import functools
import json
import logging
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
import vllm
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import Transition, TransitionData
from oat.utils.ops import masked_sum
from torch.utils.data import Dataset

import gem
from gem.utils.parsing import extract_last_boxed_answer
from gem.wrappers.wrapper_factory import get_wrapper_fns

""" +=========================================+ """
""" 1. Defining constants used in our training. """
""" +=========================================+ """

# Invalid action to be sent to the env to trigger format error penalty.
INVALID_ACTION = "<｜INVALID_ACTION｜>"


def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_no_template(observation: str) -> str:
    return observation


def apply_qwen3_general_template(question: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_code_template(question: str) -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {question}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )


TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
}


""" +=================================================+ """
""" 2. Defining extra arguments/structure for training. """
""" +=================================================+ """


@dataclass
class Args(PPOArgs):
    # Environment settings
    env_id: str = "rg:leg_counting"
    num_env: int = 1
    wrappers: str = ""
    async_env: bool = False

    # Algorithm settings
    length_norm_constant: Optional[int] = None

    # Template settings
    prompt_template: Literal["qwen3_game", "no", "qwen3_general", "code"] = "qwen3_game"

    # Reward settings
    gamma: float = 1.0  # Discount factor for Monte Carlo returns
    norm_return: bool = True

    # online evaluation settings
    eval_envs: str = None  # 'eval:AIME24|eval:MATH500'. See gem.envs
    eval_wrappers: str = ""
    eval_prompt_templates: str = "no"
    eval_async_env: bool = False
    eval_n: int = 1  # number of episodes to average for each env

    # Misc settings
    dump_experience_every: int = 1  # Dump experience data

    # Episode collection logic
    keep_generation_failed: bool = False  # Keep episodes with generation failures


""" +=======================================+ """
""" 3. Defining actor to collect experiences. """
""" +=======================================+ """


class Actor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.args.seed += 233 ** (actor_id + 1)
        self.game_state_save_path = os.path.join(self.save_path, "game_state")
        if actor_id == 0:
            os.makedirs(self.game_state_save_path, exist_ok=True)
        self.args: Args = self.args
        args = self.args
        self.oracle = None

        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            n=1,
            logprobs=True,
        )

        self.eval_sampling_params = vllm.SamplingParams(
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
            n=1,
            logprobs=True,
        )

        self.step_count = 0

        # Get environment wrappers.
        wrappers = get_wrapper_fns(self.args.wrappers, tokenizer=self.tokenizer)

        # Instantiate individual environments (not vectorized for async).
        self.envs = []
        self.env_locks = []
        for j in range(self.args.num_env):
            env = gem.make(self.args.env_id, seed=self.args.seed + j)
            for wrapper in wrappers:
                env = wrapper(env)
            self.envs.append(env)
            self.env_locks.append(asyncio.Lock())

        # Instantiate eval environments (individual envs for async)
        self.eval_envs = {}
        self.eval_env_locks = {}
        for i, eval_env_id in enumerate(self.args.eval_envs):
            wrappers = get_wrapper_fns(
                self.args.eval_wrappers[i], tokenizer=self.tokenizer
            )
            self.eval_envs[eval_env_id] = []
            self.eval_env_locks[eval_env_id] = []
            for j in range(self.args.eval_batch_size):
                env = gem.make(eval_env_id, seed=self.args.seed + j + 1000)
                for wrapper in wrappers:
                    env = wrapper(env)
                self.eval_envs[eval_env_id].append(env)
                self.eval_env_locks[eval_env_id].append(asyncio.Lock())

    def step(
        self, prompts=None, formatted_prompts=None, references=None
    ) -> List[TransitionData]:
        """Each actor.step handles the interaction between agent and environment to collect experiences."""
        # The provided parameters are ignored since we generate prompts from the environment
        del prompts, formatted_prompts, references

        info = {}

        # Play multiple episodes to generate transitions (trajectories in language MDP)
        all_trajectories = []

        finished_episodes, collection_info = self.collect_experience_async()
        for ep in finished_episodes:
            all_trajectories.extend(self.prepare_trajectories(ep))

        # logging infos
        info["actor/num_trajectories"] = len(all_trajectories)
        info["actor/mean_episode_len"] = np.mean([len(ep) for ep in finished_episodes])
        info["actor/mean_episode_return"] = np.mean(
            [
                sum(transition.rewards for transition in episode)
                for episode in finished_episodes
            ]
        )
        info["actor/mean_episode_success"] = np.mean(
            [episode[-1].rewards == 1 for episode in finished_episodes]
        )  # NOTE: assuming success reward is always 1

        # update collection info
        info.update(
            {k.replace("actor/", "actor/"): v for k, v in collection_info.items()}
        )

        # Subsample trajectories if they exceed the batch size
        if len(all_trajectories) > self.args.rollout_batch_size_per_device:
            subsample_indices = np.random.choice(
                len(all_trajectories),
                self.args.rollout_batch_size_per_device,
                replace=False,
            )
            all_trajectories = [all_trajectories[si] for si in subsample_indices]
        logging.info(f"Actor finished collecting {len(all_trajectories)} trajectories")

        for trajectory in all_trajectories:
            trajectory.info.update(**info)

        # Serialize and return the trajectories
        handle = self.ipc_client.serialize_ipc(all_trajectories)
        return handle  # type: ignore

    def collect_experience_async(self):
        """Collect experiences using async generation where each env runs independently."""
        logging.info(
            f"Actor-{self.actor_id} starting to collect experiences at step {self.step_count}"
        )
        min_steps = self.args.rollout_batch_size_per_device

        async def collect_all():
            finished_episodes = []
            finished_episodes_tool_uses = []
            finished_episodes_tool_success = []
            num_generation_failed = 0
            episodes_collected = asyncio.Event()

            # Track running episodes per env
            running_tasks = []

            async def run_single_episode(env_idx: int):
                """Run a single episode for one environment asynchronously."""
                nonlocal num_generation_failed
                episode = []

                async with self.env_locks[env_idx]:
                    obs, _ = self.envs[env_idx].reset()

                    while True:
                        # Generate action asynchronously for this single observation
                        action, extra = await self.agent_act_async(
                            obs, self.args.prompt_template
                        )

                        # Step the environment
                        next_obs, reward, terminated, truncated, info = self.envs[
                            env_idx
                        ].step(action)
                        done = terminated or truncated

                        if extra["generation_failed"]:
                            num_generation_failed += 1
                            if self.args.keep_generation_failed and len(episode) > 0:
                                episode[-1].rewards += reward
                                episode[-1].done = True
                                finished_episodes.append(deepcopy(episode))
                                finished_episodes_tool_uses.append(
                                    info.get("prev_ep_tool_use_counter", 0)
                                    if done
                                    else info.get("tool_use_counter", 0)
                                )
                                finished_episodes_tool_success.append(
                                    info.get("prev_ep_tool_success_counter", 0)
                                    if done
                                    else info.get("tool_success_counter", 0)
                                )
                            # Reset and continue if not done
                            if not done:
                                obs, _ = self.envs[env_idx].reset()
                                episode = []
                                continue
                            else:
                                break
                        else:
                            transition = Transition(
                                obs=obs,
                                action=action,
                                rewards=reward,
                                done=done,
                                prompt=extra["formatted_observation"],
                                prompt_ids=extra["prompt_ids"],
                                response=extra["response"],
                                response_ids=extra["response_ids"],
                                response_logprobs=extra["response_logprobs"],
                                response_is_truncated=extra["response_is_truncated"],
                                action_is_formatted=extra["action_is_formatted"],
                            )
                            episode.append(transition)

                            if done:
                                finished_episodes.append(deepcopy(episode))
                                finished_episodes_tool_uses.append(
                                    info.get("prev_ep_tool_use_counter", 0)
                                )
                                finished_episodes_tool_success.append(
                                    info.get("prev_ep_tool_success_counter", 0)
                                )
                                break

                            obs = next_obs

                # Check if we have enough transitions
                if len(tree.flatten(finished_episodes)) >= min_steps:
                    episodes_collected.set()

                return episode

            async def episode_runner(env_idx: int):
                """Keep running episodes for an env until we have enough."""
                while not episodes_collected.is_set():
                    await run_single_episode(env_idx)
                    # Small delay to allow checking the event
                    await asyncio.sleep(0)

            # Start all environment runners
            for env_idx in range(len(self.envs)):
                task = asyncio.create_task(episode_runner(env_idx))
                running_tasks.append(task)

            # Wait for enough episodes
            await episodes_collected.wait()

            # Cancel remaining tasks
            for task in running_tasks:
                task.cancel()

            # Wait for tasks to finish
            await asyncio.gather(*running_tasks, return_exceptions=True)

            return (
                finished_episodes,
                num_generation_failed,
                finished_episodes_tool_uses,
                finished_episodes_tool_success,
            )

        # Run the async collection using the actor's event loop
        (
            finished_episodes,
            num_generation_failed,
            finished_episodes_tool_uses,
            finished_episodes_tool_success,
        ) = self._async_loop.run_coroutine(collect_all())

        info = {
            "actor/num_generation_failed": num_generation_failed,
            "actor/prop_generation_failed": (
                num_generation_failed / len(finished_episodes)
                if self.args.keep_generation_failed
                else num_generation_failed
                / (len(finished_episodes) + num_generation_failed)
            )
            if (len(finished_episodes) + num_generation_failed) > 0
            else 0,
            "actor/num_tool_uses": np.mean(finished_episodes_tool_uses)
            if finished_episodes_tool_uses
            else 0,
            "actor/num_tool_success": np.mean(finished_episodes_tool_success)
            if finished_episodes_tool_success
            else 0,
        }

        if self.step_count % self.args.dump_experience_every == 0:
            _to_dump = {}
            for i, ep in enumerate(finished_episodes):
                key = f"episode{i}"
                _to_dump[key] = []
                for transition in ep:
                    _to_dump[key].append(transition.format())
            with open(
                os.path.join(
                    self.game_state_save_path,
                    f"actor{self.actor_id}_step{self.step_count}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    _to_dump,
                    f,
                    indent=4,
                )
        self.step_count += 1
        return finished_episodes, info

    async def agent_act_async(
        self,
        observation: str,
        prompt_template: str,
    ) -> Tuple[str, dict]:
        """Use the current LLM as a policy to act asynchronously for a single observation.

        Args:
            observation: Observation from the environment.
            prompt_template: Template to apply.

        Returns:
            Tuple[str, dict]: Action and extra data.
        """
        formatted_observation = TEMPLATE_FACTORY[prompt_template](observation)
        if self.args.apply_chat_template:
            formatted_observation = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": formatted_observation}],
                tokenize=False,
                add_generation_prompt=True,
            )

        sampling_params = (
            self.eval_sampling_params if self.eval_mode else self.sampling_params
        )

        # Check if observation exceeds max model length
        ids = self.tokenizer(formatted_observation).input_ids
        if len(ids) >= self.args.max_model_len:
            return INVALID_ACTION, {"generation_failed": True}

        # Generate asynchronously for this single observation
        output = await self.generate_async(formatted_observation, sampling_params)

        raw_action = output.outputs[0].text
        prompt_token_ids = output.prompt_token_ids
        token_ids = output.outputs[0].token_ids
        response_logprobs = output.outputs[0].logprobs
        response_logprobs = [
            item[token_ids[i]].logprob for i, item in enumerate(response_logprobs)
        ]
        response_is_truncated = output.outputs[0].finish_reason == "length"

        # Valid extraction = proper eos + proper format
        extracted_action = (
            INVALID_ACTION
            if response_is_truncated
            else self.extract_action(raw_action)
        )
        executable_action = INVALID_ACTION if response_is_truncated else raw_action

        extra = {
            "formatted_observation": formatted_observation,
            "prompt_ids": prompt_token_ids,
            "response": raw_action,
            "response_ids": token_ids,
            "response_logprobs": response_logprobs,
            "response_is_truncated": response_is_truncated,
            "action_is_formatted": extracted_action != INVALID_ACTION,
            "generation_failed": False,
            "generation_max_length_reached": (
                len(prompt_token_ids) + len(token_ids) >= self.args.max_model_len
            ),
        }

        return executable_action, extra

    def prepare_trajectories(
        self, episode: Sequence[Transition]
    ) -> List[TransitionData]:
        """
        Prepare language trajectories (transitions of episode).

        Args:
            episode: A complete episode of the agent environment interaction.

        Returns:
            List of trajectory data
        """
        trajectory_data = []
        rewards = [t.rewards for t in episode]

        # Compute returns
        returns = np.zeros_like(rewards, dtype=np.float32)
        cur = 0.0
        for i in reversed(range(len(rewards))):
            cur = rewards[i] + self.args.gamma * cur
            returns[i] = cur

        # Distribute turn-based returns to token-level returns
        for i, step_data in enumerate(episode):
            dense_rewards = self.compute_token_level_rewards(
                step_data.response_ids, returns[i]
            )
            # Add trajectory data
            trajectory_data.append(
                TransitionData(
                    prompt=step_data.prompt,
                    prompt_ids=step_data.prompt_ids,
                    response=step_data.response,
                    response_ids=step_data.response_ids,
                    response_logprobs=step_data.response_logprobs,
                    rewards=dense_rewards,
                    loss_mask=(
                        not step_data.response_is_truncated
                        if self.args.ignore_no_eos
                        else True
                    ),
                    info={
                        "actor/action_is_formatted": step_data.action_is_formatted,
                        "actor/step_reward": rewards[i],
                        "actor/discount_factor": self.args.gamma,
                        "actor/discounted_step_return": returns[i],
                        "actor/response_is_truncated": step_data.response_is_truncated,
                        "actor/timestamp": time.time_ns(),
                    },
                )
            )

        return trajectory_data

    def compute_token_level_rewards(
        self, token_ids: List[int], discounted_reward: float
    ) -> List[float]:
        # Initialize all tokens with zero reward
        dense_rewards = [0.0] * len(token_ids)
        # Last token gets full discounted reward
        dense_rewards[-1] = discounted_reward
        return dense_rewards

    def extract_action(self, text: str) -> str:
        """
        Extract and format the actual action from the model's output.

        This method handles different template formats and ensures the action
        is properly formatted for the environment.

        Args:
            text: Raw text output from the model

        Returns:
            Cleaned and formatted action string ready for the environment
        """
        if not text:
            return ""  # Handle empty text case

        try:
            formatted_action = None
            if self.args.prompt_template in ["qwen3_game", "qwen3_general"] or (
                self.args.prompt_template == "no"
                and "qwen" in self.args.pretrain.lower()
            ):
                formatted_action = extract_last_boxed_answer(text)
                if formatted_action is None:
                    formatted_action = text.strip()
            elif self.args.prompt_template == "code":
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
                if not code_blocks:
                    formatted_action = None
                else:
                    formatted_action = code_blocks[-1].strip()
            else:
                raise NotImplementedError

            if formatted_action is None:
                formatted_action = INVALID_ACTION

            return formatted_action

        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            # Return invalid action if extraction fails.
            return INVALID_ACTION

    def run_eval_episode(
        self, eval_env_id, eval_prompt_template, batch_size
    ) -> List[Transition]:
        """Run evaluation episodes using async generation."""

        def get_attr_from_wrapper(env, attr):
            if hasattr(env, attr):
                return getattr(env, attr)

            if hasattr(env, "env"):
                return get_attr_from_wrapper(env.env, attr)

            raise ValueError(f"Cannot find {attr} in env.")

        envs = self.eval_envs[eval_env_id]
        env_locks = self.eval_env_locks[eval_env_id]

        try:
            dataset = get_attr_from_wrapper(envs[0], "dataset")
        except ValueError:
            dataset = DummyPromptDataset(size=1)

        num_envs = len(envs)
        if batch_size > len(dataset):
            logging.info(
                f"eval batch size {batch_size} is larger than dataset size {len(dataset)}, set batch size to {len(dataset)}"
            )
            batch_size = len(dataset)

        async def run_single_eval_episode(env_idx: int, data_idx: int):
            """Run a single evaluation episode asynchronously."""
            episode = []

            async with env_locks[env_idx]:
                # Reset with specific index if dataset exists
                if hasattr(envs[env_idx], "reset"):
                    try:
                        obs, info = envs[env_idx].reset(idx=data_idx)
                    except TypeError:
                        obs, info = envs[env_idx].reset()
                else:
                    obs, info = envs[env_idx].reset()

                while True:
                    # Generate action asynchronously
                    action, extra = await self.agent_act_async(
                        obs, eval_prompt_template
                    )

                    if extra["generation_failed"]:
                        # Treat generation failure as episode end
                        episode.append(
                            {
                                "obs": obs,
                                "action": INVALID_ACTION,
                                "reward": 0.0,
                                "next_obs": "",
                                "done": True,
                                "info": {"generation_failed": True},
                            }
                        )
                        break

                    # Step the environment
                    next_obs, reward, terminated, truncated, info = envs[env_idx].step(
                        action
                    )
                    done = terminated or truncated

                    # Check if observation exceeds max length
                    obs_len = len(self.tokenizer.encode(next_obs))
                    obs_exceeds_max_len = obs_len >= self.args.max_model_len

                    done = done or obs_exceeds_max_len

                    episode.append(
                        {
                            "obs": obs,
                            "action": action,
                            "reward": reward,
                            "next_obs": next_obs,
                            "done": done,
                            "info": info,
                        }
                    )

                    if done:
                        break

                    obs = next_obs

            return episode

        async def run_all_eval_episodes():
            """Run all evaluation episodes with environment pooling."""
            finished_episodes = []
            pending_data_indices = list(range(len(dataset)))
            env_available = [True] * num_envs

            async def process_batch():
                tasks = []
                assignments = []  # (env_idx, data_idx)

                # Assign available envs to pending data indices
                for env_idx in range(num_envs):
                    if env_available[env_idx] and pending_data_indices:
                        data_idx = pending_data_indices.pop(0)
                        env_available[env_idx] = False
                        assignments.append((env_idx, data_idx))
                        tasks.append(run_single_eval_episode(env_idx, data_idx))

                if not tasks:
                    return []

                # Run all assigned episodes concurrently
                results = await asyncio.gather(*tasks)

                # Mark envs as available again
                for env_idx, _ in assignments:
                    env_available[env_idx] = True

                return results

            # Process all data indices
            while pending_data_indices or not all(env_available):
                batch_results = await process_batch()
                finished_episodes.extend(batch_results)

                if not pending_data_indices and all(env_available):
                    break

            return finished_episodes

        # Run async evaluation using actor's event loop
        finished_episodes = self._async_loop.run_coroutine(run_all_eval_episodes())
        return finished_episodes


class DummyPromptDataset(Dataset):
    """Empty dataset to satisfy OAT's requirements without actually loading data."""

    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        del idx
        return "", "", ""


""" +====================================+ """
""" 4. Defining learner update the policy. """
""" +====================================+ """


class Learner(PPOLearner):
    def _init(self, args: Args, actors: List[Actor]) -> None:
        """
        Initialize the learner.
        """
        # Call parent's _init but then override prepare_data
        super()._init(args, actors)
        self.args = args

        # Masked sum is the correct implementation!
        # Oat by default uses Dr.GRPO: https://arxiv.org/pdf/2503.20783
        self.masked_aggregator = functools.partial(
            masked_sum,
            constant_normalizer=args.length_norm_constant or args.generate_max_length,
        )

    def prepare_data(self, strategy, tokenizer):
        """
        Override the data preparation to avoid loading external datasets.
        Instead, create dummy datasets just to keep OAT's infrastructure happy.
        """
        # Create dummy dataset that satisfies OAT's requirements
        # but doesn't actually load any data
        # Used to control the training episode, set a large number.
        self.prompts_dataset = DummyPromptDataset(size=int(1e9))
        self.eval_prompts_dataset = DummyPromptDataset(size=1)  # no use currently

        # Create the dataloaders
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            shuffle=False,  # No need to shuffle dummy data
        )
        self.eval_prompts_dataloader = strategy.setup_dataloader(
            self.eval_prompts_dataset, batch_size=1, shuffle=False, drop_last=False
        )

    def process_feedback_data(self, data_list: List[TransitionData]):
        """Process collected feedback data, adding it to buffer."""

        logging.info("adding data into buffer")

        # Add to buffer
        self.pi_buffer.extend(data_list)

        # Also add to all_buffer if we're tracking all data
        if self.args.dump_all_buffer:
            self.all_buffer.extend(data_list)

        # Update query step (for tracking progress)
        self.query_step += len(data_list)

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        # Return without baseline
        advantages = rewards.sum(-1)
        if self.args.norm_return:
            local_sum = advantages.sum()
            local_square_sum = (advantages**2).sum()
            local_num = torch.tensor(
                [advantages.numel()], dtype=torch.float32, device=advantages.device
            )

            global_sum = self.strategy.all_reduce(local_sum, op="sum")
            global_square_sum = self.strategy.all_reduce(local_square_sum, op="sum")
            global_num = self.strategy.all_reduce(local_num, op="sum")

            mean_adv = global_sum / global_num
            std_adv = torch.sqrt(global_square_sum / global_num - mean_adv**2)
            advantages = (advantages - mean_adv) / (std_adv + 1e-9)
        return advantages

    def evaluate(self, _unused_dataloader, steps):
        """Online evaluation on TIR environments."""
        # NOTE: Evaluate all envs specified in args.eval_envs, report avg@args.eval_n
        # NOTE: prompt_template is needed for concat wrapper
        del _unused_dataloader
        assert not self.pi_beta_lags_behind, "pi beta lags behind for evaluation"
        assert (
            self.args.eval_n % len(self.actors) == 0
        ), "args.eval_n must be divisible by number of actors"
        self._pre_evaluate()
        self.strategy.print(f"Starting evaluation at {steps} steps")
        eval_env_ids = self.args.eval_envs
        eval_prompt_templates = self.args.eval_prompt_templates

        t0 = time.time()
        futs = []
        episodes = []

        metrics = {
            f"eval/{env_id}/{metric}": 0.0
            for env_id in eval_env_ids
            for metric in [
                "accuracy",
                "elapse",
                "response_tok_len",
                "mean_episode_len",
                "num_tool_success",
            ]
        }

        for eval_env_id, eval_prompt_template in zip(
            eval_env_ids, eval_prompt_templates
        ):
            episodes.clear()
            # assign task and wait for results
            n_actor = len(self.actors)
            for _ in range(self.args.eval_n // n_actor):
                if self.strategy.is_rank_0():
                    futs += [
                        actor.futures.run_eval_episode(
                            eval_env_id,
                            eval_prompt_template,
                            self.args.eval_batch_size,
                        )
                        for actor in self.actors
                    ]
                    for fut in futs:
                        episodes.extend(fut.result())
                    futs.clear()

            run_elapse = time.time() - t0
            t0 = time.time()
            metrics.update(
                {
                    f"eval/{eval_env_id}/elapse": run_elapse,
                    f"eval/{eval_env_id}/response_tok_len": np.mean(
                        [
                            sum([len(self.tokenizer.encode(t["action"])) for t in ep])
                            for ep in episodes
                        ]
                    ),
                    f"eval/{eval_env_id}/accuracy": np.mean(
                        [sum([t["reward"] for t in ep]) for ep in episodes]
                    ),
                    f"eval/{eval_env_id}/mean_episode_len": np.mean(
                        [len(ep) for ep in episodes]
                    ),
                    f"eval/{eval_env_id}/num_tool_success": np.mean(
                        [
                            ep[-1]["info"].get("tool_success_counter", 0)
                            + ep[-1]["info"].get("prev_ep_tool_success_counter", 0)
                            for ep in episodes
                        ]
                    ),
                }
            )
            # save the results
            transitions = [t for ep in episodes for t in ep]
            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            pd.DataFrame(
                {
                    "obs": [t["obs"] for t in transitions],
                    "action": [t["action"] for t in transitions],
                    "reward": [t["reward"] for t in transitions],
                    "done": [t["done"] for t in transitions],
                    "next_obs": [t["next_obs"] for t in transitions],
                    "info": [t["info"] for t in transitions],
                }
            ).to_json(
                os.path.join(eval_res_path, f"{steps}_{eval_env_id}.json"),
                orient="records",
                indent=4,
            )

        dist.barrier()
        metrics = self.strategy.broadcast(metrics)
        metrics["eval/average/accuracy"] = np.mean(
            [metrics[f"eval/{env_id}/accuracy"] for env_id in eval_env_ids]
        )
        metrics["eval/average/mean_episode_len"] = np.mean(
            [metrics[f"eval/{env_id}/mean_episode_len"] for env_id in eval_env_ids]
        )
        metrics["eval/average/response_tok_len"] = np.mean(
            [metrics[f"eval/{env_id}/response_tok_len"] for env_id in eval_env_ids]
        )
        metrics["eval/average/elapse"] = np.mean(
            [metrics[f"eval/{env_id}/elapse"] for env_id in eval_env_ids]
        )
        self._post_evaluate()
        return metrics


def train(args: Args):
    """
    Reinforcement learning starts here.

    Args:
        args: Configuration arguments for the run
    """
    # Define a distributed program that composes Actors and Learners
    program, local_resources = get_program(args, learner_cls=Learner, actor_cls=Actor)

    # Launch the program
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    # Get default arguments and customize them
    args: Args = get_default_args(Args)

    # Customization
    args.algo = "PPO"

    # CRITICAL: Disable oracle and dataset loading
    args.oracle = ""  # Empty string for no external oracle
    args.prompt_data = ""  # Don't load any dataset
    args.rollout_batch_size = args.rollout_batch_size_per_device * args.gpus

    # setup evaluation hps
    def _validate_eval_hp(hp):
        hp = hp.split("|")
        if len(hp) == 1:
            hp = hp * len(args.eval_envs)
        else:
            assert len(hp) == len(
                args.eval_envs
            ), "eval_wrappers/eval_prompt_templates should be either a string or a list of the same length as eval_envs"
        return hp

    if args.eval_envs:
        args.eval_envs = args.eval_envs.split("|")
        assert isinstance(args.eval_envs, list)
        assert len(args.eval_envs) == len(
            set(args.eval_envs)
        ), "eval_envs should be unique"
        args.eval_wrappers = _validate_eval_hp(args.eval_wrappers)
        args.eval_prompt_templates = _validate_eval_hp(args.eval_prompt_templates)
    else:
        logging.info(
            "No eval_envs specified, set `args.eval_steps` to -1,skipping evaluation."
        )
        args.eval_envs = []
        args.eval_steps = -1

    if "concat_chat" in args.wrappers:
        assert (
            args.prompt_template == "no"
        ), "chat template is applied on env side already"
    args = default_args_validation(args)

    # Let's go
    train(args)
