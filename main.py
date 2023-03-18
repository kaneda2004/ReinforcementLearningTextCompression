import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs:")
import os
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv as OriginalDummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import heapq
import torch

# Check if a GPU is available and set the device to use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"the device we're using is {device}")

class LossCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            loss = np.mean([info["loss"] for info in self.locals["infos"]])
            print(f"Step {self.num_timesteps}: Loss: {loss}")
        return True


class EfficiencyCallback(BaseCallback):
    def __init__(self, env, check_freq: int, verbose=0):
        super(EfficiencyCallback, self).__init__(verbose)
        self.env = env
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            efficiency = self.env.get_attr("total_compressed_length")[0] / self.env.get_attr("original_length")[0]
            print(f"Step {self.num_timesteps}: Efficiency: {efficiency}")
        return True


class SaveModelCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(os.path.join(self.save_path, f"model_step_{self.num_timesteps}"))
            if self.verbose > 0:
                print(f"Saving model checkpoint at step {self.num_timesteps}")
        return True

class DummyVecEnv(OriginalDummyVecEnv):
    def step_wait(self):
        for env_idx in range(self.num_envs):
            action = self.actions[env_idx] if isinstance(self.actions,
                                                         np.ndarray) and self.actions.ndim > 0 else self.actions
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                action)
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs = self.envs[env_idx].reset()
            self.buf_obs[None][env_idx] = obs
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy())

class ScalarActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return action[0] if np.ndim(action) > 0 else action

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Step {self.num_timesteps}: Training progress update")
        return True

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_freq_table(text):
    freq_table = {}
    for char in text:
        freq_table[char] = freq_table.get(char, 0) + 1
    return freq_table

def build_huffman_tree(freq_table):
    priority_queue = [HuffmanNode(char, freq) for char, freq in freq_table.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left_node = heapq.heappop(priority_queue)
        right_node = heapq.heappop(priority_queue)

        parent_node = HuffmanNode(None, left_node.freq + right_node.freq)
        parent_node.left = left_node
        parent_node.right = right_node

        heapq.heappush(priority_queue, parent_node)

    return priority_queue[0]

def build_encoding_dict_helper(node, current_code, encoding_dict):
    if node is None:
        return

    if node.char is not None:
        encoding_dict[node.char] = current_code

    build_encoding_dict_helper(node.left, current_code + '0', encoding_dict)
    build_encoding_dict_helper(node.right, current_code + '1', encoding_dict)

def build_encoding_dict(huffman_tree):
    encoding_dict = {}
    build_encoding_dict_helper(huffman_tree, '', encoding_dict)
    return encoding_dict

def compress(text, encoding_dict):
    compressed_text = ''.join(encoding_dict[char] for char in text)
    return compressed_text



def huffman_strategy(text):
    # Use the functions from the previous examples to perform Huffman coding
    freq_table = build_freq_table(text)
    huffman_tree = build_huffman_tree(freq_table)
    encoding_dict = build_encoding_dict(huffman_tree)
    compressed_text = compress(text, encoding_dict)
    return compressed_text

def dictionary_strategy(text):
    # Use a basic dictionary-based encoding
    dictionary = {' ': '00', 'e': '01', 't': '10', 'a': '110', 'o': '1110', 'n': '1111'}
    compressed_text = ''.join(dictionary.get(char, '000') for char in text)
    return compressed_text


def build_freq_table(text):
    freq_table = {}
    for char in text:
        freq_table[char] = freq_table.get(char, 0) + 1
    return freq_table


def build_efficient_dictionary(text):
    freq_table = build_freq_table(text)
    sorted_chars = sorted(freq_table.items(), key=lambda x: x[1], reverse=True)

    encoding_dict = {}
    code_length = 1
    code = 0

    for char, _ in sorted_chars:
        encoding_dict[char] = format(code, f"0{code_length}b")
        code += 1
        if code == 2 ** code_length:
            code_length += 1
            code = 0

    return encoding_dict


def efficient_dictionary_strategy(text, encoding_dict):
    compressed_text = ''.join(encoding_dict[char] for char in text)
    return compressed_text

def run_length_encoding(text):
    if not text:
        return ""

    compressed_text = []
    count = 1
    prev_char = text[0]

    for char in text[1:]:
        if char == prev_char:
            count += 1
        else:
            compressed_text.append(prev_char)
            compressed_text.append(str(count))
            count = 1
            prev_char = char

    compressed_text.append(prev_char)
    compressed_text.append(str(count))

    return "".join(compressed_text)

import lzma

def lzma_compression(text):
    compressed_data = lzma.compress(text.encode())
    compressed_text = "".join([chr(byte) for byte in compressed_data])
    return compressed_text


# Burrows-Wheeler Transform (BWT) implementation
def bw_transform(text):
    rotations = sorted([text[i:] + text[:i] for i in range(len(text))])
    bwt_text = ''.join([rotation[-1] for rotation in rotations])
    return bwt_text

# Move-to-Front (MTF) encoding implementation
def mtf_encode(text):
    alphabet = list(set(text))
    mtf_text = []
    for char in text:
        index = alphabet.index(char)
        mtf_text.append(index)
        alphabet.pop(index)
        alphabet.insert(0, char)
    return ''.join(chr(index + 128) for index in mtf_text)


def bwt_rle_mtf_strategy(text):
    bwt_text = bw_transform(text)
    rle_text = run_length_encoding(bwt_text)
    mtf_text = mtf_encode(rle_text)
    compressed_text = ''.join([format(ord(char), '08b') for char in mtf_text])
    return compressed_text



from arithmetic_coding import arithmetic_compression




import zlib

def zlib_deflate_strategy(text):
    compressed_data = zlib.compress(text.encode())
    compressed_text = ''.join([chr(byte) for byte in compressed_data])
    return compressed_text


class TextCompressionEnv(gym.Env):
    def __init__(self, text, max_context=512, max_actions=8):
        super().__init__()
        self.step_counter = 0
        self.print_freq = 10  # Set the desired frequency for print statements
        self.total_compressed_length = 0
        self.original_length = 0
        self.min_compressed_length = float('inf')  # Add this line to initialize the variable
        self.text = text
        self.max_context = max_context
        self.max_actions = max_actions
        # Build the efficient dictionary
        self.efficient_dict = build_efficient_dictionary(text)

        self.action_space = spaces.Discrete(self.max_actions)  # Update self.max_actions to include the new strategies

        self.observation_space = spaces.MultiDiscrete([256] * self.max_context)

        self.current_pos = 0
        self.context = [0] * self.max_context

    def reset(self):
        self.current_pos = 0
        self.context = [ord(char) for char in self.text[:self.max_context]]
        return np.array(self.context)

    def step(self, action):
        # Apply the action (compression strategy) and get the compressed length
        compressed_length = self.compress(self.context, action)

        self.total_compressed_length += compressed_length
        self.original_length += len(self.context)

        # Calculate the reward based on the compression efficiency
        reward = -compressed_length

        # Calculate the loss
        loss = compressed_length - self.min_compressed_length

        # Update the context and current position
        if self.current_pos < len(self.text):
            self.context.pop(0)
            self.context.append(ord(self.text[self.current_pos]))

        done = self.current_pos >= len(self.text)
        info = {}

        # Print the compression efficiency and compressed text size
        self.step_counter += 1
        if self.step_counter % self.print_freq == 0:
            efficiency = self.total_compressed_length / self.original_length
            self.min_compressed_length = min(self.min_compressed_length, self.total_compressed_length)  # Update the minimum compressed length
            print(
                f"Step {self.current_pos}: Action: {action}, Efficiency: {efficiency}, Compressed Text Size: {self.total_compressed_length} bits, Lowest Compressed Text Size: {self.min_compressed_length} bits")

        self.current_pos += 1

        return np.array(self.context), reward, done, {"loss": loss}


    def render(self, mode='human'):
        pass

    def compress(self, context, action):
        # Convert the context from integer indices back to characters
        context_text = ''.join(chr(index) for index in context)

        # Apply the selected compression strategy based on the action value
        if action == 0:
            compressed_text = huffman_strategy(context_text)
        elif action == 1:
            compressed_text = dictionary_strategy(context_text)
        elif action == 2:
            compressed_text = efficient_dictionary_strategy(context_text, self.efficient_dict)
        elif action == 3:
            compressed_text = run_length_encoding(context_text)
        elif action == 4:
            compressed_text = lzma_compression(context_text)
        elif action == 5:
            compressed_text = bwt_rle_mtf_strategy(context_text)
        elif action == 6:
            compressed_text = arithmetic_compression(context_text)
        elif action == 7:
            compressed_text = zlib_deflate_strategy(context_text)
        else:
            compressed_text = context_text
        # Return the length of the compressed text
        return len(compressed_text)




def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def find_latest_checkpoint(save_path):
    max_step = -1
    latest_checkpoint = None
    for file in os.listdir(save_path):
        if file.endswith(".zip"):
            step = int(file.split("_")[-1].split(".")[0])  # Extract the step number from the file name
            if step > max_step:
                max_step = step
                latest_checkpoint = file
    return os.path.join(save_path, latest_checkpoint) if latest_checkpoint is not None else None

def get_timesteps_from_checkpoint(checkpoint_path):
    if checkpoint_path is None:
        return 0
    filename = os.path.basename(checkpoint_path)
    step = int(filename.split("_")[-1].split(".")[0])
    return step

text = load_text('plrabn12.txt')
env = DummyVecEnv([lambda: ScalarActionWrapper(TextCompressionEnv(text))])

# Create a DQN agent with a custom policy
save_path = "checkpoints"
checkpoint_path = find_latest_checkpoint(save_path)
num_timesteps = get_timesteps_from_checkpoint(checkpoint_path)
if checkpoint_path is not None:
    print(f"Loading the model from the checkpoint: {checkpoint_path}")
    model = DQN.load(checkpoint_path, env=env, exploration_initial_eps=0.1, learning_rate=1e-3, learning_starts=num_timesteps, device=device, batch_size=1024)
else:
    model = DQN("MlpPolicy", env, exploration_initial_eps=0.1, learning_rate=1e-3, verbose=1, learning_starts=0, device=device, batch_size=1024)



save_callback = SaveModelCallback(check_freq=10000, save_path="checkpoints")


# Train the agent
check_freq = 100  # Set the desired frequency for efficiency updates
efficiency_callback = EfficiencyCallback(env, check_freq)
save_callback = SaveModelCallback(check_freq=10000, save_path="checkpoints")
loss_callback = LossCallback(check_freq)
model.learn(total_timesteps=int(1e6) - num_timesteps, callback=[save_callback, efficiency_callback, loss_callback])



# Evaluate the trained agent
eval_env = Monitor(env)
eval_env = ScalarActionWrapper(eval_env)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

# Test the trained agent
obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action[0]}, Reward: {reward[0]}, Done: {done[0]}")