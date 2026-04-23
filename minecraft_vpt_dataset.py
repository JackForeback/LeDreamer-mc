"""
VPT Minecraft Dataset for Dreamer4 World Model Training.

Converts OpenAI VPT .mp4/.jsonl recording pairs into the tensor format
required by Dreamer4's VideoTokenizer and DynamicsWorldModel.

VPT recordings consist of:
  - .mp4 files: 640x360 Minecraft gameplay video at 20 FPS
  - .jsonl files: One JSON object per frame with keyboard/mouse/camera actions

Dreamer4 expects:
  - video: (batch, channels, time, height, width) float32 tensors in [0, 1]
  - discrete_actions: (batch, time, num_discrete_action_types) long tensors
  - rewards: (batch, time) float32 tensors

The key design decision is how to map VPT's action space to Dreamer4's:

VPT Action Space:
  - 20 binary keyboard/mouse buttons (attack, forward, jump, ...)
  - 2D continuous camera (pitch, yaw) in degrees, discretized to 11x11=121 bins

Dreamer4 Action Space Options:
  1. All discrete: 20 binary buttons (each Discrete(2)) + 1 camera (Discrete(121))
     → num_discrete_actions = (2,2,2,...,2, 121) = tuple of 21 ints
  2. Hybrid: 20 binary buttons discrete + 2 continuous camera
     → num_discrete_actions = (2,)*20, num_continuous_actions = 2

We use option 1 (all discrete) because:
  - VPT already discretizes camera via mu-law encoding
  - Avoids continuous action normalization complexity
  - Matches how VPT's own policy head works (CategoricalActionHead)
  - Camera bins preserve the mu-law foveation (more precision near center)

Action encoding for Dreamer4:
  discrete_actions[:, :, 0:20]  = button states (0 or 1 each, num_choices=2)
  discrete_actions[:, :, 20]    = camera bin index (0 to 120, num_choices=121)

This gives: num_discrete_actions = (2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 121)
"""

import os
import json
import glob
import random
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Prefer decord for fast random-access frame seeking; fall back to cv2
try:
    import decord
    decord.bridge.set_bridge("native")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

# ─── VPT Action Constants ───────────────────────────────────────────

# The 20 binary button actions in VPT, in the order defined by Buttons.ALL
# in Video-Pre-Training/lib/actions.py
BUTTONS_ALL = [
    "attack", "back", "forward", "jump", "left", "right",
    "sneak", "sprint", "use", "drop", "inventory",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
]

N_BUTTONS = len(BUTTONS_ALL)  # 20

# VPT keyboard key → MineRL action name mapping
# (from Video-Pre-Training/run_inverse_dynamics_model.py)
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

# Camera quantization parameters matching VPT's ActionTransformer
# from Video-Pre-Training/agent.py: ACTION_TRANSFORMER_KWARGS
CAMERA_MAXVAL = 10
CAMERA_BINSIZE = 2
CAMERA_MU = 10  # mu-law parameter
N_CAMERA_BINS = 11  # per axis → 11*11 = 121 total camera actions

# Sensitivity scaler from VPT recording format
CAMERA_SCALER = 360.0 / 2400.0

# Dreamer4 action space specification:
# 20 binary buttons (each with 2 choices) + 1 camera (121 choices)
DREAMER4_NUM_DISCRETE_ACTIONS = tuple([2] * N_BUTTONS + [N_CAMERA_BINS * N_CAMERA_BINS])
# Total action dimensions for the discrete_actions tensor: 21


# ─── Camera Discretization (matching VPT's mu-law scheme) ───────────

def mu_law_encode(x: np.ndarray, mu: float = CAMERA_MU) -> np.ndarray:
    """Apply mu-law compression: maps [-maxval, maxval] → [-maxval, maxval].

    This is the same encoding used by VPT's CameraQuantizer with
    quantization_scheme="mu_law". It compresses the dynamic range so
    small camera movements get more precision (foveated discretization).
    """
    x_clipped = np.clip(x, -CAMERA_MAXVAL, CAMERA_MAXVAL)
    x_norm = x_clipped / CAMERA_MAXVAL
    encoded = np.sign(x_norm) * (np.log(1.0 + mu * np.abs(x_norm)) / np.log(1.0 + mu))
    return encoded * CAMERA_MAXVAL


def discretize_camera(camera_xy: np.ndarray) -> np.ndarray:
    """Discretize continuous camera (pitch, yaw) to bin indices.

    Matches VPT's CameraQuantizer.discretize() with mu_law scheme:
      1. Clip to [-CAMERA_MAXVAL, CAMERA_MAXVAL]
      2. Apply mu-law encoding
      3. Linear quantization to N_CAMERA_BINS bins

    Args:
        camera_xy: (..., 2) array of [pitch, yaw] in degrees

    Returns:
        (..., 2) array of bin indices, each in [0, N_CAMERA_BINS-1]
    """
    encoded = mu_law_encode(camera_xy)
    bins = np.round((encoded + CAMERA_MAXVAL) / CAMERA_BINSIZE).astype(np.int64)
    bins = np.clip(bins, 0, N_CAMERA_BINS - 1)
    return bins


def camera_bins_to_joint_index(pitch_bin: int, yaw_bin: int) -> int:
    """Combine 2D camera bins into a single index for Dreamer4.

    Joint index = pitch_bin * N_CAMERA_BINS + yaw_bin
    Range: [0, 120] for 11x11 grid
    """
    return pitch_bin * N_CAMERA_BINS + yaw_bin


# ─── JSONL Action Parsing ───────────────────────────────────────────

def parse_jsonl_action(step_data: dict, attack_is_stuck: bool = False) -> tuple:
    """Parse a single JSONL action record into MineRL-style env action.

    This replicates the logic from VPT's data_loader.py and
    run_inverse_dynamics_model.py:json_action_to_env_action().

    Args:
        step_data: One parsed JSON object from the .jsonl file
        attack_is_stuck: Whether the attack button is stuck down (recorder bug)

    Returns:
        (env_action_dict, is_null_action, new_attack_is_stuck)
    """
    # Handle attack-stuck bug (same as VPT data_loader.py lines 86-95)
    if attack_is_stuck:
        step_data["mouse"]["buttons"] = [
            b for b in step_data["mouse"]["buttons"] if b != 0
        ]

    # Build env action dict (matches NOOP_ACTION structure)
    env_action = {b: 0 for b in BUTTONS_ALL}
    env_action["ESC"] = 0
    env_action["pickItem"] = 0
    env_action["swapHands"] = 0
    env_action["camera"] = np.array([0.0, 0.0])

    is_null_action = True

    # Keyboard keys
    keyboard_keys = step_data.get("keyboard", {}).get("keys", [])
    for key in keyboard_keys:
        if key in KEYBOARD_BUTTON_MAPPING:
            action_name = KEYBOARD_BUTTON_MAPPING[key]
            if action_name in env_action:
                env_action[action_name] = 1
                if action_name in BUTTONS_ALL:
                    is_null_action = False

    # Mouse buttons: 0=attack, 1=use, 2=pickItem
    mouse = step_data.get("mouse", {})
    mouse_buttons = mouse.get("buttons", [])
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    # Camera: mouse dx/dy → degrees
    camera_action = env_action["camera"]
    camera_action[0] = mouse.get("dy", 0) * CAMERA_SCALER  # pitch
    camera_action[1] = mouse.get("dx", 0) * CAMERA_SCALER  # yaw

    if mouse.get("dx", 0) != 0 or mouse.get("dy", 0) != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    return env_action, is_null_action


def env_action_to_dreamer4(env_action: dict) -> np.ndarray:
    """Convert MineRL env action dict to Dreamer4 discrete action vector.

    Returns:
        (21,) int64 array:
          [0:20] = button values (0 or 1)
          [20]   = camera joint index (0 to 120)
    """
    action = np.zeros(N_BUTTONS + 1, dtype=np.int64)

    # Buttons
    for i, button_name in enumerate(BUTTONS_ALL):
        action[i] = int(env_action.get(button_name, 0))

    # Camera: discretize continuous degrees to bins, then to joint index
    camera_xy = env_action["camera"]  # [pitch, yaw] in degrees
    bins = discretize_camera(camera_xy)
    action[N_BUTTONS] = camera_bins_to_joint_index(int(bins[0]), int(bins[1]))

    return action


# ─── Trajectory Loading ─────────────────────────────────────────────

def load_trajectory(
    video_path: str,
    jsonl_path: str,
    target_height: int = 384,
    target_width: int = 640,
    skip_null_actions: bool = True,
) -> tuple:
    """Load a single VPT recording into arrays.

    Reads the .mp4 and .jsonl pair, parsing actions frame-by-frame using
    the same logic as VPT's data_loader.py (null-action filtering,
    attack-stuck handling, hotbar tracking, cursor overlay for GUI).

    Args:
        video_path: Path to .mp4 file
        jsonl_path: Path to .jsonl file
        target_height: Resize frames to this height
        target_width: Resize frames to this width
        skip_null_actions: Whether to skip null actions (as VPT paper does)

    Returns:
        (frames, actions, rewards) where:
          frames: (T, H, W, 3) uint8 array
          actions: (T, 21) int64 array (Dreamer4 discrete format)
          rewards: (T,) float32 array (zeros — VPT recordings have no reward)
    """
    # Load JSONL actions
    with open(jsonl_path, encoding="utf-8") as f:
        json_lines = f.readlines()
        json_data = json.loads("[" + ",".join(json_lines) + "]")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    actions = []

    # Attack-stuck workaround (same as VPT data_loader.py)
    attack_is_stuck = False
    last_hotbar = 0

    # Optional: load cursor image for GUI overlay
    cursor_path = os.path.join(
        os.path.dirname(__file__), "Video-Pre-Training", "cursors",
        "mouse_cursor_white_16x16.png"
    )
    cursor_image = None
    cursor_alpha = None
    if os.path.exists(cursor_path):
        cursor_img = cv2.imread(cursor_path, cv2.IMREAD_UNCHANGED)
        if cursor_img is not None:
            cursor_img = cursor_img[:16, :16, :]
            cursor_alpha = cursor_img[:, :, 3:] / 255.0
            cursor_image = cursor_img[:, :, :3]

    for i, step_data in enumerate(json_data):
        # Handle attack-stuck bug
        if i == 0:
            if step_data.get("mouse", {}).get("newButtons") == [0]:
                attack_is_stuck = True
        elif attack_is_stuck:
            if 0 in step_data.get("mouse", {}).get("newButtons", []):
                attack_is_stuck = False

        if attack_is_stuck:
            step_data["mouse"]["buttons"] = [
                b for b in step_data["mouse"]["buttons"] if b != 0
            ]

        # Parse action
        env_action, is_null = parse_jsonl_action(step_data, attack_is_stuck=False)

        # Hotbar tracking (VPT data_loader.py lines 99-103)
        current_hotbar = step_data.get("hotbar", 0)
        if current_hotbar != last_hotbar:
            env_action[f"hotbar.{current_hotbar + 1}"] = 1
            is_null = False
        last_hotbar = current_hotbar

        # Read corresponding video frame
        ret, frame = cap.read()
        if not ret:
            break

        # Skip null actions (as done in VPT paper)
        if skip_null_actions and is_null:
            continue

        # GUI cursor overlay (VPT data_loader.py lines 113-117)
        if step_data.get("isGuiOpen", False) and cursor_image is not None:
            h_orig = 720  # MINEREC_ORIGINAL_HEIGHT_PX
            scale = frame.shape[0] / h_orig
            cx = int(step_data["mouse"]["x"] * scale)
            cy = int(step_data["mouse"]["y"] * scale)
            _composite_cursor(frame, cursor_image, cursor_alpha, cx, cy)

        # BGR → RGB
        cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Zero-pad to target resolution (paper: 360x640 → 384x640)
        # If native resolution matches target width, pad height only;
        # otherwise fall back to resize for non-standard source video.
        h, w = frame.shape[:2]
        if w == target_width and h < target_height:
            pad_total = target_height - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            frame = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        elif h != target_height or w != target_width:
            frame = cv2.resize(frame, (target_width, target_height),
                               interpolation=cv2.INTER_LINEAR)

        # Convert to Dreamer4 action format
        d4_action = env_action_to_dreamer4(env_action)

        frames.append(frame)
        actions.append(d4_action)

    cap.release()

    if len(frames) == 0:
        return np.empty((0, target_height, target_width, 3), dtype=np.uint8), \
               np.empty((0, N_BUTTONS + 1), dtype=np.int64), \
               np.empty((0,), dtype=np.float32)

    frames = np.stack(frames, axis=0)
    actions = np.stack(actions, axis=0)
    # VPT recordings don't include reward, so we use zeros.
    # During behavior cloning this is fine — rewards are only needed
    # for RL training (Phase 3: DreamTrainer).
    rewards = np.zeros(len(frames), dtype=np.float32)

    return frames, actions, rewards


def _composite_cursor(image, cursor_img, cursor_alpha, x, y):
    """Draw cursor onto image at (x, y). Modifies image in-place."""
    ch = max(0, min(image.shape[0] - y, cursor_img.shape[0]))
    cw = max(0, min(image.shape[1] - x, cursor_img.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = cursor_alpha[:ch, :cw]
    image[y:y+ch, x:x+cw, :] = (
        image[y:y+ch, x:x+cw, :] * (1 - alpha) +
        cursor_img[:ch, :cw, :] * alpha
    ).astype(np.uint8)


# ─── Lazy Trajectory Pre-scan ──────────────────────────────────────

def prescan_trajectory(
    mp4_path: str,
    jsonl_path: str,
    skip_null_actions: bool = True,
) -> tuple:
    """Pre-scan a VPT recording to extract metadata without loading video frames.

    Reads only the JSONL (for actions and null-action filtering) and the MP4
    header (for frame count). No pixel data is loaded into memory.

    Returns:
        valid_frame_indices: (N,) int32 array of raw MP4 frame indices that
            survived null-action filtering.
        actions: (N, 21) int16 array of Dreamer4 discrete actions.
    """
    # Read JSONL
    with open(jsonl_path, encoding="utf-8") as f:
        json_data = [json.loads(line) for line in f]

    # Get frame count from video header only (no pixel decode)
    cap = cv2.VideoCapture(mp4_path)
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    valid_indices = []
    actions = []

    attack_is_stuck = False
    last_hotbar = 0

    for i, step_data in enumerate(json_data):
        if i >= n_video_frames:
            break

        # Handle attack-stuck bug (same as load_trajectory)
        if i == 0:
            if step_data.get("mouse", {}).get("newButtons") == [0]:
                attack_is_stuck = True
        elif attack_is_stuck:
            if 0 in step_data.get("mouse", {}).get("newButtons", []):
                attack_is_stuck = False

        if attack_is_stuck:
            step_data["mouse"]["buttons"] = [
                b for b in step_data["mouse"]["buttons"] if b != 0
            ]

        # Parse action
        env_action, is_null = parse_jsonl_action(step_data, attack_is_stuck=False)

        # Hotbar tracking (VPT data_loader.py lines 99-103)
        current_hotbar = step_data.get("hotbar", 0)
        if current_hotbar != last_hotbar:
            env_action[f"hotbar.{current_hotbar + 1}"] = 1
            is_null = False
        last_hotbar = current_hotbar

        # Skip null actions (as done in VPT paper)
        if skip_null_actions and is_null:
            continue

        # Convert to Dreamer4 format
        d4_action = env_action_to_dreamer4(env_action)

        valid_indices.append(i)
        actions.append(d4_action)

    valid_indices = np.array(valid_indices, dtype=np.int32)
    if len(actions) > 0:
        actions = np.stack(actions, axis=0).astype(np.int16)
    else:
        actions = np.empty((0, N_BUTTONS + 1), dtype=np.int16)

    return valid_indices, actions


# ─── Frame Decoding Backends ───────────────────────────────────────

def _zero_pad_frame(frame, image_height, image_width):
    """Zero-pad a frame to (image_height, image_width) if width matches.

    Follows the paper: 360x640 → 384x640 via symmetric zero-padding on height.
    Falls back to resize if dimensions are unexpected.
    """
    h, w = frame.shape[:2]
    if w == image_width and h < image_height:
        pad_total = image_height - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    if h != image_height or w != image_width:
        return cv2.resize(frame, (image_width, image_height),
                          interpolation=cv2.INTER_LINEAR)
    return frame


def _decode_frames_decord(mp4_path, frame_indices, image_height, image_width):
    """Decode specific frames from MP4 using decord (fast random access)."""
    vr = decord.VideoReader(mp4_path, num_threads=1)
    frames = vr.get_batch(frame_indices.tolist()).asnumpy()  # (T, H, W, 3) RGB
    if frames.shape[1] != image_height or frames.shape[2] != image_width:
        frames = np.stack([
            _zero_pad_frame(f, image_height, image_width) for f in frames
        ])
    return frames


def _decode_frames_cv2(mp4_path, frame_indices, image_height, image_width):
    """Decode specific frames from MP4 using cv2 sequential read.

    Seeks to the first needed frame and reads sequentially to the last,
    keeping only the requested indices. Efficient when frame_indices are
    roughly contiguous (as they are after null-action filtering).
    """
    first_idx = int(frame_indices[0])
    last_idx = int(frame_indices[-1])
    needed = set(int(i) for i in frame_indices)

    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)

    frames = {}
    for raw_idx in range(first_idx, last_idx + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if raw_idx in needed:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = _zero_pad_frame(frame, image_height, image_width)
            frames[raw_idx] = frame
    cap.release()

    return np.stack([frames[int(i)] for i in frame_indices])


_decode_frames = _decode_frames_decord if HAS_DECORD else _decode_frames_cv2


# ─── PyTorch Dataset ────────────────────────────────────────────────

class MinecraftVPTDataset(Dataset):
    """Lazy-loading PyTorch Dataset for VPT Minecraft recordings.

    Decodes video frames on-the-fly from the original MP4 files instead of
    pre-loading everything into RAM.  This means:
      - Zero extra disk storage (the MP4 files ARE the dataset)
      - RAM usage is independent of video count (~1 MB metadata per trajectory)
      - Scales to thousands of videos without issue

    At init time, only the JSONL files and MP4 headers are read to build a
    lightweight index of valid frame positions and pre-parsed actions.  The
    actual pixel data is decoded from the MP4 on each __getitem__ call.

    For best throughput, use DataLoader with num_workers > 0 so that frame
    decoding runs in parallel worker processes.  Install ``decord`` for
    faster random-access seeking (falls back to cv2 otherwise).

    Shape convention (per sample):
        video:   (3, seq_len, H, W) float32 in [0, 1]
        actions: (seq_len, 21) int64
        rewards: (seq_len,) float32
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 16,
        stride: int = 8,
        image_height: int = 384,
        image_width: int = 640,
        skip_null_actions: bool = True,
        max_trajectories: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory containing .mp4/.jsonl pairs
            seq_len: Number of frames per training clip
            stride: Step size between consecutive clips (for overlap)
            image_height: Target image height
            image_width: Target image width
            skip_null_actions: Skip null actions as VPT paper does
            max_trajectories: Limit number of loaded trajectories (for debugging)
        """
        super().__init__()
        self.seq_len = seq_len
        self.image_height = image_height
        self.image_width = image_width

        # Find all .mp4 files and their matching .jsonl files
        mp4_files = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
        if max_trajectories is not None:
            mp4_files = mp4_files[:max_trajectories]

        # Pre-scan: read only JSONLs + MP4 headers (no pixel data loaded)
        # Per trajectory we store:
        #   mp4_path:       string          (~100 bytes)
        #   valid_indices:  int32 array     (~4 bytes/frame)
        #   actions:        int16 array     (~42 bytes/frame)
        # Total: ~46 bytes per valid frame — vs. 196,608 bytes per frame
        # for the old eager float32 approach (4,272x smaller).
        self.trajectories = []   # list of (mp4_path, valid_indices, actions)
        self.clip_index = []     # list of (traj_idx, offset_in_valid_frames)

        print(f"Scanning {len(mp4_files)} trajectories from {data_dir}...")
        for mp4_path in mp4_files:
            base = os.path.splitext(mp4_path)[0]
            jsonl_path = base + ".jsonl"
            if not os.path.exists(jsonl_path):
                print(f"  Warning: no .jsonl for {mp4_path}, skipping")
                continue

            try:
                valid_indices, actions = prescan_trajectory(
                    mp4_path, jsonl_path,
                    skip_null_actions=skip_null_actions,
                )
            except Exception as e:
                print(f"  Error scanning {mp4_path}: {e}")
                continue

            if len(valid_indices) < seq_len:
                print(f"  Trajectory too short ({len(valid_indices)} valid frames "
                      f"< {seq_len}), skipping")
                continue

            traj_idx = len(self.trajectories)
            self.trajectories.append((mp4_path, valid_indices, actions))

            # Build clip entries with sliding window
            num_clips = (len(valid_indices) - seq_len) // stride + 1
            for c in range(num_clips):
                self.clip_index.append((traj_idx, c * stride))

        print(f"Created {len(self.clip_index)} clips of length {seq_len} "
              f"from {len(self.trajectories)} trajectories"
              f" (backend: {'decord' if HAS_DECORD else 'cv2'})")

    def __len__(self):
        """Return the number of sliding-window clips across all trajectories."""
        return len(self.clip_index)

    def __getitem__(self, idx, _retries=3):
        """Decode a clip on-the-fly and return as a dict.

        If frame decoding fails (corrupt MP4 data), retries with a random
        different clip up to ``_retries`` times so a single bad file doesn't
        crash the entire training run.

        Returns dict compatible with Dreamer4's BehaviorCloneTrainer:
            'video': (3, seq_len, H, W) float32 in [0, 1]
            'discrete_actions': (seq_len, 21) int64
            'rewards': (seq_len,) float32
        """
        for attempt in range(_retries + 1):
            try:
                traj_idx, offset = self.clip_index[idx]
                mp4_path, valid_indices, actions = self.trajectories[traj_idx]

                # Slice the frame indices and actions for this clip
                clip_frame_indices = valid_indices[offset:offset + self.seq_len]
                clip_actions = actions[offset:offset + self.seq_len]

                # Decode frames on-the-fly from the MP4 (no data stored in RAM)
                frames = _decode_frames(
                    mp4_path, clip_frame_indices,
                    self.image_height, self.image_width,
                )  # (seq_len, H, W, 3) uint8 RGB

                # (T, H, W, 3) uint8 → (3, T, H, W) float32 in [0, 1]
                video = torch.from_numpy(frames).permute(3, 0, 1, 2).float().div_(255.0)

                return {
                    'video': video,
                    'discrete_actions': torch.from_numpy(clip_actions.astype(np.int64)),
                    'rewards': torch.zeros(self.seq_len, dtype=torch.float32),
                }
            except Exception as e:
                if attempt < _retries:
                    print(f"  Warning: decode error at clip {idx} "
                          f"({mp4_path}), retrying with a different clip: {e}")
                    idx = random.randint(0, len(self.clip_index) - 1)
                else:
                    raise RuntimeError(
                        f"Failed to decode clip after {_retries + 1} attempts. "
                        f"Last error: {e}"
                    ) from e


def collate_minecraft_batch(batch: list) -> dict:
    """Custom collate function that stacks dicts into batched tensors.

    Transforms list of per-sample dicts into a single dict of batched tensors,
    matching the shapes expected by DynamicsWorldModel.forward():
      video:            (B, 3, T, H, W)
      discrete_actions: (B, T, 21)
      rewards:          (B, T)
    """
    return {
        'video': torch.stack([s['video'] for s in batch]),
        'discrete_actions': torch.stack([s['discrete_actions'] for s in batch]),
        'rewards': torch.stack([s['rewards'] for s in batch]),
    }
