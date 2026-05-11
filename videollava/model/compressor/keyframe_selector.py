import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F

def sample_video(video, sample_count, exclude_start, exclude_end):
    T = video.shape[0]

    start = 0 + round(exclude_start * T)
    end = (T - 1) - round(exclude_end * T)

    remaining = (end + 1) - start
    needed = sample_count - remaining

    if needed > 0:
        exclude_total = exclude_start + exclude_end
        left_expand =  round((exclude_start / exclude_total) * needed)
        right_expand = needed - left_expand

        start = max(0, start - left_expand)
        end = min(T - 1, end + right_expand)

    idx = start + ((torch.arange(sample_count) + 0.5) / sample_count) * (end - start)
    idx = idx.long()

    return video[idx]

def video_to_batch(video):
    B, C, T, H, W = video.shape  # [1, 3, 64, 224, 224]
    return video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [64, 3, 224, 224]

def batch_to_video(batch):
    return batch.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)

class KeyframeSelectorBase(ABC):

    def __init__(self, model_config: dict | None = None):
        self.model_config = model_config or {}

        self.video_segments = None
        self.selected_features = None

    @abstractmethod
    def load_model(self, device) -> None:
        pass

    @abstractmethod
    def process_video(self, videos: Any):
        pass

    @abstractmethod
    def project_features(self, features):
        pass

    @abstractmethod
    def process_text(self, prompt: str):
        pass

    # This function is intended for a single video, not multiple batches of videos
    # image_features expected shape: [F, P, D]
    def merge_frames(self, image_features, target_frame_count=1, thresh=0.75):
        frame_count = image_features.shape[0]
        shifted_features = torch.roll(image_features, -1, dims=0)

        # Initialise lists
        right_neighbour = torch.arange(1, frame_count + 1, device=image_features.device)
        left_neighbour = torch.arange(-1, frame_count - 1, device=image_features.device)
        token_count = torch.ones(frame_count, device=image_features.device, dtype=torch.float16)

        scores = F.cosine_similarity(image_features, shifted_features, dim=-1).mean(dim=-1)
        scores[-1] = -torch.inf

        for _ in range(frame_count - target_frame_count):
            m = torch.argmax(scores, dim=-1)

            if scores[m].item() < thresh:
                break

            i = left_neighbour[m]
            j = right_neighbour[m]

            # Merge the selected token and its right neighbour
            new_feature = self._merge(image_features, m, j, token_count)

            # Update the token_count list (this tensor also acts as a mask)
            token_count[j] += token_count[m]
            token_count[m] = 0

            # Update the tensors with the merged value
            shifted_features[i] = new_feature
            image_features[j] = new_feature

            # Update the scores with the merged value
            if i >= 0:
                scores[i] = F.cosine_similarity(image_features[i], new_feature, dim=-1).mean(dim=-1)
            if j < frame_count - 1:
                scores[j] = F.cosine_similarity(new_feature, shifted_features[j], dim=-1).mean(dim=-1)

            scores[m] = -torch.inf

            right_neighbour[i] = j
            left_neighbour[j] = i

        return image_features[token_count > 0], token_count

    def segmentize(self, tensor, token_count):
        res = []

        end_frames = torch.where(token_count > 0)[0].tolist()
        start_frame = 0

        for end_frame in end_frames:
            res.append(tensor[start_frame:end_frame+1, :, :, :])
            start_frame = end_frame + 1

        return res

    def filter_segments(self, segments, features):
        res = []
        feature_indices = []

        # Segment shape: (B, C, W, H)
        for i in range(len(segments)):
            if segments[i].shape[0] > 1:
                res.append(segments[i])
                feature_indices.append(i)

        return res, features[feature_indices]

    def _merge(self, image_features, i, j, token_count):
        ci = token_count[i].float()
        cj = token_count[j].float()

        total = ci + cj + 1e-8

        wi = ci / total
        wj = cj / total

        merged = wi * image_features[i] + wj * image_features[j]

        return merged.to(dtype=image_features.dtype)

    def __call__(self, video):
        video_features = self.process_video(video)

        selected_features, token_count = self.merge_frames(video_features, thresh=0.85)

        segments = self.segmentize(video, token_count)

        self.video_segments, self.selected_features = self.filter_segments(segments, selected_features)

        return self.video_segments

    def select_keyframes(self, prompt):
        prompt_encoding = self.process_text(prompt)

        with torch.no_grad():
            # Project the video tokens to the same feature space as the text encoding to enable comparison
            projected = self.project_features(self.selected_features)

        # Calculate cosine similarity between the prompt encoding and the mean CLS feature of the video segments
        similarity = projected[:, 0, :] @ prompt_encoding.T
        #print(similarity.topk(3, dim=0).indices)

        # We don't want to select any video segments shorter than 4 frames
        # mask = torch.tensor([len(seg) < 4 for seg in self.video_segments])
        # similarity[mask] = -torch.inf

        i = torch.argmax(similarity).item()

        # In the case that the video is made up from a single segment, sample uniformly from it
        if len(self.video_segments) == 1:
            keyframes = sample_video(self.video_segments[0], 8, 0, 0)
            return keyframes

        sample_counts = [1, 1, 1, 2, 1, 1, 1]
        sample_indices = [i-3, i-2, i-1, i, i+1, i+2, i+3]
        centre_idx = len(sample_indices) // 2

        st = 0
        st_count = 0
        end = len(sample_indices) - 1
        end_count = 0

        while st <= end:
            if sample_indices[st] < 0:
                st_count += sample_counts[st]
                st += 1
            elif sample_indices[end] >= len(self.video_segments):
                end_count += sample_counts[end]
                end -= 1
            else:
                break

        if st <= end:

            start_frames_len = 1 + centre_idx - st
            additional_end_count = math.floor(end_count / start_frames_len)
            remaining_end_count = end_count % start_frames_len
            for i in range(start_frames_len):
                sample_counts[centre_idx - i] += additional_end_count
            sample_counts[centre_idx] += remaining_end_count

            end_frames_len =  1 + end - centre_idx
            additional_start_count = math.floor(st_count / end_frames_len)
            remaining_start_count = st_count % end_frames_len
            for i in range(end_frames_len):
                sample_counts[centre_idx + i] += additional_start_count
            sample_counts[centre_idx] += remaining_start_count

        sample_indices = sample_indices[st:end + 1]
        sample_counts = sample_counts[st:end + 1]

        keyframes = []
        for i in range(len(sample_indices)):
            exclude_start = 0.1
            exclude_end = 0.1

            sample = sample_video(self.video_segments[sample_indices[i]], sample_counts[i], exclude_start, exclude_end)
            keyframes.append(sample)

        keyframes_combined = torch.cat(keyframes, dim=0)

        return keyframes_combined

    def select_keyframes_top3(self, prompt):
        prompt_encoding = self.process_text(prompt)

        with torch.no_grad():
            # Project the video tokens to the same feature space as the text encoding to enable comparison
            projected = self.project_features(self.selected_features)

        # Calculate cosine similarity between the prompt encoding and the mean CLS feature of the video segments
        similarity = projected[:, 0, :] @ prompt_encoding.T

        sample_indices = similarity.topk(3, dim=0).indices
        sample_counts = [3,3,2]

        # In the case that the video is made up from a single segment, sample uniformly from it
        if len(self.video_segments) == 1:
            keyframes = sample_video(self.video_segments[0], 8, 0, 0)
            return keyframes


        keyframes = []
        for i in range(len(sample_indices)):
            exclude_start = 0.1
            exclude_end = 0.1

            sample = sample_video(self.video_segments[sample_indices[i]], sample_counts[i], exclude_start, exclude_end)
            keyframes.append(sample)

        keyframes_combined = torch.cat(keyframes, dim=0)

        return keyframes_combined