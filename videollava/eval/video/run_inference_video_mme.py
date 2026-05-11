import math
import os
import argparse
import json

import av
import pandas as pd
import torch
from monitor_module import MonitoringModule
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.compressor.keyframe_selector import video_to_batch, batch_to_video
from videollava.model.compressor.llava_compressed import KeyframeSelectorLanguageBind


def split_list(lst, n):
    """Split a list into n chunks using strided splitting"""
    return [lst[i::n] for i in range(n)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    print("Chunk size:", len(chunks[k]))
    return chunks[k]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to VideoMME parquet or JSON file.', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--output_name', required=True)
    parser.add_argument('--durations', nargs='+', default=['short', 'medium'],
                        help='Duration splits to evaluate. Options: short medium long. Default: short medium')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--model_base', default=None, type=str)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--merge_count", type=int, default=None)
    parser.add_argument("--prune_count", type=int, default=None)
    parser.add_argument("--keyframe_selector", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


def load_videomme(gt_file, durations):
    """Load VideoMME annotations, filtering to the requested duration splits."""
    if gt_file.endswith('.parquet'):
        df = pd.read_parquet(gt_file)
        samples = df.to_dict(orient='records')
    else:
        with open(gt_file) as f:
            samples = json.load(f)

    filtered = [s for s in samples if s['duration'] in durations]
    print(f"Loaded {len(filtered)} samples for durations: {durations}")
    return filtered


def get_model_output(model, video_processor, tokenizer, video, qs, options, duration, args, keyframe_selector=None):
    full_question = f"{qs}\n{options}\nSelect the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D)."
    if model.config.mm_use_im_start_end:
        full_question = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN] * 8) + DEFAULT_VID_END_TOKEN + '\n' + full_question
    else:
        full_question = ''.join([DEFAULT_IMAGE_TOKEN] * 8) + '\n' + full_question

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], full_question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Sample from the video, sample rate is different depending on video length, if there is no keyframe_selector present then uniform sampling is used
    frame_count = 0
    if keyframe_selector is None:
        video_processor.config.vision_config.num_frames = 8
    else:
        container = av.open(video)
        if container is not None:
            frame_count = int(container.duration / av.time_base)
            video_processor.config.vision_config.num_frames = min(frame_count, 512)
        else:
            print("Could not get video length")
            video_processor.config.vision_config.num_frames = 8
    
    print(frame_count)
    # Shape [C, T, H, W]
    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if keyframe_selector is not None:
        keyframe_selector(video_to_batch(video_tensor.unsqueeze(0)))
        keyframes = keyframe_selector.select_keyframes(qs)
        print(f"keyframes type: {type(keyframes)}, shape: {keyframes.shape}")
        keyframes = batch_to_video(keyframes).squeeze(0)
        print(f"after batch_to_video shape: {keyframes.shape}")
    else :
        keyframes = video_tensor

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[keyframes],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff > 0:
        print(f'[Warning] {n_diff} output_ids differ from input_ids')

    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def run_inference(args):
    model_name = get_model_name_from_path(args.model_path)

    compressor_config = {}
    if args.merge_count:
        compressor_config['merge_count'] = args.merge_count
    if args.prune_count:
        compressor_config['prune_count'] = args.prune_count
    if not compressor_config:
        compressor_config = None

    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        load_4bit=True, compressor_config=compressor_config
    )
    print(f"Model device: {model.device}")
    print(f"Is 4-bit: {getattr(model, 'is_loaded_in_4bit', False)}")
    print(f"Compression config: {compressor_config}")
    print(f"KeyframeSelector enabled: {args.keyframe_selector}")

    model = MonitoringModule(model)

    # Load Keyframe Selector
    if args.keyframe_selector:
        keyframe_selector = KeyframeSelectorLanguageBind()
        keyframe_selector.load_model(model.device)
    else:
        keyframe_selector = None

    samples = load_videomme(args.gt_file, args.durations)
    samples = get_chunk(samples, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    ans_file = open(answers_file, "w")

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    for sample in tqdm(samples):
        video_id = sample['videoID']       # actual filename e.g. "fFjv93ACGo8"
        question_id = sample['video_id']   # tracking id e.g. "001"
        question = sample['question']
        # Options is a numpy array: ["A. ...", "B. ...", "C. ...", "D. ..."]
        options_str = '\n'.join(sample['options'])
        gt_answer = sample['answer']   # e.g. "A"
        duration = sample['duration']

        sample_set = {
            'id': question_id,
            'video_id': video_id,
            'question': question,
            'answer': gt_answer,
            'duration': duration,
        }

        # Find the video file
        video_path = None
        for fmt in video_formats:
            temp_path = os.path.join(args.video_dir, f"{video_id}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        if video_path is None:
            print(f"[Warning] Video not found for id: {video_id}, skipping.")
            continue

        try:
            output = get_model_output(model, processor['video'], tokenizer, video_path, question, options_str, duration,
                                      args, keyframe_selector=keyframe_selector)
            sample_set['pred'] = output
            print(f"{video_id}: {question} - Generated Answer: {output} - Actual Answer: {gt_answer}", flush=True)

            if hasattr(model, 'inference_results'):
                if latency := model.inference_results.get("inference_latency"):
                    sample_set["inference_latency"] = latency
        except Exception as e:
            print(f"[Warning] Failed on video {video_id}: {e}")
            sample_set['pred'] = ""

        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()
    print(f"Results saved to {answers_file}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)