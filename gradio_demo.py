import gradio as gr
import numpy as np
import torch
import random
import av
import cv2

from PIL import Image, ImageDraw

from videollava.eval.video.monitor_module import MonitoringModule
from videollava.model.builder import load_pretrained_model
from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.compressor.llava_compressed import KeyframeSelectorLanguageBind
from videollava.model.compressor.keyframe_selector import video_to_batch
from videollava.model.compressor.keyframe_selector import batch_to_video

def tensors_to_mp4(video_tensor, output_path=None, fps=8):
    mean = np.array([0.48145466, 0.4578275,  0.40821073]).reshape(1, 3, 1, 1)
    std  = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)

    if isinstance(video_tensor, torch.Tensor):
        video_tensor = video_tensor.detach().cpu().numpy().astype(np.float32)

    frames = video_tensor * std + mean
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)
    frames = frames.transpose(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)

    if output_path is not None:
        T, H, W, _ = frames.shape
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        for i in range(T):
            writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"Saved {T} frames to {output_path}")

    return frames

def visualise_patches(frames, label_sets, show_borders, show_avg_colours):
    results = []
    base_frames= []
    for batch in label_sets:
        for i, frame in enumerate(batch):
            base_img = Image.fromarray(frames[i]).convert('RGB')
            w, h = base_img.size
            overlay = Image.new('RGB', (w + 1, h + 1), (0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for label in frame:
                label_colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                avg_colour = get_average_colour(label, base_img)
                for token in label:
                    x1, y1, x2, y2 = get_token_cords(token)

                    if show_avg_colours:
                        draw.rectangle([x1,y1,x2,y2], fill=tuple(avg_colour.tolist()) + (255,))
                    else:
                        patch = base_img.crop((x1, y1, x2, y2))
                        overlay.paste(patch, (x1,y1))

                    if show_borders:
                        col = token % 16
                        if col == 15 or token + 1 not in label:
                            draw.line([(x2,y1), (x2, y2)], fill=label_colour, width=1)
                        if col == 0 or token - 1 not in label:
                            draw.line([(x1,y1), (x1, y2)], fill=label_colour, width=1)
                        if token + 16 not in label:
                            draw.line([(x1,y2), (x2, y2)], fill=label_colour, width=1)
                        if token - 16 not in label:
                            draw.line([(x1,y1), (x2, y1)], fill=label_colour, width=1)

            results.append(overlay)
            base_frames.append(base_img)
    return results, base_frames

def get_token_cords(t):
    y, x = divmod(int(t), 16)
    x1, x2 = x * 14, (x + 1) * 14
    y1, y2 = y * 14, (y + 1) * 14
    return x1, y1, x2, y2


def get_average_colour(s, img):
    avg_colour = np.zeros(3, dtype=np.float32)
    for token in s:
        x1, y1, x2, y2 = get_token_cords(token)
        patch = np.array(img.crop((x1, y1, x2, y2)))  # (16, 16, 3)
        avg_colour += patch.mean(axis=(0, 1)).astype(int)  # average over H and W
    return (avg_colour / len(s)).astype(int)

def segmentize_video(video_input):
    video_processor = processor['video']

    container = av.open(video_input)
    frame_count = int(container.duration / av.time_base)
    print("Frame count:", frame_count)

    # Determines uniform sampling rate
    video_processor.config.vision_config.num_frames = frame_count

    video_tensor = video_processor(video_input, return_tensors='pt')['pixel_values']
    tensor = video_tensor.to(model.device, dtype=torch.float16)

    segments = keyframe_selector(video_to_batch(tensor))
    for i, clip in enumerate(segments):
        tensors_to_mp4(clip, f"clip_{i}.mp4", fps=1)

    return gr.update(interactive=True)

def generate_model_output(qs, merge_rate, retention_ratio, show_borders, show_avg_colours):
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    conv = conv_templates[conv_mode].copy()  # fresh conversation each time

    print(f"{roles[0]}: {qs}")
    # Important, placeholder img_token count needs to match the actual number of tokens
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * 8) + '\n' + qs

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    keyframes = keyframe_selector.select_keyframes(prompt)
    selected_frames = tensors_to_mp4(keyframes, fps=1)
    keyframes = batch_to_video(keyframes)

    retained_token_count = retention_ratio * 256
    merge_count = int(256 - ((256-retained_token_count) * merge_rate))
    token_count = int(retained_token_count)

    print(f"Token Count: {token_count}")
    print(f"Merge Count: {merge_count}")

    print(model.compressor)

    # merge_count and prune_count represent the number of tokens outputted by the Merger and the Pruner respectively.
    model.compressor.merge_count = merge_count
    model.compressor.prune_count = token_count

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=keyframes,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(f"{roles[1]}: {outputs}")
    merger_output, base_frames = visualise_patches(selected_frames, model.compressor.merger.get_label_set(), show_borders, show_avg_colours)
    pruner_output, _ = visualise_patches(selected_frames, model.compressor.pruner.get_label_set(), show_borders, show_avg_colours)

    return outputs, base_frames, merger_output, pruner_output

compressor_config = {'merge_count': 128, 'prune_count': 64}
tokenizer, model, processor, context_len = load_pretrained_model("LanguageBind/Video-LLaVA-7B", None,
                                                                 "Video-LLaVA-7B", load_4bit=True,
                                                                 device_map={"": "cuda:0"}, compressor_config=compressor_config)
keyframe_selector = KeyframeSelectorLanguageBind()
keyframe_selector.load_model(model.device)

model = MonitoringModule(model)

def disable_btn():
    return gr.update(interactive=False)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            qs = gr.Textbox(label="User Question")
            video = gr.Video()

            merge_rate = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label="Merge Rate", interactive=True)
            retention_ratio = gr.Slider(minimum=0, maximum=1, value=0.10, step=0.05, label="Retention Ratio", interactive=True)

            generate_btn = gr.Button("Generate Answer", interactive=False)

            video.change(fn=disable_btn, inputs=None, outputs=generate_btn)
            video.change(fn=segmentize_video, inputs=video, outputs=generate_btn)

        with gr.Column(scale=2):
            output = gr.Textbox(label="Model Output")

            with gr.Tabs():
                with gr.TabItem("Base"):
                    base_frames = gr.Gallery(label="Base Frames", columns=4)
                with gr.TabItem("Merge"):
                    merger_output = gr.Gallery(label="Merger Output", columns=4)
                with gr.TabItem("Merge + Prune"):
                    pruner_output = gr.Gallery(label="Pruner Output", columns=4)

            show_borders = gr.Checkbox(label="Show Patch Borders", value=True)
            show_avg_colour = gr.Checkbox(label="Show Averaged Patch Colours", value=True)

            generate_btn.click(fn=generate_model_output, inputs=[qs, merge_rate, retention_ratio, show_borders, show_avg_colour],
                               outputs=[output, base_frames, merger_output, pruner_output], api_name="video_inference")

demo.launch(share=True)

