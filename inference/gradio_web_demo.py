"""
THis is the main file for the gradio web demo. It uses the CogVideoX-2B model to generate videos gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.

Usage:
    OpenAI_API_KEY=your_openai_api_key OpenAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

import os
import threading
import time
import gradio as gr
import torch
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler, CogVideoXVideoToVideoPipeline
from diffusers.utils import export_to_video, load_video
from datetime import datetime, timedelta
from openai import OpenAI
import moviepy.editor as mp

initialized = ""
pipe = None
def init(name, dtype_str):
    global initialized
    global pipe
    id = f"{name}-txt2vid"
    if initialized != name:
        print(f"initializing pipeline: {name}")
        if dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        pipe = CogVideoXPipeline.from_pretrained(name, torch_dtype=dtype)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        initialized = id
        print(f"initialized {initialized}")
def init2(name, dtype_str):
    global initialized
    global pipe
    id = f"{name}-vid2vid"
    if initialized != id:
        print(f"initializing pipeline: {name}")
        if dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(name, torch_dtype=dtype)
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        initialized = id
        print(f"initialized {initialized}")

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = OpenAI()
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            model="gpt-4o",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=250,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt


def infer(prompt: str, num_inference_steps: int, guidance_scale: float, name: str, dtype: str, progress=gr.Progress(track_tqdm=True)):
    global pipe
    torch.cuda.empty_cache()
    init(name, dtype)
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        guidance_scale=guidance_scale,
    ).frames[0]
    return video

def infer2(prompt: str, videopath: str, strength: float, num_inference_steps: int, guidance_scale: float, name: str, dtype: str, progress=gr.Progress(track_tqdm=True)):
    global pipe
    torch.cuda.empty_cache()
    init2(name, dtype)
    resize_video(videopath)
    input_video = load_video(videopath)
    video = pipe(
        video=input_video,
        prompt=prompt,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).frames[0]
    return video

def resize_video(input_path, target_size=(720, 480)):
    print(f"resize video {input_path}")
    # Load the video clip
    clip = mp.VideoFileClip(input_path)

    # Remove audio
    clip = clip.without_audio()

    # Calculate the scaling factor
    width_ratio = target_size[0] / clip.w
    height_ratio = target_size[1] / clip.h
    scale_factor = min(width_ratio, height_ratio)

    print(f"resize {scale_factor}")

    # Resize the clip
    resized_clip = clip.resize(scale_factor)

    # If the resized clip is smaller than the target size, pad it
    if resized_clip.w < target_size[0] or resized_clip.h < target_size[1]:
        resized_clip = resized_clip.on_color(
            size=target_size,
            color=(0, 0, 0),  # Black padding
            pos='center'
        )

    # Write the result to a file
    resized_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Close the clips
    clip.close()
    resized_clip.close()

def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    return video_path


def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
#    gr.Markdown("""
#           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
#               CogVideoX-2B Huggingface Space🤗
#           </div>
#           <div style="text-align: center;">
#               <a href="https://huggingface.co/THUDM/CogVideoX-2B">🤗 2B Model Hub</a> |
#               <a href="https://github.com/THUDM/CogVideo">🌐 Github</a> |
#               <a href="https://arxiv.org/pdf/2408.06072">📜 arxiv </a>
#           </div>
#
#           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
#            ⚠️ This demo is for academic research and experiential use only.
#            Users should strictly adhere to local laws and ethics.
#            </div>
#           """)
    with gr.Tabs(selected=0) as tabs:
        with gr.TabItem("text-to-video", id=0):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt (Less than 200 Words. The more detailed the better.)", placeholder="Enter your prompt here", lines=5)

                    with gr.Row():
                        gr.Markdown(
                            "✨ To enhance the prompt, either set the OPENAI_API_KEY variable from the Configure menu (if you have an OpenAI API key), or just use chatgpt to enhance the prompt manually (Recommended)",
                        )
                        enhance_button = gr.Button("✨ Enhance Prompt(Optional)")

                    with gr.Row():
                        model_choice = gr.Dropdown(["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"], value="THUDM/CogVideoX-2b", label="Model")
                    with gr.Row():
                        num_inference_steps = gr.Number(label="Inference Steps", value=50)
                        guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                    with gr.Row():
                        dtype_choice = gr.Radio(["bfloat16", "float16"], label="dtype (older machines may not support bfloat16. try float16 if bfloat16 doesn't work)", value="bfloat16")
                    generate_button = gr.Button("🎬 Generate Video")

                with gr.Column():
                    video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
                    with gr.Row():
                        download_video_button = gr.File(label="📥 Download Video", visible=False)
                        download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                        send_to_vid2vid_button = gr.Button("Send to video-to-video", visible=False)
            gr.Markdown("""
            <table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
                <div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;">
                    Demo Videos with 50 Inference Steps and 6.0 Guidance Scale.
                </div>
                <tr>
                    <td style="width: 25%; vertical-align: top; font-size: 0.8em;">
                        <p>A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.</p>
                    </td>
                    <td style="width: 25%; vertical-align: top;">
                        <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="100%" controls autoplay></video>
                    </td>
                    <td style="width: 25%; vertical-align: top; font-size: 0.8em;">
                        <p>The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from its tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.</p>
                    </td>
                    <td style="width: 25%; vertical-align: top;">
                        <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="100%" controls autoplay></video>
                    </td>
                </tr>
                <tr>
                    <td style="width: 25%; vertical-align: top; font-size: 0.8em;">
                        <p>A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall.</p>
                    </td>
                    <td style="width: 25%; vertical-align: top;">
                        <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="100%" controls autoplay></video>
                    </td>
                    <td style="width: 25%; vertical-align: top; font-size: 0.8em;">
                        <p>In the haunting backdrop of a war-torn city, where ruins and crumbled walls tell a story of devastation, a poignant close-up frames a young girl. Her face is smudged with ash, a silent testament to the chaos around her. Her eyes glistening with a mix of sorrow and resilience, capturing the raw emotion of a world that has lost its innocence to the ravages of conflict.</p>
                    </td>
                    <td style="width: 25%; vertical-align: top;">
                        <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="100%" controls autoplay></video>
                    </td>
                </tr>
            </table>
            """)
        with gr.TabItem("video-to-video", id=1):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Driving Video")
                    strength = gr.Number(value=0.8, minimum=0.01, maximum=0.99)
                    prompt2 = gr.Textbox(label="Prompt (Less than 200 Words. The more detailed the better.)", placeholder="Enter your prompt here", lines=5)

                    with gr.Row():
                        gr.Markdown(
                            "✨ To enhance the prompt, either set the OPENAI_API_KEY variable from the Configure menu (if you have an OpenAI API key), or just use chatgpt to enhance the prompt manually (Recommended)",
                        )
                        enhance_button2 = gr.Button("✨ Enhance Prompt(Optional)")

                    with gr.Row():
                        model_choice2 = gr.Dropdown(["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"], value="THUDM/CogVideoX-2b", label="Model")
                    with gr.Row():
                        num_inference_steps2 = gr.Number(label="Inference Steps", value=50)
                        guidance_scale2 = gr.Number(label="Guidance Scale", value=6.0)
                    with gr.Row():
                        dtype_choice2 = gr.Radio(["bfloat16", "float16"], label="dtype (older machines may not support bfloat16. try float16 if bfloat16 doesn't work)", value="bfloat16")
                    generate_button2 = gr.Button("🎬 Generate Video")

                with gr.Column():
                    video_output2 = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
                    with gr.Row():
                        download_video_button2 = gr.File(label="📥 Download Video", visible=False)
                        download_gif_button2 = gr.File(label="📥 Download GIF", visible=False)


    def generate(prompt, num_inference_steps, guidance_scale, model_choice, dtype, progress=gr.Progress(track_tqdm=True)):
        tensor = infer(prompt, num_inference_steps, guidance_scale, model_choice, dtype, progress=progress)
        video_path = save_video(tensor)
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)
        vid2vid_update = gr.update(visible=True)

        return video_path, video_update, gif_update, vid2vid_update

    def generate2(prompt, video, strength, num_inference_steps, guidance_scale, model_choice, dtype, progress=gr.Progress(track_tqdm=True)):
        tensor = infer2(prompt, video, strength, num_inference_steps, guidance_scale, model_choice, dtype, progress=progress)
        video_path = save_video(tensor)
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)

        return video_path, video_update, gif_update

    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)

    def send_to_vid2vid(vid):
        vid2vid = gr.update(value=vid)
        tabs = gr.Tabs(selected=1)
        return [vid2vid, tabs]

    generate_button.click(
        generate,
        inputs=[prompt, num_inference_steps, guidance_scale, model_choice, dtype_choice],
        outputs=[video_output, download_video_button, download_gif_button, send_to_vid2vid_button],
    )

    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])
    send_to_vid2vid_button.click(send_to_vid2vid, inputs=[video_output], outputs=[video, tabs])

    generate_button2.click(
        generate2,
        inputs=[prompt2, video, strength, num_inference_steps2, guidance_scale2, model_choice2, dtype_choice2],
        outputs=[video_output2, download_video_button2, download_gif_button2],
    )

    enhance_button2.click(enhance_prompt_func, inputs=[prompt2], outputs=[prompt2])
if __name__ == "__main__":
    demo.launch()
