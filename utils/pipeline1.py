import ast
import os
import json
import logging
import random
import time
import re
import trimesh
import sys

import backoff
import os
import base64
from PIL import Image
from io import BytesIO
from typing import Union
import traceback
import pickle
import subprocess

import open3d as o3d
import numpy as np
import math
import importlib.util

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load_module_by_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_opcad_prompt_mod = _load_module_by_path(
    "opcad_prompt_mod", os.path.join(project_path, "utils", "op-cad", "prompt.py")
)
build_incremental_cq_prompt = _opcad_prompt_mod.build_incremental_cq_prompt

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--task_name', type=str)
argparser.add_argument('--task_prompt_json_dir', type=str)
argparser.add_argument('--designer_source', type=str, default='azure')
argparser.add_argument('--critic_source', type=str, default='azure')
argparser.add_argument('--designer_lm_id', type=str, default='o3-mini')
argparser.add_argument('--critic_lm_id', type=str, default='gpt-4o')
argparser.add_argument('--step_generator_source', type=str, default=None)
argparser.add_argument('--step_generator_lm_id', type=str, default=None)
argparser.add_argument(
    '--exec_python',
    type=str,
    default=os.environ.get('ROBOTSMITH_EXEC_PYTHON', sys.executable),
    help='Python interpreter used by subprocesses that execute generated model code.'
)
args = argparser.parse_args()


log_dir = os.path.join(project_path, args.task_name, 'trial')
os.makedirs(log_dir, exist_ok=True)
n_tries = len([fil for fil in os.listdir(log_dir) if not '.' in fil])
log_dir = os.path.join(log_dir, f"{n_tries:03d}")
os.makedirs(log_dir, exist_ok=True)

def encode_image(img: Union[str, Image.Image]) -> str:
    if isinstance(img, str): # if it's image path, open and then encode/decode
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(img, Image.Image): # if it's image already, buffer and then encode/decode
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise Exception("img can only be either str or Image.Image")

class Generator:
    def __init__(self, lm_source, lm_id, max_tokens=4096, temperature=0.7, top_p=1, logger=None):
        self.lm_source = lm_source
        self.lm_id = lm_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logger = logger
        self.caller_analysis = {}
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler())
        self.max_retries = 3
        self.cost = 0 # cost in us dollars
        self.cache_path = f"cache_{lm_id}.pkl"
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}
        if self.lm_id == "gpt-4o":
            self.input_token_price = 2.5 * 10 ** -6
            self.output_token_price = 10 * 10 ** -6
        elif self.lm_id == "o3-mini":
            self.input_token_price = 1.1 * 10 ** -6
            self.output_token_price = 4.4 * 10 ** -6
        elif self.lm_id == "gpt-35-turbo":
            self.input_token_price = 1 * 10 ** -6
            self.output_token_price = 2 * 10 ** -6
        else:
            self.input_token_price = -1 * 10 ** -6
            self.output_token_price = -2 * 10 ** -6
        if self.lm_source == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
                max_retries=self.max_retries,
            ) if 'OPENAI_API_KEY' in os.environ else None
        elif self.lm_source == "azure":
            from openai import AzureOpenAI
            try:
                api_keys = json.load(open(os.path.join(project_path, "api_keys.json"), "r"))
                if "embedding" in self.lm_id:
                    api_keys = api_keys["embedding"]
                else:
                    api_keys = api_keys["all"]
                api_keys = random.sample(api_keys, 1)[0]
                self.logger.info(f"Using Azure API key: {api_keys['AZURE_ENDPOINT']}")
                self.client = AzureOpenAI(
                    azure_endpoint=api_keys['AZURE_ENDPOINT'],
                    api_key=api_keys['OPENAI_API_KEY'],
                    api_version="2024-12-01-preview",
                )
            except Exception as e:
                self.logger.error(f"Error loading .api_keys.json: {e} with traceback: {traceback.format_exc()}")
                self.client = None
        elif self.lm_source == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            # self.client = AutoModelForCausalLM.from_pretrained(self.lm_id)
            # self.tokenizer = AutoTokenizer.from_pretrained(self.lm_id)
            self.client = pipeline(
                "text-generation",
                model=self.lm_id,
                device_map="auto",
            )
            # lm_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif self.lm_source == "llava":
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.constants import (
                IMAGE_TOKEN_INDEX,
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
                IMAGE_PLACEHOLDER,
            )
            from llava.conversation import conv_templates, SeparatorStyle
            import torch
            from llava.mm_utils import (
                process_images,
                tokenizer_image_token,
                get_model_name_from_path,
                KeywordsStoppingCriteria,
            )
            self.model_name = get_model_name_from_path(self.lm_id)
            if 'lora' in self.model_name and '7b' in self.model_name:
                self.lm_base = "liuhaotian/llava-v1.5-7b"
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.lm_id, model_base=self.lm_base, model_name=self.model_name, )  # load_4bit=True)
        elif self.lm_source == "vla": # will merge to huggingface later
            from transformers import AutoModelForVision2Seq, AutoProcessor
            from peft import PeftModel
            import torch
            self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
            self.base_model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                "/home/zheyuanzhang/Documents/GitHub/VLA/adapter_tmp/openvla-7b+ella_dataset+b16+lr-0.0005+lora-r32+dropout-0.0", # will add the lora adapter path into env config later
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to("cuda:0")
        elif self.lm_source == "google":
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            from google.api_core.exceptions import ResourceExhausted
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.client = genai.GenerativeModel(self.lm_id)
        elif self.lm_source == "local":
            from openai import OpenAI
            from tools.model_manager import global_model_manager
            self.client = global_model_manager.get_model("completion")
            self.embed_client = global_model_manager.get_model("embedding")
            self.input_token_price = 0
            self.output_token_price = 0
        else:
            raise NotImplementedError(f"{self.lm_source} is not supported!")

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None, img=None, json_mode=False, chat_history=None, caller="none"):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
            
        if self.lm_source == 'openai' or self.lm_source == 'azure':
            return self.openai_generate(prompt, max_tokens, temperature, top_p, img, json_mode, chat_history, caller)
        elif self.lm_source == 'gemini':
            return self.gemini_generate(prompt)
        elif self.lm_source == 'huggingface':
            return self.huggingface_generate(prompt, max_tokens, temperature, top_p)
        elif self.lm_source == 'vla':
            return self.vla_generate(prompt, img, max_tokens)
        elif self.lm_source == 'local':
            message = [] if chat_history is None else chat_history
            message.append({ "role": "user", "content": prompt })
            return self.client.complete(message, max_tokens, temperature, top_p)
        else:
            raise ValueError(f"Invalid lm_source: {self.lm_source}")

    def openai_generate(self, prompt, max_tokens, temperature, top_p, img: Union[str, Image.Image, None, list], json_mode=False, chat_history=None, caller="none"):
        @backoff.on_exception(
            backoff.expo,  # Exponential backoff
            Exception,  # Base exception to catch and retry on
            max_tries=self.max_retries,  # Maximum number of retries
            jitter=backoff.full_jitter,  # Add full jitter to the backoff
            logger=self.logger  # Logger for retry events, which is in the level of INFO
        )
        def _generate():
            content = [{
                        "type": "text",
                        "text": prompt
                    }, ]
            if img is not None:
                if type(img) != list:
                    imgs = [img]
                else:
                    imgs = img
                for each_img in imgs:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(each_img)}"},
                        # "detail": "low"
                    })
            if chat_history is not None:
                messages = chat_history
            else:
                messages = []
            messages.append(
                {
                    "role": "user",
                    "content": content
                })
            start = time.perf_counter()
            if self.lm_id[0] == 'o':
                response = self.client.chat.completions.create(
                    # reasoning_effort='high',
                    model=self.lm_id,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    timeout=40,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.lm_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    response_format={
                        "type": "json_object" if json_mode else "text"
                    },
                    timeout=40,
                )
            self.logger.debug(f"api request time: {time.perf_counter() - start}")
            with open(f"chat_raw.jsonl", 'a') as f:
                chat_entry = {
                    "prompt": prompt,
                    "response": response.model_dump_json(indent=4)
                }
                # Write as a single JSON object per line
                f.write(json.dumps(chat_entry))
                f.write('\n')
            usage = dict(response.usage)
            self.cost += usage['completion_tokens'] * self.output_token_price + usage['prompt_tokens'] * self.input_token_price
            if caller in self.caller_analysis:
                self.caller_analysis[caller].append(usage['total_tokens'])
            else:
                self.caller_analysis[caller] = [usage['total_tokens']]
            response = response.choices[0].message.content
            # self.logger.debug(f'======= prompt ======= \n{prompt}', )
            # self.logger.debug(f'======= response ======= \n{response}')
            # self.logger.debug(f'======= usage ======= \n{usage}')
            if self.cost > 7:
                self.logger.critical(f'COST ABOVE 7 dollars! There must be sth wrong. Stop the exp immediately!')
                raise Exception(f'COST ABOVE 7 dollars! There must be sth wrong. Stop the exp immediately!')
            self.logger.info(f'======= total cost ======= {self.cost}')
            return response
        try:
            return _generate()
        except Exception as e:
            self.logger.error(f"Error with openai_generate: {e}, the prompt was:\n {prompt}")
            return None

    def gemini_generate(self, prompt):
        try:
            response = self.client.generate_content(prompt, safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            })
            usage = response.usage_metadata.total_token_count
            self.cost += usage * self.input_token_price
            response = response.text
            self.logger.debug(f'======= prompt ======= \n{prompt}', )
            self.logger.debug(f'======= response ======= \n{response}')
            self.logger.debug(f'======= usage ======= \n{usage}')
            self.logger.debug(f'======= total cost ======= {self.cost}')
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise e
        return response

    def huggingface_generate(self, prompt, max_tokens, temperature, top_p):
        messages = []
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant."
            })
        messages.append(
            {
                "role": "user",
                "content": prompt
            })
        response = self.client(
            prompt,
            do_sample = False if temperature == 0 else True,
            temperature=temperature if temperature != 0 else 1,
            top_p=top_p,
            max_new_tokens=max_tokens,)
        response = response[0]['generated_text']
        self.logger.debug(f'======= prompt ======= \n{prompt}', )
        self.logger.debug(f'======= response ======= \n{response}')
        return response
        # inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding="max_length", max_length=self.max_tokens, truncation=True)
        # outputs = self.client.generate(**inputs, max_length=self.max_tokens, num_return_sequences=1, temperature=self.temperature, top_p=self.top_p)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # self.logger.debug(f'======= prompt ======= \n{prompt}', )
        # self.logger.debug(f'======= response ======= \n{response}')

    def vla_generate(self, prompt, img, max_tokens):
        import torch
        inputs = self.processor(prompt, Image.fromarray(img)).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = self.lora_model.generate(**inputs, max_length=max_tokens)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

designer = None
critic = None
step_generator_agent = None


def init_agents(designer_source='azure', critic_source='azure', designer_lm_id='o3-mini', critic_lm_id='gpt-4o',
                step_generator_source=None, step_generator_lm_id=None):
    global designer, critic, step_generator_agent
    designer = Generator(
        lm_source=designer_source,
        lm_id=designer_lm_id,
        max_tokens=16000,
        temperature=0.7,
        top_p=1.0,
        logger=None
    )
    step_generator_agent = None
    if step_generator_source and step_generator_lm_id:
        step_generator_agent = Generator(
            lm_source=step_generator_source,
            lm_id=step_generator_lm_id,
            max_tokens=16000,
            temperature=0.2,
            top_p=1.0,
            logger=None
        )
    critic = Generator(
        lm_source=critic_source,
        lm_id=critic_lm_id,
        max_tokens=16000,
        temperature=0.7,
        top_p=1.0,
        logger=None
    )

def parse_json(prompt, response, last_call=False):
    json_str = None
    if "```json" in response:
        # Step 1: Extract the JSON part
        start = response.find("```json") + len("```json")
        end = response.find("```", start)
        json_str = response[start:end].strip()
    else:
        if not last_call:
            chat_history = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            data = designer.generate(
                f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!", chat_history=chat_history)
            return parse_json(None, data, last_call=True)
        else:
            return None
    try:
        response = json.loads(json_str)
    except json.JSONDecodeError as e:
        if not last_call:
            chat_history = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            data = designer.generate(
                f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!", chat_history=chat_history)
            return parse_json(None, data, last_call=True)
    return response

def parse_code_block(response, language_hint="python"):
    if response is None:
        return None
    fence = f"```{language_hint}"
    if fence in response:
        start = response.find(fence) + len(fence)
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    return response.strip()

def get_single_prompt(prompt):
        incremental_prompt = """
"### Role\n"
"You are an expert CAD modeling assistant specialized in CadQuery.\n"
"Generate ONLY the incremental CadQuery code needed to perform the requested operation, as a continuation of the provided previous code context."

### Context (already executed Python code)
```python
{previous_code}
```

### Instruction
Perform the following operation **as a continuation** of the existing model:
> {operation_instruction}


""".strip()
        return incremental_prompt


def _strip_markdown_fence(text):
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _result_var_name(step_id):
    return f"result_{step_id}"


def _build_opcad_step_prompt(previous_code, step_id, instruction):
    current_var = None if step_id == 0 else _result_var_name(step_id - 1)
    next_var = _result_var_name(step_id)
    return build_incremental_cq_prompt(
        previous_code=previous_code or "",
        operation_instruction=instruction,
        current_var_name=current_var,
        next_var_name=next_var,
        allow_comments=False,
        add_size_guidelines=True,
        op_kind=f"tool_mesh_step_{step_id}",
    )


def _finalize_assemble_func_to_stl(assemble_func_code):
    cleaned = _strip_markdown_fence(assemble_func_code).rstrip()
    if "def assemble(" not in cleaned:
        raise ValueError("Generated assemble code must contain `def assemble(parts):`")
    if "return mesh_files" in cleaned:
        cleaned = cleaned.replace("return mesh_files", "return _ensure_stl_outputs(mesh_files)")
    elif "return filenames" in cleaned:
        cleaned = cleaned.replace("return filenames", "return _ensure_stl_outputs(filenames)")
    elif re.search(r"\breturn\s+\[", cleaned):
        cleaned = re.sub(r"\breturn\s+(.+)$", r"return _ensure_stl_outputs(\1)", cleaned, count=1, flags=re.M)
    else:
        cleaned += "\n\n    return _ensure_stl_outputs(mesh_files)"

    stl_helper = """

def _ensure_stl_outputs(paths):
    normalized = []
    for p in paths:
        ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p else ""
        if ext == ".stl":
            normalized.append(p)
            continue
        mesh = trimesh.load(p, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        stl_path = (p.rsplit(".", 1)[0] if "." in p else p) + ".stl"
        mesh.export(stl_path)
        normalized.append(stl_path)
    return normalized
""".rstrip()
    if "def _ensure_stl_outputs(" not in cleaned:
        cleaned = cleaned + "\n" + stl_helper + "\n"
    return cleaned


def _resolve_opcad_generator(designer_prompt_json, fallback_generator, agent_generator=None):
    opcad_cfg = designer_prompt_json.get("OPCAD_GENERATOR", {}) if isinstance(designer_prompt_json, dict) else {}
    if not isinstance(opcad_cfg, dict):
        opcad_cfg = {}
    source = opcad_cfg.get("source")
    model_id = opcad_cfg.get("lm_id")
    if source and model_id:
        append_execution_log(log_dir, f"Using dedicated OP-CAD generator: source={source}, lm_id={model_id}")
        return Generator(
            lm_source=source,
            lm_id=model_id,
            max_tokens=opcad_cfg.get("max_tokens", 16000),
            temperature=opcad_cfg.get("temperature", 0.2),
            top_p=opcad_cfg.get("top_p", 1.0),
            logger=None
        )
    if agent_generator is not None:
        append_execution_log(
            log_dir,
            f"Using dedicated step-generator agent: source={agent_generator.lm_source}, lm_id={agent_generator.lm_id}"
        )
        return agent_generator
    return fallback_generator


import re


def _indent_block(code: str, indent: str = "    ") -> str:
    lines = code.splitlines()
    return "\n".join((indent + line) if line.strip() else "" for line in lines)


def _find_last_result_assignment(code: str):
    """
    找最后一个 result / result_x 的赋值位置
    """
    pattern = re.compile(r"(?m)^(?P<indent>\s*)(?P<name>result(?:_\d+)?)\s*=")
    matches = list(pattern.finditer(code))
    return matches[-1] if matches else None


def _normalize_incremental_step_code(step_code: str, step_id: int) -> str:
    """
    把 LLM 返回的“单步增量代码”规范成：
    - 当前步输出变量一定是 result_{step_id}
    - 上一步结果统一引用为 result_{step_id-1}
    - 兼容模型输出 result / result_n / result_x 等写法
    """
    step_code = _strip_markdown_fence(step_code).strip()
    if not step_code:
        raise ValueError(f"Step {step_id} returned empty code.")

    prev_var = f"result_{step_id - 1}" if step_id > 0 else None
    next_var = f"result_{step_id}"

    last_assign = _find_last_result_assignment(step_code)
    if last_assign is None:
        raise ValueError(
            f"Step {step_id} code must contain a final assignment to result / result_x."
        )

    # 先把“最后一个结果赋值目标”打个占位，避免后面替换 RHS 时把 LHS 也误改了
    step_code = (
        step_code[:last_assign.start("name")]
        + "__RESULT_TARGET__"
        + step_code[last_assign.end("name"):]
    )

    # 把 prompt 里的占位式写法 result_n 统一替换成上一结果
    if prev_var is not None:
        step_code = re.sub(r"\bresult_n\b", prev_var, step_code)

        # 如果模型偷懒写成 result.union(...) / result.cut(...) / result.edges(...)
        # 这里也统一解释成“上一结果”
        step_code = re.sub(r"\bresult\b", prev_var, step_code)
    else:
        # 第一步不该引用历史 result；如果模型写了 result_n，直接保留给后续报错更清晰
        step_code = re.sub(r"\bresult_n\b", "result_n", step_code)

    # 最后把当前步输出变量固定为 result_{step_id}
    step_code = step_code.replace("__RESULT_TARGET__", next_var)

    return step_code.strip()


def _detect_last_result_var(code: str) -> str:
    """从前序代码中自动检测最后一个 result_x 变量名；没有则返回 default。"""
    matches = re.findall(r"\b(result_\d+)\b\s*=", code)
    return matches[-1] if matches else None

def _wrap_incremental_code_as_assemble(accumulated_code: str) -> str:
    """
    把累计的增量代码包成完整 assemble(parts) 函数。
    默认导出为 STL。
    """
    final_var = _detect_last_result_var(accumulated_code)
    if not final_var:
        raise ValueError("No result_x variable found in accumulated incremental code.")

    body = _indent_block(accumulated_code)

    assemble_code = f"""
def assemble(parts):
{body}

    out_path = "assembled_tool.stl"
    cq.exporters.export({final_var}, out_path)
    return [out_path]
""".strip()

    return assemble_code


def generate_tool_from_steps(tool_json, designer_prompt_json, design_chat_history, step_generator=None):
    steps = tool_json.get("construction_steps", None)
    if not steps:
        raise ValueError("construction_steps is required for OP-CAD incremental generation")
    if step_generator is None:
        raise ValueError("A step generator is required for OP-CAD incremental generation")

    current_code = ""   # 这里只存“累计后的增量 CadQuery 片段”

    for step in steps:
        step_id = int(step.get("step_id", 0))
        instruction = step.get("Instruction", "")

        prompt = _build_opcad_step_prompt(
            previous_code=current_code,
            step_id=step_id,
            instruction=instruction,
        )

        llm_response = step_generator.generate(
            prompt=prompt,
            img=None,
            json_mode=False,
            chat_history=None
        )

        raw_step_code = parse_code_block(llm_response, language_hint="python")
        raw_step_code = _strip_markdown_fence(raw_step_code)

        normalized_step_code = _normalize_incremental_step_code(raw_step_code, step_id)

        updated_code = (
            (current_code.rstrip() + "\n\n" + normalized_step_code.lstrip()).strip()
            if current_code else normalized_step_code
        )

        # 每一步都对“完整 assemble 函数”做校验，而不是只校验单步片段
        candidate_assemble_func = _wrap_incremental_code_as_assemble(updated_code)
        is_safe, safety_msg = static_validate_assemble_func(candidate_assemble_func)
        append_execution_log(log_dir, f"[step_codegen {step_id}] static validation: {safety_msg}")
        if not is_safe:
            raise ValueError(f"Step {step_id} generated unsafe assemble code: {safety_msg}")

        current_code = updated_code

        design_chat_history.append({
            "role": "assistant",
            "content": f"Step {step_id} updated assemble code."
        })

    final_assemble_func = _wrap_incremental_code_as_assemble(current_code)
    return _finalize_assemble_func_to_stl(final_assemble_func)


def write_design_code(filename, tool_json):
    outp = ''
    print('tool_json', tool_json)

    with open(os.path.join(project_path, 'utils', 'api_tool_design.py'), 'r') as fi:
        outp += fi.read()
        outp += '\n\n\n\n\n'

    outp += tool_json['assemble_func']
    outp += '\n\n\n'

    parts_obj = tool_json.get('parts', {})   # 没有 parts 就给空 dict
    outp += 'parts = '
    parts = json.dumps(parts_obj, indent=4)
    parts = parts.replace('true', 'True').replace('false', 'False')
    outp += parts
    outp += '\n'

    outp += 'filenames = assemble(parts)\n'
    outp += 'print(filenames)\n'

    print(outp)

    with open(filename, 'w') as fo:
        fo.write(outp)


ALLOWED_CALL_ROOTS = {
    "primitive", "generate_3d", "rotate_to_align", "get_position",
    "get_axis_align_bounding_box", "get_volume", "rescale", "move",
    "empty_grid", "add_mesh", "sub_mesh", "cut_grid", "grid_to_mesh",
    "trimesh", "assemble", "Plane", "Vector", "Workplane",
    "len", "range", "float", "int", "str", "list", "dict", "tuple", "set",
    "abs", "min", "max", "sum", "enumerate", "zip"
}
BLOCKED_ROOTS = {
    "os", "subprocess", "socket", "requests", "urllib", "http", "https",
    "ftplib", "telnetlib", "paramiko"
}
BLOCKED_ATTRS = {
    "system", "popen", "Popen", "run", "call", "check_output", "urlopen",
    "get", "post", "put", "delete", "request"
}


def _get_call_root(node):
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def static_validate_assemble_func(assemble_func):
    try:
        tree = ast.parse(assemble_func)
    except SyntaxError as e:
        return False, f"Syntax error in assemble_func: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Import statements are not allowed in assemble_func."
        if isinstance(node, ast.Call):
            root = _get_call_root(node.func)
            if root in BLOCKED_ROOTS:
                return False, f"Blocked callable root detected: {root}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_ATTRS:
                return False, f"Blocked callable attribute detected: {node.func.attr}"
            if isinstance(node.func, ast.Name) and root is not None and root not in ALLOWED_CALL_ROOTS:
                return False, f"Call root not in whitelist: {root}"
        if isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRS:
            return False, f"Blocked attribute access detected: {node.attr}"
        if isinstance(node, ast.Name) and node.id in BLOCKED_ROOTS:
            return False, f"Blocked module reference detected: {node.id}"
    return True, "assemble_func passed static whitelist validation"


def append_execution_log(log_dir, message):
    with open(os.path.join(log_dir, "execution.log"), "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

def look_at(cam_pos, target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    forward = (target - cam_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.eye(4)
    R[:3, :3] = np.stack([right, up, -forward], axis=1)
    R[:3, 3] = -cam_pos
    return np.linalg.inv(R)

def render_and_save(mesh_path, output_folder, num_views=10):
    os.makedirs(output_folder, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())        
    vertices = np.asarray(mesh.vertices)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1]) 

    for i in range(num_views):
        # Random camera position on a sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        radius = max(abs(vertices.min()), vertices.max()) * 3
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        cam_pos = np.array([x, y, z])

        # Compute extrinsic
        extrinsic = look_at(cam_pos)

        # Set the view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()

        # Save image
        img = vis.capture_screen_float_buffer(False)
        img = (255 * np.asarray(img)).astype(np.uint8)
        o3d.io.write_image(f"{output_folder}/{i:03d}.png", o3d.geometry.Image(img))

    vis.destroy_window()

def render_and_save_with_objects(mesh_path, json_filename, output_folder, num_views=10):
    os.makedirs(output_folder, exist_ok=True)

    tool_json = json.load(open(json_filename, 'r'))
    tool_placement = tool_json["placement_func"]
    placement_filename_matches = re.findall(r"(filename)\s*=\s*\"([^\"]+)\"", tool_placement)
    if len(placement_filename_matches) == 0:
        raise ValueError("placement_func must contain filename=\"...\" in gs.Morph.Mesh.")
    _, placement_filename = placement_filename_matches[0]

    mesh_dir = os.path.dirname(mesh_path)
    candidate_mesh_path = os.path.join(mesh_dir, placement_filename)
    if os.path.exists(candidate_mesh_path):
        mesh_path = candidate_mesh_path
    elif not os.path.exists(mesh_path):
        available_mesh_filenames = sorted(
            [f for f in os.listdir(mesh_dir) if os.path.isfile(os.path.join(mesh_dir, f)) and os.path.splitext(f)[1].lower() in [".obj", ".stl"]]
        )
        raise FileNotFoundError(
            f"Mesh file not found: {placement_filename}. Available mesh files: {available_mesh_filenames}"
        )

    _, p1, p2, p3 = re.findall(r"(pos)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", tool_placement)[0]
    _, e1, e2, e3 = re.findall(r"(euler)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", tool_placement)[0]
    _, s1, s2, s3 = re.findall(r"(scale)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", tool_placement)[0]
    tool_pos = np.array([float(p1), float(p2), float(p3)])
    tool_euler = np.array([float(e1), float(e2), float(e3)])
    tool_scale = np.array([float(s1), float(s2), float(s3)])

    mesh_ext = os.path.splitext(mesh_path)[1].lower()
    if mesh_ext not in [".stl", ".obj"]:
        raise ValueError(f"Unsupported mesh format: {mesh_ext}. Only .stl and .obj are supported.")

    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    trimesh.repair.fix_normals(mesh, multibody=True)
    trimesh.repair.fill_holes(mesh)

    repaired_mesh_path = os.path.splitext(mesh_path)[0] + "_repaired" + mesh_ext
    mesh.export(repaired_mesh_path)

    mesh.apply_scale(tool_scale)
    rotation = trimesh.transformers.euler_matrix(
        math.radians(float(e1)), math.radians(float(e2)), math.radians(float(e3)), axes='sxyz'
    )
    mesh.apply_transform(rotation)
    mesh.apply_translation(tool_pos)
    new_mesh_path = os.path.splitext(repaired_mesh_path)[0] + "_loaded.obj"
    mesh.export(new_mesh_path)

    mesh = o3d.io.read_triangle_mesh(new_mesh_path)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())        
    vertices = np.asarray(mesh.vertices)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1]) 

    for i in range(num_views):
        # Random camera position on a sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        radius = max(abs(vertices.min()), vertices.max()) * 3
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        cam_pos = np.array([x, y, z])

        # Compute extrinsic
        extrinsic = look_at(cam_pos)

        # Set the view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()

        # Save image
        img = vis.capture_screen_float_buffer(False)
        img = (255 * np.asarray(img)).astype(np.uint8)
        o3d.io.write_image(f"{output_folder}/{i:03d}.png", o3d.geometry.Image(img))

    vis.destroy_window()

def render_and_save_with_genesis(mesh_path, output_folder, num_views=10):
    os.makedirs(output_folder, exist_ok=True)
    scene_envname = 'Reaching'
    prog = f"import genesis as gs\nclass Env({scene_envname}):\n        "

    with open('tmp.py', 'w') as fo:
        fo.write(prog)
    
    


def post_process_output_meshes(output_files, base_dir):
    processed_files = []
    for rel_path in output_files:
        raw_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Generated mesh does not exist: {raw_path}")
        mesh = trimesh.load(raw_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise RuntimeError(f"Generated mesh is empty: {raw_path}")
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fill_holes(mesh)
        repaired_path = os.path.splitext(raw_path)[0] + "_post.stl"
        mesh.export(repaired_path)
        processed_files.append(repaired_path)
    return processed_files


def _force_placement_filename(placement_func: str, filename: str = "assembled_tool.stl") -> str:
    if not placement_func:
        return placement_func
    return re.sub(r'filename\s*=\s*"[^"]+"', f'filename="{filename}"', placement_func)

def _normalize_construction_steps(tool_json):
    if isinstance(tool_json, str):
        candidate = tool_json.strip()
        try:
            tool_json = json.loads(candidate)
        except json.JSONDecodeError:
            if "```json" in candidate:
                start = candidate.find("```json") + len("```json")
                end = candidate.find("```", start)
                candidate = candidate[start:end].strip() if end != -1 else candidate[start:].strip()
                tool_json = json.loads(candidate)
            else:
                raise ValueError("Designer response is a string but not valid JSON content")

    if not isinstance(tool_json, dict):
        raise ValueError(f"Designer response must be a JSON object, got {type(tool_json).__name__}")

    steps = tool_json.get("construction_steps", [])
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("construction_steps must be a non-empty list")

    steps = sorted(steps, key=lambda x: int(x.get("step_id", 0)))

    normalized = []
    for expected_id, step in enumerate(steps):
        step = dict(step)
        actual_id = int(step.get("step_id", -1))
        if actual_id != expected_id:
            step["step_id"] = expected_id
        normalized.append(step)

    tool_json["construction_steps"] = normalized
    return tool_json


def _resolve_task_prompt_json_path(task_prompt_json_dir: str, task_name: str = None) -> str:
    """Resolve task prompt json path from either a file path or a task directory."""
    candidates = []

    def _add(path):
        if path and path not in candidates:
            candidates.append(path)

    normalized = os.path.expanduser(task_prompt_json_dir) if task_prompt_json_dir else task_prompt_json_dir
    if normalized:
        _add(normalized)
        if os.path.isdir(normalized):
            _add(os.path.join(normalized, 'task_prompt.json'))
        elif not normalized.endswith('.json'):
            _add(os.path.join(normalized, 'task_prompt.json'))

        if not os.path.isabs(normalized):
            _add(os.path.join(project_path, normalized))
            _add(os.path.join(project_path, normalized, 'task_prompt.json'))
        else:
            abs_parts = os.path.normpath(normalized).split(os.sep)
            if 'RobotSmith-git' in abs_parts:
                idx = abs_parts.index('RobotSmith-git')
                rel_after_repo = os.path.join(*abs_parts[idx + 1:]) if idx + 1 < len(abs_parts) else ''
                _add(os.path.join(project_path, rel_after_repo))
                _add(os.path.join(project_path, rel_after_repo, 'task_prompt.json'))

    if task_name:
        _add(os.path.join(project_path, task_name, 'task_prompt.json'))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Cannot locate task_prompt.json from '{task_prompt_json_dir}'. Tried: {candidates}"
    )


def run_tool_design(task_name, task_prompt_json_dir,
                    designer_source='azure', critic_source='azure',
                    designer_lm_id='o3-mini', critic_lm_id='gpt-4o',
                    step_generator_source=None, step_generator_lm_id=None):
    init_agents(
        designer_source=designer_source,
        critic_source=critic_source,
        designer_lm_id=designer_lm_id,
        critic_lm_id=critic_lm_id,
        step_generator_source=step_generator_source,
        step_generator_lm_id=step_generator_lm_id
    )

    append_execution_log(
        log_dir,
        (
            f"Initialized agents: designer=({designer_source}, {designer_lm_id}), "
            f"critic=({critic_source}, {critic_lm_id}), "
            f"step_generator=({step_generator_source}, {step_generator_lm_id})"
        )
    )

    designer_prompt = open(os.path.join(project_path, 'utils', 'template_tool_design.txt'), 'r').read()
    task_prompt_json_path = _resolve_task_prompt_json_path(task_prompt_json_dir, task_name=task_name)
    designer_prompt_json = json.load(open(task_prompt_json_path, 'r'))

    append_execution_log(log_dir, f"Using task prompt json: {task_prompt_json_path}")

    designer_prompt = designer_prompt.replace("$3D_OBJECT_DESCRIPTION$", designer_prompt_json['3D_OBJECT_DESCRIPTION'])
    designer_prompt = designer_prompt.replace("$GOAL_DESCRIPTION$", designer_prompt_json['GOAL_DESCRIPTION'])
    designer_prompt = designer_prompt.replace("$3D_CONFIGURATION$", designer_prompt_json['3D_CONFIGURATION'])
    designer_prompt = designer_prompt.replace("$TIPS_FOR_DESIGNER$", designer_prompt_json['TIPS_FOR_DESIGNER'])

    step_generator = _resolve_opcad_generator(designer_prompt_json, designer, step_generator_agent)

    designer_response = designer.generate(prompt=designer_prompt, img=None, json_mode=False)

    critic_cnt = 0
    design_chat_history = [{
        'role': 'user',
        'content': designer_prompt
    }]

    while critic_cnt <= 5:
        critic_cnt += 1
        designer_response_parsed = None

        try:
            designer_response = parse_json(designer_prompt, designer_response)
            designer_response = _normalize_construction_steps(designer_response)
            designer_response_parsed = designer_response

            required_keys = [
                "name",
                "functional_purpose",
                "construction_steps",
                "simulation_properties",
                "placement_func",
            ]
            for k in required_keys:
                if k not in designer_response:
                    raise ValueError(f"Designer JSON missing required field: {k}")

            # 统一 placement_func 中的文件名，和 generator 输出一致
            designer_response["placement_func"] = _force_placement_filename(
                designer_response.get("placement_func", ""),
                "assembled_tool.stl"
            )

            # generator 根据 designer 的 construction_steps 生成 assemble_func
            stepwise_assemble_func = generate_tool_from_steps(
                tool_json=designer_response,
                designer_prompt_json=designer_prompt_json,
                design_chat_history=design_chat_history,
                step_generator=step_generator
            )
            designer_response["assemble_func"] = stepwise_assemble_func

            json_filename = os.path.join(log_dir, f"design{critic_cnt}.json")
            json.dump(designer_response, open(json_filename, 'w'), indent=4)

            code_filename = os.path.join(log_dir, f"design{critic_cnt}.py")

            is_safe, safety_msg = static_validate_assemble_func(designer_response["assemble_func"])
            append_execution_log(log_dir, f"[critic {critic_cnt}] static validation: {safety_msg}")
            if not is_safe:
                raise ValueError(f"Static whitelist check failed: {safety_msg}")

            write_design_code(code_filename, designer_response)

            design_chat_history.append({
                'role': 'assistant',
                'content': json.dumps(designer_response)
            })

            result = subprocess.run(
                [args.exec_python, code_filename],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise Exception(f"Error in subprocess: {result.stderr}")

            output_files = ast.literal_eval(result.stdout)
            if not isinstance(output_files, list):
                raise ValueError("Output files should be a list")

            output_files = post_process_output_meshes(output_files, os.getcwd())

            os.makedirs(f"{log_dir}/{critic_cnt}", exist_ok=True)
            imgs = []

            for output_file in output_files:
                subprocess.run(["cp", output_file, f"{log_dir}/{critic_cnt}/"], check=False)
                render_and_save_with_objects(
                    f"{log_dir}/{critic_cnt}/{os.path.basename(output_file)}",
                    json_filename,
                    f"{log_dir}/{critic_cnt}/rendered_views",
                    num_views=5
                )
                for i in range(5):
                    imgs.append(os.path.join(f"{log_dir}/{critic_cnt}/rendered_views", f"{i:03d}.png"))

            critic_prompt = open(os.path.join(project_path, 'utils', 'template_tool_critic.txt'), 'r').read()
            critic_prompt = critic_prompt.replace("$3D_OBJECT_DESCRIPTION$", designer_prompt_json['3D_OBJECT_DESCRIPTION'])
            critic_prompt = critic_prompt.replace("$GOAL_DESCRIPTION$", designer_prompt_json['GOAL_DESCRIPTION'])
            critic_prompt = critic_prompt.replace("$3D_CONFIGURATION$", designer_prompt_json['3D_CONFIGURATION'])

            critic_response = critic.generate(prompt=critic_prompt, img=imgs, json_mode=False)

            if 'DONE' in critic_response:
                break

            designer_prompt = critic_response
            design_chat_history.append({
                'role': 'user',
                'content': designer_prompt
            })

            designer_response = designer.generate(
                prompt=designer_prompt,
                img=None,
                json_mode=False,
                chat_history=design_chat_history
            )

        except Exception as e:
            err_summary = f"Error in critic {critic_cnt}: {e}"
            append_execution_log(log_dir, f"{err_summary}\n{traceback.format_exc()}")

            if designer_response_parsed is not None:
                fallback_json_filename = os.path.join(log_dir, f"design{critic_cnt}_failed.json")
                json.dump(designer_response_parsed, open(fallback_json_filename, 'w'), indent=4)

            with open(os.path.join(log_dir, f"error_summary_{critic_cnt}.txt"), 'w') as fo:
                fo.write(err_summary + "\n")
                fo.write(traceback.format_exc())

            designer_prompt = (
                "The previous tool design or generated tool code failed. "
                f"Failure summary: {err_summary}. "
                "Please revise the tool plan, construction_steps, or placement_func and return corrected JSON only."
            )

            design_chat_history.append({
                'role': 'user',
                'content': designer_prompt
            })

            designer_response = designer.generate(
                prompt=designer_prompt,
                img=None,
                json_mode=False,
                chat_history=design_chat_history
            )
            continue

    planing_chat_history = [design_chat_history[0], design_chat_history[-1]]
    planing_prompt = open(os.path.join(project_path, 'utils', 'template_manipulate.txt'), 'r').read()
    planing_prompt = planing_prompt.replace("$3D_OBJECT_DESCRIPTION$", designer_prompt_json['3D_OBJECT_DESCRIPTION'])
    planing_prompt = planing_prompt.replace("$GOAL_DESCRIPTION$", designer_prompt_json['GOAL_DESCRIPTION'])
    planing_prompt = planing_prompt.replace("$3D_CONFIGURATION$", designer_prompt_json['3D_CONFIGURATION'])

    planing_response = designer.generate(prompt=planing_prompt, img=None, json_mode=False, chat_history=planing_chat_history)
    with open(os.path.join(log_dir, 'plan.txt'), 'w') as fo:
        fo.write(planing_response)

if __name__ == "__main__":

    run_tool_design(
        task_name=args.task_name,
        task_prompt_json_dir=args.task_prompt_json_dir,
        designer_source=args.designer_source,
        critic_source=args.critic_source,
        designer_lm_id=args.designer_lm_id,
        critic_lm_id=args.critic_lm_id,
        step_generator_source=args.step_generator_source,
        step_generator_lm_id=args.step_generator_lm_id
    )
