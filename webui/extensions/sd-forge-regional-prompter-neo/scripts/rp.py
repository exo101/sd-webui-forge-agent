import os.path, json
import gradio as gr
import numpy as np
from PIL import Image
import modules
from modules import paths, scripts, shared, extra_networks, prompt_parser
from modules.processing import Processed
from modules.script_callbacks import on_cfg_denoised, on_cfg_denoiser
import scripts.attention
import scripts.latent
import scripts.regions
from scripts.attention import TOKENS, hook_forwards
from scripts.latent import denoised_callback_s, denoiser_callback_s, unloadlorafowards
from scripts.regions import KEYBRK, ALLKEYS
from io import BytesIO
import base64
import torch
from modules import devices

PTPRESET = modules.scripts.basedir()
ATTNSCALE = 8

forge = True  # Forge Neo always

def lange(l):
    return range(len(l))

KEYBRK_R = "BREAK"  # Alternative BREAK key for internal processing
orig_batch_cond_uncond = shared.opts.batch_cond_uncond if hasattr(shared.opts, "batch_cond_uncond") else shared.batch_cond_uncond

class Script(modules.scripts.Script):
    def __init__(self, active=False, mode="Visual", calc="Attention", h=0, w=0, debug=False, usebase=False, 
                 usecom=False, usencom=False, batch=1, isxl=False, lstop=0, lstop_hr=0, diff=None):
        self.active = active
        self.mode = mode
        self.calc = calc
        self.h = h
        self.w = w
        self.debug = debug
        self.usebase = usebase
        self.usecom = usecom
        self.usencom = usencom
        self.batch_size = batch
        self.isxl = isxl
        self.pt = []
        self.nt = []
        self.aratios = []
        self.bratios = []
        self.divide = 0
        self.count = 0
        self.eq = True
        self.pn = True
        self.hr = False
        self.hr_scale = 0
        self.hr_w = 0
        self.hr_h = 0
        self.in_hr = False
        self.xsize = 0
        self.imgcount = 0
        self.filters = []
        self.lora_applied = False
        self.lstop = int(lstop)
        self.lstop_hr = int(lstop_hr)
        self.regmasks = None
        self.regbase = None
        self.pe = []
        self.step = 0
        self.diff = diff
        self.rps = None
        self.hooked = False
        self.condi = 0
        self.ppt = []
        self.pnt = []
        self.used_prompt = ""
        self.logprops = ["active","mode","usebase","usecom","usencom","batch_size","isxl","h","w","aratios",
                        "divide","count","eq","pn","hr","pe","step","diff","used_prompt"]
        self.log = {}
        # Latent mode attributes
        self.isbefore15 = False
        self.layer_name = "lora_layer_name"
        self.ui_version = 0
        self.slowlora = False
        self.x = None
        self.pfirst = False
        self.th = []
        self.ex = False
        self.rebacked = False

    def title(self):
        return "Regional Prompter"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):
        eladd = "i2i" if is_img2img else "t2i"
        with gr.Accordion("场景编辑器 (Regional Prompter)", open=False, elem_id="RP_main" + eladd):
            with gr.Row():
                active = gr.Checkbox(value=False, label="启用", interactive=True, elem_id="RP_active" + eladd)
            with gr.Row():
                calcmode = gr.Radio(label="生成模式", choices=["Attention", "Latent"], value="Attention", type="value", interactive=True, elem_id="RP_generation_mode" + eladd)
            # Visual editor
            ve_html = open(os.path.join(PTPRESET, "visual_editor.html"), encoding="utf-8").read()
            ve_html = ve_html.replace("VE_ELADD", eladd)
            visual_html = gr.HTML(value=ve_html)
            visual_mask = gr.Textbox(visible=True, elem_id="ve-mask-output-" + eladd, elem_classes="ve-hidden-output")
            visual_prompts = gr.Textbox(visible=True, elem_id="ve-prompts-output-" + eladd, elem_classes="ve-hidden-output")
        return [active, calcmode, visual_mask, visual_prompts]

    def process(self, p, active, calcmode, visual_mask="", visual_prompts=""):
        print(f"[Visual Editor] active={active}, calcmode={calcmode}, mask_len={len(visual_mask) if visual_mask else 0}, prompts_len={len(visual_prompts) if visual_prompts else 0}")
        if not active:
            return unloader(self, p)

        self.__init__(active, "Visual-Mask", calcmode, p.height, p.width)

        self.all_prompts = p.all_prompts.copy()
        self.all_negative_prompts = p.all_negative_prompts.copy()

        self.isvanilla = p.sampler_name in ["DDIM", "PLMS", "UniPC"]
        if forge: self.isvanilla = not self.isvanilla

        if self.h % ATTNSCALE != 0 or self.w % ATTNSCALE != 0:
            self.h = self.h - self.h % ATTNSCALE
            self.w = self.w - self.w % ATTNSCALE

        if hasattr(p, "enable_hr"):
            self.hr = p.enable_hr
            self.hr_w = (p.hr_resize_x if p.hr_resize_x > p.width else p.width * p.hr_scale)
            self.hr_h = (p.hr_resize_y if p.hr_resize_y > p.height else p.height * p.hr_scale)
            if self.hr_h % ATTNSCALE != 0 or self.hr_w % ATTNSCALE != 0:
                self.hr_h = self.hr_h - self.hr_h % ATTNSCALE
                self.hr_w = self.hr_w - self.hr_w % ATTNSCALE

        if visual_mask:
            print(f"[Visual Editor] Processing mask with {len(visual_prompts)} prompts")
            keyreplacer(self, p)
            self._process_visual_mask(visual_mask, visual_prompts, p)

            if "Att" in calcmode:
                self.handle = hook_forwards(self, p.sd_model.forge_objects.unet.model)
                if hasattr(shared.opts, "batch_cond_uncond"):
                    shared.opts.batch_cond_uncond = orig_batch_cond_uncond
                else:
                    shared.batch_cond_uncond = orig_batch_cond_uncond
                unloadlorafowards(p)
            else:
                self.handle = hook_forwards(self, p.sd_model.forge_objects.unet.model, remove=True)
                from scripts.latent import setuploras
                setuploras(self)
        else:
            print(f"[Visual Editor] WARNING: visual_mask is empty, skipping region processing")

        keyreplacer(self, p)
        blankdealer(self, p)
        commondealer(p, self.usecom, self.usencom)
        if "La" in self.calc:
            allchanger(p, KEYBRK, "AND")
        if tokendealer(self, p):
            return unloader(self, p)

        if not self.diff: hrdealer(p)

        print(f"Regional Prompter Active, Pos tokens: {self.ppt}, Neg tokens: {self.pnt}")
        self.used_prompt = p.all_prompts[0]

    def _process_visual_mask(self, visual_mask, visual_prompts, p):
        """Process visual editor mask image and prompts into region masks."""
        try:
            self.usebase = True
            if visual_mask.startswith("data:image"):
                visual_mask = visual_mask.split(",", 1)[1]
            img_data = base64.b64decode(visual_mask)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img_np = np.array(img)

            prompt_data = json.loads(visual_prompts) if visual_prompts else []

            # Build color-to-prompt mapping from JS data (preserves element order)
            color_map = {}
            for item in prompt_data:
                if isinstance(item, dict) and 'color' in item:
                    c = tuple(item['color'])
                    color_map[c] = item

            mask_flat = img_np.reshape(-1, 3)
            white_mask = (mask_flat == [255, 255, 255]).all(axis=1)
            unique_colors = np.unique(mask_flat[~white_mask], axis=0)

            # Build regions preserving JS element order via color_map
            regions = []
            for item in prompt_data:
                if not isinstance(item, dict) or 'color' not in item:
                    continue
                c = tuple(item['color'])
                # Find this color in unique_colors (should exist if element was placed)
                color_arr = np.array([c], dtype=np.uint8)
                matches = (unique_colors == color_arr).all(axis=1)
                if not matches.any():
                    continue
                regions.append({
                    "color": color_arr[0],
                    "prompt": item.get("prompt", ""),
                    "weight": item.get("weight", 1.0)
                })
            # Fallback: if no color_map matched, use legacy unique_colors order
            if not regions:
                for i, color in enumerate(unique_colors):
                    region_prompt = prompt_data[i].get("prompt", "") if i < len(prompt_data) and isinstance(prompt_data[i], dict) else ""
                    region_weight = prompt_data[i].get("weight", 1.0) if i < len(prompt_data) and isinstance(prompt_data[i], dict) else 1.0
                    regions.append({"color": color, "prompt": region_prompt, "weight": region_weight})

            print(f"[Visual Editor] Regions: {len(regions)}, prompts: {[r['prompt'][:20] for r in regions]}")
            print(f"[Visual Editor] Mask size: {img_np.shape}, bratios: {[f'{max(0.0, 1.0 - w):.1f}' for w in [r['weight'] for r in regions]]}")
            prompt_parts = [r["prompt"] for r in regions]
            for i in range(len(p.all_prompts)):
                p.all_prompts[i] = p.all_prompts[i] + " " + KEYBRK.join(prompt_parts)

            h, w = img_np.shape[:2]
            self.regmasks = []
            tm = np.zeros((h, w), dtype=np.float32)

            for i, r in enumerate(regions):
                color = r["color"].reshape(1, 1, 3)
                m = ((img_np == color).all(axis=2)).astype(np.float32)
                # Debug: find region bounding box
                ys, xs = np.where(m > 0)
                if len(xs) > 0:
                    print(f"[VE] Region {i}: color={r['color']}, prompt='{r['prompt'][:20]}', " +
                          f"bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()}), " +
                          f"center=({(xs.min()+xs.max())//2},{(ys.min()+ys.max())//2})")
                else:
                    print(f"[VE] Region {i}: color={r['color']}, NO PIXELS FOUND in mask!")
                tm = tm + m
                m_tensor = torch.from_numpy(m.reshape(1, 1, h, w)).to(devices.device)
                self.regmasks.append(m_tensor)

            base_mask = 1 - np.clip(tm, 0, 1)
            self.regbase = torch.from_numpy(base_mask.reshape(1, 1, h, w)).to(devices.device)

            self.bratios = [[max(0.0, 1.0 - r["weight"]) for r in regions]]

            if self.debug:
                print(f"Visual mode: {len(regions)} regions detected")
                for i, r in enumerate(regions):
                    print(f"  Region {i}: color={r['color']}, weight={r['weight']}, prompt='{r['prompt'][:30]}'")

        except Exception as e:
            print(f"Error processing visual mask: {e}")
            import traceback
            traceback.print_exc()

    def before_process_batch(self, p, *args, **kwargs):
        if self.active:
            self.current_prompts = kwargs["prompts"].copy()
            p.disable_extra_networks = False

    def before_hr(self, p, active, calcmode, visual_mask="", visual_prompts=""):
        if self.active:
            self.in_hr = True
            if "La" in self.calc:
                pass

    def process_batch(self, p, active, calcmode, visual_mask="", visual_prompts="", **kwargs):
        if self.active:
            resetpcache(p)
            self.in_hr = False
            self.xsize = 0
            if not hasattr(self, "current_prompts"):
                self.current_prompts = kwargs["prompts"].copy()
            p.all_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size] = self.all_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
            p.all_negative_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size] = self.all_negative_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
            if "La" in self.calc:
                if not self.lora_applied:
                    denoiserdealer(self)
                    self.lora_applied = True

    def postprocess(self, p, processed, *args):
        if self.active:
            with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                processedx = Processed(p, [], p.seed, "")
                file.write(processedx.infotext(p, 0))
        unloader(self, p)

    def denoiser_callback(self, params):
        denoiser_callback_s(self, params)

    def denoised_callback(self, params):
        denoised_callback_s(self, params)


def unloader(self, p):
    if hasattr(self, "handle"):
        hook_forwards(self, p.sd_model.forge_objects.unet.model, remove=True)
        del self.handle
    self.__init__()
    if hasattr(shared.opts, "batch_cond_uncond"):
        shared.opts.batch_cond_uncond = orig_batch_cond_uncond
    else:
        shared.batch_cond_uncond = orig_batch_cond_uncond
    unloadlorafowards(p)


def denoiserdealer(self):
    if self.calc == "Latent":
        if not hasattr(self, "dd_callbacks"):
            self.dd_callbacks = on_cfg_denoised(self.denoised_callback)
        if hasattr(shared.opts, "batch_cond_uncond"):
            shared.opts.batch_cond_uncond = False
        else:
            shared.batch_cond_uncond = False
    if not hasattr(self, "dr_callbacks"):
        self.dr_callbacks = on_cfg_denoiser(self.denoiser_callback)
    if self.diff:
        if not hasattr(self, "dd_callbacks"):
            self.dd_callbacks = on_cfg_denoised(self.denoised_callback)


# ----- Prompt processing helpers -----
def blankdealer(self, p):
    seps = "AND" if "La" in self.calc else KEYBRK
    all_prompts = []
    for prompt in p.all_prompts:
        regions = prompt.split(seps)
        if regions[-1].strip() in ["", ","]:
            prompt = prompt + " _"
        all_prompts.append(prompt)
    p.all_prompts = all_prompts


def commondealer(p, usecom, usencom):
    def comadder(prompt):
        ppl = prompt.split(KEYBRK)
        for i in range(len(ppl)):
            if i == 0: continue
            ppl[i] = ppl[0] + ", " + ppl[i]
        ppl = ppl[1:]
        return f"{KEYBRK} ".join(ppl)

    if usecom:
        all_prompts = [comadder(pr) for pr in p.all_prompts]
        p.all_prompts = all_prompts
        p.prompt = all_prompts[0]
    if usencom:
        all_neg = [comadder(pr) for pr in p.all_negative_prompts]
        p.all_negative_prompts = all_neg
        p.negative_prompt = all_neg[0]


def hrdealer(p):
    p.hr_prompt = p.prompt
    p.hr_negative_prompt = p.negative_prompt
    p.all_hr_prompts = p.all_prompts
    p.all_hr_negative_prompts = p.all_negative_prompts


def allchanger(p, a, b):
    p.prompt = p.prompt.replace(a, b)
    for i in lange(p.all_prompts):
        p.all_prompts[i] = p.all_prompts[i].replace(a, b)
    p.negative_prompt = p.negative_prompt.replace(a, b)
    for i in lange(p.all_negative_prompts):
        p.all_negative_prompts[i] = p.all_negative_prompts[i].replace(a, b)


def tokendealer(self, p):
    seps = "AND" if "La" in self.calc else KEYBRK
    self.seps = seps

    text, _ = extra_networks.parse_prompt(p.all_prompts[0])
    text = prompt_parser.get_learned_conditioning_prompt_schedules([text], p.steps)[0][0][1]
    ppl = text.split(seps)
    ppl = [p.replace(KEYBRK_R, KEYBRK) for p in ppl]

    ntext, _ = extra_networks.parse_prompt(p.all_negative_prompts[0])
    npl = ntext.split(seps)
    npl = [p.replace(KEYBRK_R, KEYBRK) for p in npl]
    eqb = len(ppl) == len(npl)
    pt, nt, ppt, pnt = [], [], [], []
    padd = 0

    try:
        if hasattr(p.sd_model, 'text_processing_engine_g'):
            tokenizer = p.sd_model.text_processing_engine_g
        elif hasattr(p.sd_model, 'conditioner'):
            tokenizer = p.sd_model.conditioner.embedders[0]
        else:
            for pp in ppl:
                pt.append([padd, padd + 1])
                ppt.append(75)
                padd = padd + 1
            padd = 0
            for np in npl:
                nt.append([padd, padd + 1])
                pnt.append(75)
                padd = padd + 1
            self.pt = pt
            self.nt = nt
            self.ppt = ppt
            self.pnt = pnt
            self.eq = padd == padd and eqb
            return False

        for pp in ppl:
            if hasattr(tokenizer, 'tokenize_line'):
                tokens, tokensnum = tokenizer.tokenize_line(pp)
            else:
                tokens = tokenizer.encode(pp)
                tokensnum = len(tokens[0]) if isinstance(tokens, list) else len(tokens)
            pt.append([padd, tokensnum // TOKENS + 1 + padd])
            ppt.append(tokensnum)
            padd = tokensnum // TOKENS + 1 + padd

        paddp = padd
        padd = 0
        for np in npl:
            if hasattr(tokenizer, 'tokenize_line'):
                _, tokensnum = tokenizer.tokenize_line(np)
            else:
                tokens = tokenizer.encode(np)
                tokensnum = len(tokens[0]) if isinstance(tokens, list) else len(tokens)
            nt.append([padd, tokensnum // TOKENS + 1 + padd])
            pnt.append(tokensnum)
            padd = tokensnum // TOKENS + 1 + padd

        self.pt = pt
        self.nt = nt
        self.ppt = ppt
        self.pnt = pnt
        self.eq = paddp == padd and eqb

        if not self.pt or not self.nt:
            return True
        return False
    except Exception:
        return True


def keyreplacer(self, p):
    for key in ALLKEYS:
        for i in lange(p.all_prompts):
            p.all_prompts[i] = p.all_prompts[i].replace(key, KEYBRK)
        for i in lange(p.all_negative_prompts):
            p.all_negative_prompts[i] = p.all_negative_prompts[i].replace(key, KEYBRK)


def resetpcache(p):
    p.cached_c = [None, None]
    p.cached_uc = [None, None]
    p.cached_hr_c = [None, None]
    p.cached_hr_uc = [None, None]