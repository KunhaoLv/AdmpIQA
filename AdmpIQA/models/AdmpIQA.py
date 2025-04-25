import torch
import torch.nn as nn
import os
import clip.clip as clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg['MODEL']['BACKBONE']['NAME']
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=os.path.expanduser("~/.cache/clip"))

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": 'MaPLe',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg['TRAINER']['COOP']['N_CTX']
    }

    model = clip.build_model_maple(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PromptLearnerTarget(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['TRAINER']['COOP']['N_CTX']
        ctx_init = cfg['TRAINER']['COOP']['CTX_INIT']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg['TRAINER']['COOP']['CSC']:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        self.vision_text_prompts_depth = 12
        self.text_depth = cfg['TRAINER']['Ours']['text_depth']
        self.vision_depth = cfg['TRAINER']['Ours']['vision_depth']
        self.proj = nn.Linear(768 * 2, 768)

        self.prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(n_ctx, ctx_dim)) for _ in range(self.vision_text_prompts_depth - 1)])
        self.prompts_vis = nn.ParameterList(
            [nn.Parameter(torch.empty(n_ctx, 768)) for _ in range(self.vision_text_prompts_depth)])
        for single_para in self.prompts_text + self.prompts_vis:
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(768 * 2, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.vision_text_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg['TRAINER']['COOP']['CLASS_TOKEN_POSITION']
        self.use_aux = cfg['TRAINER']['Ours']['use_aux']

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, share, vis):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if not self.use_aux:
            ttx = self.prompts_text if self.text_depth else []
            vtx = self.prompts_vis[1:] if self.vision_depth else []
            return prompts, self.prompts_vis[0], ttx, vtx

        s = self.prompts_vis[0]
        vtx = self.prompts_vis[1:]
        visual_deep_prompts = [layer(torch.cat((vtx[i], vis[i]), dim=1)) for i, layer in
                               enumerate(self.compound_prompt_projections)]
        vision_prompt = self.proj(torch.cat((s, share), dim=1))
        return prompts, vision_prompt, self.prompts_text, visual_deep_prompts


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(True),
            nn.Linear(1024, in_features)
        )
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return x + self.block(x)


class VisualFeatureAdapter(nn.Module):
    def __init__(self, in_features, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(in_features) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner_tar = PromptLearnerTarget(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.visual_feature_adapter = VisualFeatureAdapter(in_features=512,
                                                           num_blocks=cfg['TRAINER']['COOP']['N_BLOCKS'])

    def forward(self, image, wordid):
        logit_scale = self.logit_scale.exp()
        first_prompt_text, first_prompt_vision, deep_prompts_text, deep_prompts_vision = self.prompt_learner_tar(0, 0)
        text_features = self.text_encoder(first_prompt_text, self.prompt_learner_tar.tokenized_prompts,
                                          deep_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), first_prompt_vision, deep_prompts_vision)
        image_features = self.visual_feature_adapter(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        probabilities = logits.softmax(dim=1)
        weights = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2], device=self.device)
        score = (probabilities * weights).sum(dim=1, keepdim=True)
        return score, 0


class CoOp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.build_model()

    def build_model(self):
        cfg = self.cfg
