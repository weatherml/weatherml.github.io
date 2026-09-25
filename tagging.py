"""Rule-based tags for papers, derived from the title and abstract.

Tags are computed at build time (not stored in papers.yml), so improving a
rule here re-tags the whole collection on the next build.

Each tag has a list of regex patterns (matched case-insensitively on word
boundaries you write into the pattern) and a `need` threshold:

- a match in the title always counts as enough;
- otherwise the tag needs `need` matches in the abstract, ignoring sentences
  that only compare against other work ("outperforms transformer baselines").

Resolution and time-step tags are handled separately because they depend on
numbers in the text ("3 km", "$0.25^\\circ$", "6-hourly").
"""

import re

GROUPS = ['Method', 'Theme', 'Domain', 'Resolution', 'Time step']

# (name, group, patterns, need)
TAGS = [
    # Method
    ('Diffusion & flow matching', 'Method', [
        r'diffusion models?', r'denoising diffusion', r'score-based', r'\bddpms?\b',
        r'flow matching', r'rectified flow', r'latent diffusion', r'diffusion-based',
        r'generative diffusion', r'consistency models?',
    ], 1),
    ('GANs', 'Method', [
        r'generative adversarial', r'\bgans?\b', r'\b\w+-gans?\b',
        r'\b(?:wgan|cyclegan|stylegan|pix2pix|srgan|esrgan)\b',
    ], 1),
    ('Transformers', 'Method', [
        r'\btransformers?\b', r'transformer-based', r'vision transformer', r'\bvits?\b',
        r'self-attention', r'\bswin\b',
    ], 1),
    ('Graph neural networks', 'Method', [
        r'graph neural', r'\bgnns?\b', r'graph networks?', r'message[- ]passing',
        r'graph-based (?:model|network|neural)',
    ], 1),
    ('Neural operators', 'Method', [
        r'neural operators?', r'fourier neural operator', r'\bfnos?\b', r'\bsfnos?\b',
        r'deeponet', r'operator learning',
    ], 1),
    ('CNN / U-Net', 'Method', [
        r'convolutional neural', r'\bcnns?\b', r'\bu-?nets?\b', r'\bresnets?\b',
        r'convolutional (?:network|layers?|architecture|model)',
    ], 1),
    ('Recurrent networks', 'Method', [
        r'\blstms?\b', r'recurrent neural', r'\bgrus?\b', r'convlstm', r'\brnns?\b',
    ], 1),
    ('Foundation models', 'Method', [
        r'foundation models?', r'pre-?trained on', r'large-scale pre-?training',
    ], 1),
    ('Physics–ML hybrid', 'Method', [
        r'physics[- ]informed', r'\bpinns?\b', r'physics-constrained', r'physics-guided',
        r'physics-embedded', r'hybrid (?:physics|physical|model|modell?ing|approach|framework)',
        r'differentiable (?:model|physics|solver|dynamical core|gcm)', r'neuralgcm',
        r'conservation (?:law|constraint)s?', r'hard constraints?',
    ], 1),
    ('LLMs & agents', 'Method', [
        r'large language models?', r'\bllms?\b', r'vision[- ]language', r'\bvlms?\b',
        r'multimodal large', r'\bagentic\b', r'\bgpt-?\d',
    ], 1),
    ('Classical ML', 'Method', [
        r'random forests?', r'gradient[- ]boost', r'xgboost', r'lightgbm', r'catboost',
        r'support vector', r'\bsvms?\b', r'decision trees?', r'gaussian process',
    ], 1),
    ('Reinforcement learning', 'Method', [
        r'reinforcement learning',
    ], 1),

    # Theme
    ('Precipitation', 'Theme', [
        r'precipitation', r'rainfall', r'\brain\b', r'radar reflectivity', r'\bqpf\b',
    ], 2),
    ('Tropical cyclones', 'Theme', [
        r'tropical cyclones?', r'hurricanes?', r'typhoons?',
    ], 1),
    ('Extremes', 'Theme', [
        r'extreme (?:weather|events?|precipitation|rainfall|heat|temperatures?|winds?)',
        r'heat ?waves?', r'droughts?', r'floods?', r'flooding', r'wildfires?',
        r'cold (?:spells?|waves?)', r'severe (?:weather|storms?|convection)',
    ], 2),
    ('Subseasonal to seasonal', 'Theme', [
        r'sub-?seasonal', r'\bs2s\b', r'seasonal (?:forecast|prediction)', r'\benso\b',
        r'el ni[ñn]o', r'madden[- ]julian', r'\bmjo\b',
    ], 1),
    ('Uncertainty & ensembles', 'Theme', [
        r'probabilistic', r'uncertainty quantification', r'\bensembles?\b', r'\bcrps\b',
        r'calibrat\w*',
    ], 2),
    ('Benchmarks & datasets', 'Theme', [
        # Only when the paper itself introduces one
        r'(?:introduce|present|release|propose|provide|curate|construct|build)\w*\s+'
        r'(?:[\w-]+\s+){0,6}(?:dataset|benchmark|data set)s?\b',
    ], 1),
    ('Evaluation', 'Theme', [
        r'intercomparison', r'verification', r'\bevaluat\w+ (?:of )?(?:the )?'
        r'(?:ai|ml|machine|data-driven|deep|neural)', r'forecast skill',
        r'how (?:well|good|reliable)',
    ], 2),
    ('Interpretability', 'Theme', [
        r'interpretab\w+', r'explainab\w+', r'\bxai\b', r'saliency', r'\bshap\b',
        r'mechanistic interpretab\w*',
    ], 2),
    ('Efficiency', 'Theme', [
        r'computational(?:ly)? (?:cost|efficien\w+|expensive|burden|demand)',
        r'speed-?ups?', r'orders? of magnitude faster', r'\d+\s*(?:x|×|times) faster',
        r'lightweight', r'fraction of the (?:cost|compute)', r'\bgpu-hours?\b',
    ], 2),
    ('Energy', 'Theme', [
        r'wind (?:power|energy|farms?|turbines?)', r'solar (?:power|energy|irradiance)',
        r'photovoltaic', r'renewable', r'energy (?:demand|load)', r'\bload forecast\w*',
    ], 1),

    # Domain
    ('Global', 'Domain', [
        r'\bglobal (?:weather|forecast\w*|models?|medium|scale|domain|prediction|'
        r'atmospher\w*|ocean\w*|climate|coverage|grid|emulat\w+)',
        r'globally', r'\bworldwide\b',
    ], 1),
    ('Regional', 'Domain', [
        r'\bregional (?:weather|climate|models?|modell?ing|forecast\w*|domains?|scales?|'
        r'downscaling|predictions?|emulat\w+|nwp|reanalys\w+)', r'limited[- ]area', r'\blams?\b', r'\bconus\b', r'continental',
        r'\bover (?:the )?(?:europe|china|india|africa|australia|japan|korea|'
        r'(?:the )?(?:united states|us|u\.s\.)|north america|south america|'
        r'the mediterranean|the alps|germany|france|italy|spain|norway|sweden|'
        r'finland|switzerland|the uk|brazil|canada|mexico|iran|pakistan|'
        r'bangladesh|east asia|southeast asia|the tibetan plateau)\b',
    ], 1),
    ('Station / point', 'Domain', [
        r'weather stations?', r'station[- ](?:data|observations?|measurements?|level|based)',
        r'point forecasts?', r'site-specific', r'\bin[- ]situ\b', r'\bstations\b',
    ], 2),
]

# Sentences with these words are mostly about other work, so method/theme
# mentions inside them don't count
COMPARISON = re.compile(
    r'\b(?:outperform\w*|compared (?:to|with)|comparison|unlike|than|baselines?|'
    r'such as|e\.g\.|existing|previous(?:ly)?|prior|state-of-the-art|sota|'
    r'traditional|conventional|competitive with|rival\w*|surpass\w*|'
    r'recent(?:ly)?|widely|ha(?:ve|s) (?:shown|emerged|become|achieved|gained))\b'
)

# ---- Resolution -------------------------------------------------------------

# "3 km", "3-km", "3km", "3 kilometres"
KM = re.compile(r'(\d+(?:\.\d+)?)\s*-?\s*(?:km|kilomet(?:er|re)s?)\b')
# "0.25°", "0.25 degree", "$0.25^\circ$", "0.25-deg" — but not temperatures
DEG = re.compile(
    r'(\d+(?:\.\d+)?)\s*-?\s*(?:°|\$?\^\{?\\circ\}?\$?|deg(?:ree)?s?\b)(?!\s*[cfknsew]\b)'
)
# A number only counts as resolution when words like these are close by, so
# distances ("eddies below 10 km") and latitudes ("60°N") are ignored
RESOLUTION_CONTEXT = re.compile(r'resolution|resolv|grid|spacing|mesh|horizontal')
KM_SCALE_WORDS = re.compile(
    r'kilomet(?:er|re)-scale|\bkm-scale|convection-(?:permitting|resolving)|'
    r'storm-resolving|hectometric|sub-kilomet'
)

# ---- Time step --------------------------------------------------------------

TIME_STEPS = [
    ('Sub-hourly', [
        r'sub-?hourly', r'\b\d+[- ]?min(?:ute)?s?\b(?! (?:of|to) )', r'minute-level',
    ]),
    ('Hourly', [
        r'(?<![\d-])\bhourly\b', r'\b(?:1|one)[- ]?h(?:ou)?r?\s+'
        r'(?:resolution|time[- ]?steps?|intervals?|increments?)', r'hour-level',
    ]),
    ('6-hourly', [
        r'\b(?:6|six)[- ]?hourly', r'\b6[- ]?h(?:ou)?r?\s+'
        r'(?:resolution|time[- ]?steps?|intervals?|increments?)',
    ]),
    ('Daily', [r'\bdaily\b']),
    ('Monthly', [r'\bmonthly\b']),
]

TAG_ORDER = [name for name, *_ in TAGS] + [
    'Km-scale', '0.25°', 'Coarse (≥1°)',
] + [name for name, _ in TIME_STEPS]

TAG_GROUP = {name: group for name, group, *_ in TAGS}
TAG_GROUP.update({'Km-scale': 'Resolution', '0.25°': 'Resolution',
                  'Coarse (≥1°)': 'Resolution'})
TAG_GROUP.update({name: 'Time step' for name, _ in TIME_STEPS})

_COMPILED = [(name, [re.compile(p) for p in patterns], need)
             for name, _, patterns, need in TAGS]
_TIME_COMPILED = [(name, [re.compile(p) for p in patterns])
                  for name, patterns in TIME_STEPS]


def _sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def _resolution_values(pattern, text):
    for m in pattern.finditer(text):
        window = text[max(0, m.start() - 50):m.end() + 30]
        if RESOLUTION_CONTEXT.search(window):
            yield float(m.group(1))


def resolution_tags(text):
    tags = set()
    if KM_SCALE_WORDS.search(text):
        tags.add('Km-scale')
    for km in _resolution_values(KM, text):
        if 0 < km <= 5:
            tags.add('Km-scale')
        elif 20 <= km <= 35:
            tags.add('0.25°')
    for deg in _resolution_values(DEG, text):
        if 0.2 <= deg <= 0.3:
            tags.add('0.25°')
        elif 1 <= deg <= 6:
            tags.add('Coarse (≥1°)')
    return tags


def tag_paper(paper):
    """Return the paper's tags in display order."""
    title = paper['title'].lower()
    abstract = paper.get('abstract', '').replace('\n', ' ').lower()
    own_sentences = [s for s in _sentences(abstract) if not COMPARISON.search(s)]

    tags = set()
    for name, patterns, need in _COMPILED:
        if any(p.search(title) for p in patterns):
            tags.add(name)
            continue
        hits = sum(len(p.findall(s)) for s in own_sentences for p in patterns)
        if hits >= need:
            tags.add(name)

    text = f"{title} {abstract}"
    tags |= resolution_tags(text)
    for name, patterns in _TIME_COMPILED:
        if any(p.search(text) for p in patterns):
            tags.add(name)

    return [name for name in TAG_ORDER if name in tags]
