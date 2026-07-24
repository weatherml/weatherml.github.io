import argparse
import arxiv
import yaml
from datetime import datetime, timedelta
import re

GITHUB_URL_PATTERN = re.compile(r'https?://github\.com/[\w\-\.]+/[\w\-\.]+')

# Categories with keyword patterns for auto-categorization
CATEGORIES = {
    'Global Models': {
        'keywords': [
            'global weather', 'medium-range weather', 'medium-range forecast',
            'global forecast', 'neural weather prediction', 'ai weather',
            'weather prediction model', 'global weather model',
            'weather forecasting model', 'operational weather forecast',
            'machine-learned weather', 'machine learned weather',
            'autoregressive weather', 'global prediction model',
        ],
        'strong_keywords': [
            'fourcastnet', 'pangu-weather', 'pangu weather', 'graphcast',
            'gencast', 'fuxi', 'fuxiweather', 'climax', 'neuralgcm', 'aifs',
            'stormer', 'fengwu', 'skyai', 'aurora weather', 'weatherbench',
        ],
    },
    'Nowcasting': {
        'keywords': [
            'nowcasting', 'nowcast', 'precipitation nowcasting',
            'short-term precipitation', 'radar-based precipitation',
            'precipitation prediction', 'rain prediction',
        ],
        'strong_keywords': [
            'metnet', 'nowcastnet', 'dgmr', 'raindiff', 'stormdit',
        ],
    },
    'Downscaling': {
        'keywords': [
            'weather downscaling', 'climate downscaling', 'precipitation downscaling',
            'temperature downscaling', 'wind downscaling', 'meteorological downscaling',
            'statistical downscaling', 'dynamical downscaling', 'spatial downscaling',
            'super-resolution weather', 'super-resolution climate',
            'downscaling forecast', 'downscaling reanalysis', 'downscaling model',
            'downscaling land surface', 'atmospheric downscaling',
        ],
    },
    'Data Assimilation': {
        'keywords': [
            'data assimilation', '4dvar', '4d-var', 'kalman filter',
            'variational assimilation', 'observation operator',
            'state estimation', 'ensemble kalman',
        ],
    },
    'Ensembles': {
        'keywords': [
            'ensemble forecast', 'ensemble prediction', 'ensemble model',
            'probabilistic forecast', 'ensemble weather', 'ensemble spread',
            'post-processing ensemble', 'ensemble member',
        ],
    },
    'Climate Modeling': {
        'keywords': [
            'climate model', 'climate simulation', 'climate change',
            'climate projection', 'climate scenario', 'global climate',
            'climate emulator', 'earth system model', 'climate risk',
            'climate prediction', 'climate variability', 'climate forcing',
        ],
    },
    'Extreme Weather': {
        'keywords': [
            'extreme weather', 'tropical cyclone', 'hurricane prediction',
            'hurricane forecast', 'typhoon', 'severe storm', 'extreme precipitation',
            'flood prediction', 'flood forecast', 'heatwave', 'heat wave',
            'wildfire prediction', 'tornado',
        ],
    },
}

# arXiv API queries. Kept deliberately broad — relevance filtering happens
# in Python via is_weather_related() and is_ml_related(), so a paper only
# needs to land in one of these sweeps, not match an exact phrase.
SEARCH_QUERIES = [
    # Everything in (or cross-listed to) atmospheric/oceanic physics
    'cat:physics.ao-ph',
    # ML-venue papers with a weather/climate word in the title that were
    # not cross-listed to physics.ao-ph
    '(cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:stat.ML OR cat:eess.IV OR cat:eess.SP)'
    ' AND (ti:weather OR ti:climate OR ti:atmospheric OR ti:precipitation'
    ' OR ti:nowcasting OR ti:cyclone OR ti:hurricane OR ti:typhoon'
    ' OR ti:meteorological OR ti:rainfall OR ti:"data assimilation"'
    ' OR ti:reanalysis OR ti:downscaling)',
]

# Papers whose weather angle is incidental (e.g. perception in rain for
# self-driving cars) — excluded even if they match the weather terms
EXCLUDE_TERMS = [
    'autonomous driving', 'autonomous vehicle', 'self-driving',
    'driving scene', 'driver assistance',
]

# Terms that indicate a paper uses ML/AI methods (needed because the
# cat:physics.ao-ph sweep also returns pure physics/NWP papers)
ML_TERMS = [
    'machine learning', 'machine-learn', 'ml-based', 'ml-driven', ' ml ',
    'ml model', 'deep learning', 'neural network',
    'neural operator', 'data-driven', 'data driven', 'artificial intelligence',
    ' ai ', 'ai-based', 'ai-driven', 'ai weather', 'ai model', 'transformer',
    'attention mechanism', 'self-attention', 'diffusion model', 'generative model',
    'foundation model', 'graph neural', 'convolutional', 'u-net', 'unet',
    'autoencoder', ' gan ', 'adversarial network', 'lstm', 'recurrent neural',
    'reinforcement learning', 'self-supervised', 'supervised learning',
    'pretrain', 'pre-train', 'fine-tun', 'emulator', 'surrogate model',
    'end-to-end learn', 'learned model', 'gaussian process', 'random forest',
    'gradient boosting', 'deep generative', 'flow matching', 'score-based',
]

# Terms that indicate a paper is about weather/climate/atmosphere
WEATHER_TERMS = [
    'weather', 'climate', 'meteorolog', 'atmospher', 'precipitation',
    'rainfall', 'temperature forecast', 'wind forecast', 'wind speed',
    'storm', 'cyclone', 'hurricane', 'typhoon', 'monsoon', 'drought',
    'flood', 'ocean', 'sea surface', 'nwp', 'numerical weather',
    'reanalysis', 'era5', 'ecmwf', 'noaa', 'gfs', 'cmip', 'cordex',
    'geopotential', 'convection', 'mesoscale', 'synoptic', 'troposphere',
    'stratosphere', 'boundary layer', 'land surface temperature',
    'soil moisture', 'snow cover', 'el nino', 'la nina', 'enso',
    'jet stream', 'pressure field', 'isobar', 'forecast lead time',
    'medium-range', 'subseasonal', 'seasonal forecast', 'barotropic',
    'navier-stokes', 'geophysic', 'earth system', 'planetary boundary',
    'radar observation', 'satellite observation', 'radiosonde',
    'global model', 'forecast skill', 'forecast accuracy',
]

# ML method tags to auto-extract
METHOD_TAGS = {
    'transformer': ['transformer', 'attention mechanism', 'self-attention', 'cross-attention'],
    'diffusion': ['diffusion model', 'denoising diffusion', 'score-based', 'ddpm', 'flow matching'],
    'GAN': ['generative adversarial', ' gan ', 'adversarial network'],
    'CNN': ['convolutional neural', ' cnn ', 'u-net', 'unet', 'resnet'],
    'GNN': ['graph neural', 'graph network', 'message passing'],
    'physics-informed': ['physics-informed', 'physics informed', 'physics-based', 'physics based'],
    'reinforcement-learning': ['reinforcement learning'],
    'variational': ['variational inference', 'variational autoencoder', ' vae '],
    'foundation-model': ['foundation model', 'large-scale pretrain', 'pre-trained'],
    'operator-learning': ['neural operator', 'fourier neural operator', 'deeponet'],
    'recurrent': ['lstm', 'recurrent neural', ' rnn ', ' gru '],
    'probabilistic': ['probabilistic', 'uncertainty quantification', 'bayesian'],
}


def is_weather_related(title, abstract):
    """Check if a paper is related to weather/climate/atmospheric science."""
    text = f"{title} {abstract}".lower()
    if any(term in text for term in EXCLUDE_TERMS):
        return False
    return any(term in text for term in WEATHER_TERMS)


def is_ml_related(title, abstract):
    """Check if a paper uses ML/AI methods."""
    text = f" {title} {abstract} ".lower()
    return any(term in text for term in ML_TERMS)


def categorize_paper(title, abstract):
    """Auto-categorize a paper based on title and abstract keywords."""
    text = f"{title} {abstract}".lower()

    scores = {}
    for category, config in CATEGORIES.items():
        score = 0
        for kw in config.get('keywords', []):
            if kw in text:
                score += 1
        for kw in config.get('strong_keywords', []):
            if kw in text:
                score += 5
        scores[category] = score

    best_category = max(scores, key=scores.get)
    if scores[best_category] > 0:
        return best_category
    return 'Other'


def extract_tags(title, abstract, arxiv_categories=None):
    """Extract method tags from paper title and abstract."""
    text = f" {title} {abstract} ".lower()
    tags = []

    for tag, keywords in METHOD_TAGS.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)

    if arxiv_categories:
        for cat in arxiv_categories:
            tags.append(cat)

    return tags


def make_paper_entry(result):
    """Build a papers.yml entry from an arxiv.Result."""
    title = result.title
    abstract = result.summary.replace('\n', ' ')

    arxiv_cats = [result.primary_category] + [
        c for c in result.categories if c != result.primary_category
    ]

    # Extract GitHub URL from abstract or comments
    github_url = None
    comments = result.comment or ''
    for text in [abstract, comments]:
        match = GITHUB_URL_PATTERN.search(text)
        if match:
            github_url = match.group(0).rstrip('.')
            break

    paper = {
        'category': categorize_paper(title, abstract),
        'title': title,
        'authors': ', '.join(author.name for author in result.authors),
        'year': result.published.year,
        'arxiv': result.entry_id.split('/')[-1],
        'abstract': abstract,
        'tags': extract_tags(title, abstract, arxiv_cats),
        'arxiv_categories': arxiv_cats,
    }
    if github_url:
        paper['github'] = github_url
    return paper


def save_papers(papers):
    with open('papers.yml', 'w') as f:
        yaml.dump(papers, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


def base_arxiv_id(arxiv_id):
    """Strip the version suffix from an arXiv ID."""
    return re.sub(r'v\d+$', '', arxiv_id)


def find_new_papers(lookback_days=14, max_results=300, dry_run=False):
    """Find new papers from arXiv and add them to papers.yml."""
    with open('papers.yml', 'r') as f:
        existing_papers = yaml.safe_load(f) or []

    existing_arxiv_ids = {base_arxiv_id(p['arxiv']) for p in existing_papers}

    client = arxiv.Client(
        page_size=100,
        delay_seconds=5.0,
        num_retries=5,
    )

    cutoff = datetime.now().astimezone() - timedelta(days=lookback_days)
    new_papers = []
    seen_this_run = set()
    skipped = 0

    for query in SEARCH_QUERIES:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for result in client.results(search):
            arxiv_id = base_arxiv_id(result.entry_id.split('/')[-1])

            if result.published < cutoff:
                continue
            if arxiv_id in existing_arxiv_ids or arxiv_id in seen_this_run:
                continue

            title = result.title
            abstract = result.summary.replace('\n', ' ')

            # Filter out non-weather and non-ML papers
            if not (is_weather_related(title, abstract)
                    and is_ml_related(title, abstract)):
                skipped += 1
                continue

            seen_this_run.add(arxiv_id)
            new_papers.append(make_paper_entry(result))

    if new_papers:
        print(f"Found {len(new_papers)} new papers ({skipped} skipped as not relevant).")
        for p in new_papers:
            print(f"  [{p['category']}] {p['arxiv']}: {p['title'][:80]}")
        if not dry_run:
            save_papers(existing_papers + new_papers)
    else:
        print(f"No new papers found ({skipped} skipped as not relevant).")
    if dry_run:
        print("Dry run: papers.yml not modified.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find new weather-ML papers on arXiv.')
    parser.add_argument('--lookback-days', type=int, default=14)
    parser.add_argument('--max-results', type=int, default=300,
                        help='max results fetched per search query')
    parser.add_argument('--dry-run', action='store_true',
                        help='print findings without modifying papers.yml')
    args = parser.parse_args()
    find_new_papers(args.lookback_days, args.max_results, args.dry_run)
