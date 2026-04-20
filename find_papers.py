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

# Search keywords for arXiv queries
SEARCH_KEYWORDS = [
    'weather forecasting deep learning',
    'climate model machine learning',
    'precipitation nowcasting',
    'neural weather prediction',
    'data assimilation machine learning',
    'ensemble weather forecasting',
    'climate downscaling',
    'numerical weather prediction deep learning',
    'extreme weather prediction',
    'atmospheric prediction neural',
    'weather model transformer',
    'storm prediction deep learning',
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
    return any(term in text for term in WEATHER_TERMS)


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


def find_new_papers(lookback_days=14):
    """Find new papers from arXiv and add them to papers.yml."""
    with open('papers.yml', 'r') as f:
        existing_papers = yaml.safe_load(f) or []

    existing_arxiv_ids = {p['arxiv'] for p in existing_papers}

    # Build search query - search both title and abstract
    queries = []
    for kw in SEARCH_KEYWORDS:
        queries.append(f'ti:"{kw}"')
        queries.append(f'abs:"{kw}"')
    search_query = " OR ".join(queries)

    client = arxiv.Client(
        page_size=50,
        delay_seconds=5.0,
        num_retries=5,
    )
    search = arxiv.Search(
        query=search_query,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    cutoff = datetime.now().astimezone() - timedelta(days=lookback_days)
    new_papers = []
    skipped = 0

    for result in client.results(search):
        arxiv_id = result.entry_id.split('/')[-1]

        if result.published < cutoff:
            continue
        if arxiv_id in existing_arxiv_ids:
            continue

        title = result.title
        abstract = result.summary.replace('\n', ' ')

        # Filter out non-weather papers
        if not is_weather_related(title, abstract):
            skipped += 1
            continue

        arxiv_cats = [result.primary_category] + [
            c for c in result.categories if c != result.primary_category
        ]

        category = categorize_paper(title, abstract)
        tags = extract_tags(title, abstract, arxiv_cats)

        # Extract GitHub URL from abstract or comments
        github_url = None
        comments = result.comment or ''
        for text in [abstract, comments]:
            match = GITHUB_URL_PATTERN.search(text)
            if match:
                github_url = match.group(0).rstrip('.')
                break

        paper = {
            'category': category,
            'title': title,
            'authors': ', '.join(author.name for author in result.authors),
            'year': result.published.year,
            'arxiv': arxiv_id,
            'abstract': abstract,
            'tags': tags,
            'arxiv_categories': arxiv_cats,
        }
        if github_url:
            paper['github'] = github_url
        new_papers.append(paper)

    if new_papers:
        print(f"Found {len(new_papers)} new papers ({skipped} skipped as non-weather-related).")
        all_papers = existing_papers + new_papers
        with open('papers.yml', 'w') as f:
            yaml.dump(all_papers, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)
    else:
        print(f"No new papers found ({skipped} skipped as non-weather-related).")


if __name__ == '__main__':
    find_new_papers()
