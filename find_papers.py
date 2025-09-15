import arxiv
import yaml
from datetime import datetime, timedelta
import os

# Keywords to search for
KEYWORDS = [
    "weather forecasting", "climate model", "precipitation nowcasting",
    "deep learning meteorology", "neural weather prediction", "data assimilation",
    "ensemble forecasting", "downscaling", "numerical weather prediction"
]

def find_new_papers():
    with open('papers.yml', 'r') as f:
        existing_papers = yaml.safe_load(f)

    existing_arxiv_ids = {p['arxiv'] for p in existing_papers}

    search_query = " OR ".join(f'ti:"{kw}"' for kw in KEYWORDS)

    search = arxiv.Search(
        query=search_query,
        max_results=100, # Adjust as needed
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    new_papers = []
    for result in search.results():
        arxiv_id = result.entry_id.split('/')[-1]
        # Check if paper is recent (last 7 days) and not already in our list
        if result.published > datetime.now().astimezone() - timedelta(days=7):
            if arxiv_id not in existing_arxiv_ids:
                paper = {
                    'category': 'New',
                    'title': result.title,
                    'authors': ", ".join(author.name for author in result.authors),
                    'year': result.published.year,
                    'arxiv': arxiv_id,
                    'abstract': result.summary.replace('\n', ' '),
                    'tags': [] # Let user add tags
                }
                new_papers.append(paper)

    if new_papers:
        print(f"Found {len(new_papers)} new papers.")
        all_papers = existing_papers + new_papers
        with open('papers.yml', 'w') as f:
            yaml.dump(all_papers, f, default_flow_style=False, sort_keys=False)

        # Run the build script
        os.system('python build_pages.py')
    else:
        print("No new papers found.")

if __name__ == '__main__':
    find_new_papers()
