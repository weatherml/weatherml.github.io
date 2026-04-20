"""One-time script to re-categorize existing papers in papers.yml.

Applies auto-categorization to papers currently in the 'New' category,
filters out non-weather-related papers, and adds method tags.

Usage:
    python categorize_papers.py
"""

import yaml
import re
from find_papers import (
    categorize_paper,
    extract_tags,
    is_weather_related,
    GITHUB_URL_PATTERN,
)


def main():
    with open('papers.yml', 'r') as f:
        papers = yaml.safe_load(f) or []

    stats = {
        'recategorized': 0,
        'filtered': 0,
        'tagged': 0,
        'kept': 0,
    }

    updated_papers = []

    for paper in papers:
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        # Check weather relevance
        if not is_weather_related(title, abstract):
            print(f"  FILTERED (not weather): {title[:80]}")
            stats['filtered'] += 1
            continue

        # Re-categorize papers in "New" category
        if paper['category'] == 'New':
            new_category = categorize_paper(title, abstract)
            if new_category != paper['category']:
                print(f"  {paper['category']} -> {new_category}: {title[:60]}")
                stats['recategorized'] += 1
            paper['category'] = new_category

        # Add/update tags if empty
        if not paper.get('tags'):
            arxiv_cats = paper.get('arxiv_categories', [])
            paper['tags'] = extract_tags(title, abstract, arxiv_cats)
            if paper['tags']:
                stats['tagged'] += 1

        # Extract GitHub URL if not already present
        if not paper.get('github'):
            match = GITHUB_URL_PATTERN.search(abstract)
            if match:
                paper['github'] = match.group(0).rstrip('.')

        stats['kept'] += 1
        updated_papers.append(paper)

    # Save updated papers
    with open('papers.yml', 'w') as f:
        yaml.dump(updated_papers, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    print(f"\nResults:")
    print(f"  Papers kept: {stats['kept']}")
    print(f"  Recategorized: {stats['recategorized']}")
    print(f"  Filtered out: {stats['filtered']}")
    print(f"  Tags added: {stats['tagged']}")

    # Show category distribution
    from collections import Counter
    cats = Counter(p['category'] for p in updated_papers)
    print(f"\nCategory distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == '__main__':
    main()
