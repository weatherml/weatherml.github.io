# WeatherML.github.io Documentation Site

WeatherML.github.io is a MkDocs-based documentation website that curates and presents academic papers on deep learning applications in weather forecasting. The site automatically generates categorized documentation pages from a YAML database of papers and is deployed via GitHub Pages.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Build Process
- Install Python dependencies:
  - `pip install -r requirements.txt` -- takes 15 seconds. NEVER CANCEL.
- Generate documentation pages from papers database:
  - `python build_pages.py` -- takes <1 second
- Build static site:
  - `mkdocs build` -- takes 3 seconds. NEVER CANCEL.
- Serve site locally:
  - `mkdocs serve --dev-addr=0.0.0.0:8000` -- starts in 3 seconds. Site available at http://localhost:8000

### Complete Working Workflow
Always run these commands in sequence for full development workflow:
```bash
pip install -r requirements.txt
python build_pages.py
mkdocs build
mkdocs serve --dev-addr=0.0.0.0:8000
```

### Paper Management Workflow
- Edit papers database: modify `papers.yml` 
- Regenerate pages: `python build_pages.py`
- Rebuild site: `mkdocs build`
- **WARNING**: `python find_papers.py` -- fails due to network restrictions to arxiv.org. Document as known limitation.

## Validation Requirements

### ALWAYS Test Complete User Scenarios
After making any changes, ALWAYS validate by:
1. Run complete build workflow (commands above)
2. Verify site serves correctly: `curl -s http://localhost:8000 | head -20`
3. Test category navigation: `curl -s http://localhost:8000/new/` 
4. Verify paper content is rendered correctly in generated pages
5. Check that all documentation pages are accessible

### Manual Testing Scenarios
- **Paper Addition**: Add a new paper to papers.yml, rebuild, verify it appears on site
- **Category Navigation**: Verify all category pages (New, Downscaling, Ensembles, Data Assimilation, Global Models, Nowcasting) are accessible
- **Site Generation**: Ensure build_pages.py correctly updates both README.md and mkdocs.yml navigation
- **Deploy Simulation**: Run the exact commands from .github/workflows/deploy.yml locally

## Build and Test Configuration

### No Linting or Testing Infrastructure
- Repository has NO linting tools configured
- Repository has NO test suite
- Repository has NO code formatting requirements
- CI/CD only validates successful build and deployment

### Timing and Performance
- **NEVER CANCEL**: pip install takes 15 seconds maximum
- **NEVER CANCEL**: mkdocs build takes 3 seconds maximum
- **NEVER CANCEL**: python build_pages.py takes <1 second
- Site generation is extremely fast - no long-running processes
- Set timeout to 60 seconds minimum for any build commands

### Known Network Limitations
- `python find_papers.py` -- FAILS due to network restrictions (cannot access export.arxiv.org)
- Document this limitation when modifying paper discovery functionality
- Manual paper addition to papers.yml is the only reliable method in restricted environments

## Repository Structure and Navigation

### Key Files and Directories
```
.github/workflows/          # GitHub Actions (deploy.yml, update_papers.yml)
docs/                      # Generated MkDocs content (auto-generated)
.github/copilot-instructions.md  # This file
README.md                  # Auto-updated with paper summaries
mkdocs.yml                 # MkDocs configuration (auto-updated)
papers.yml                 # Paper database (MANUAL EDITS ONLY)
requirements.txt           # Python dependencies
build_pages.py             # Page generation script
find_papers.py             # arXiv paper discovery (network restricted)
```

### Generated Content (Do Not Edit Manually)
- All files in `docs/` directory
- Navigation section in `mkdocs.yml`
- Paper list section in `README.md` (between `<!-- PAPERS_START -->` markers)

### Manual Edit Files Only
- `papers.yml` -- paper database
- `docs/index.md` -- site homepage content
- `mkdocs.yml` -- site configuration (except nav section)
- `requirements.txt` -- dependencies

## Deployment and CI/CD

### GitHub Actions Workflows
- **deploy.yml**: Triggered on push to main branch
  - Installs dependencies
  - Runs `python build_pages.py`
  - Deploys with `mkdocs gh-deploy --force`
- **update_papers.yml**: Scheduled weekly, manual trigger available
  - Runs `python find_papers.py` to discover new papers
  - Commits and pushes changes if new papers found

### Local Development Simulation
To simulate CI/CD locally:
```bash
# Simulate deploy.yml workflow
pip install -r requirements.txt
python build_pages.py  
mkdocs gh-deploy --force  # Only run if you want to actually deploy

# Simulate update_papers.yml workflow (will fail due to network)
pip install -r requirements.txt
python find_papers.py  # Expected to fail with connection error
```

## Common Development Tasks

### Adding New Papers
1. Edit `papers.yml` directly with new paper entry
2. Run `python build_pages.py` 
3. Verify changes with `mkdocs serve`
4. Test complete user scenarios

### Modifying Site Content
1. Edit `docs/index.md` for homepage changes
2. Edit `mkdocs.yml` for configuration changes (avoid nav section)
3. Run build workflow to test changes
4. Validate site functionality

### Troubleshooting
- **Site not building**: Check `mkdocs.yml` syntax
- **Papers not appearing**: Verify `papers.yml` format and run `python build_pages.py`
- **Navigation broken**: Check that `build_pages.py` completed successfully
- **Network errors**: Expected for `find_papers.py` in restricted environments

### Performance Expectations
- Complete development workflow: <30 seconds total
- Individual commands execute in seconds, not minutes
- No long-running builds or complex dependencies
- Immediate visual feedback when serving locally

## Integration Points

### External Dependencies
- **arxiv.org**: Required for `find_papers.py` (often network restricted)
- **GitHub Pages**: Deployment target
- **MkDocs Material Theme**: UI framework

### Data Flow
1. Papers data in `papers.yml`
2. `build_pages.py` generates markdown files in `docs/`
3. `build_pages.py` updates `README.md` and `mkdocs.yml`
4. `mkdocs build` creates static site in `site/`
5. `mkdocs serve` or GitHub Pages serves the final site

Always build and exercise your changes using the complete workflow above.