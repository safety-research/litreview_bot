You need:
- an Anthropic API key (env var `ANTHROPIC_API_KEY`)
- SerpAPI api key (env var `SERPAPI_API_KEY`)

Usage:
```
usage: hello.py [-h] --query QUERY [--num-results NUM_RESULTS] [--output-md-file OUTPUT_MD_FILE]
                [--output-html-file OUTPUT_HTML_FILE] [--filter-query FILTER_QUERY]
```

Example:
run
```
python hello.py --query="large language model lie detection" \
--filter-query="This paper describes a method to measure and detect large language models lying" \
--num-results=50
```
and look for out.md and out.html files