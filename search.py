# search.py (modify to accept logger)
import httpx
from selectolax.parser import HTMLParser

async def duckduckgo_search(query, max_results=5, logger=None):
    if logger:
        await logger.log(f"[duckduckgo_search] Searching for query: {query}")

    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=10)

    html = HTMLParser(response.text)
    results = []

    for result in html.css("div.result")[:max_results]:
        title_el = result.css_first("a.result__a")
        snippet_el = result.css_first(".result__snippet")

        if title_el and snippet_el:
            title = title_el.text(strip=True)
            link = title_el.attributes.get("href", "")
            snippet = snippet_el.text(strip=True)
            results.append({"title": title, "link": link, "snippet": snippet})
            if logger:
                await logger.log(f"[duckduckgo_search] Found result: {title} - {link}")
        else:
            if logger:
                await logger.log("[duckduckgo_search] Skipped a result due to missing title or snippet.")

    if logger:
        await logger.log(f"[duckduckgo_search] Total results found: {len(results)}")

    return results
