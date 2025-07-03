import httpx
from selectolax.parser import HTMLParser

async def duckduckgo_search(query, max_results=5):
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=10)
    html = HTMLParser(response.text)
    results = []
    for result in html.css(".result__snippet")[:max_results]:
        link_el = result.parent.css_first("a")
        if link_el:
            title = link_el.text(strip=True)
            link = link_el.attributes.get("href", "")
            snippet = result.text(strip=True)
            results.append({"title": title, "link": link, "snippet": snippet})
    return results
