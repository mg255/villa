from typing import List, Optional

import aiohttp

async def fetch(session: aiohttp.ClientSession, url: str, failures: Optional[List[str]] = None) -> str:
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientError as e:
        print(f"Failed to fetch {url}: {e}")
        if failures is not None:
            failures.append(url)
        return ""