import asyncio
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        print('✅ Browser launched successfully')
        await browser.close()

asyncio.run(test())