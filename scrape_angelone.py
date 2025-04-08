import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

BASE_URL = "https://www.angelone.in/support"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

visited = set()
pages_data = {}

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.netloc == "www.angelone.in" and "/support" in parsed.path

def get_all_links(soup, base_url):
    links = set()
    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, a_tag['href'])
        if is_valid_url(full_url) and full_url not in visited:
            links.add(full_url)
    return links

def scrape_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        title = soup.title.string if soup.title else "No Title"
        pages_data[url] = {
            "title": title,
            "content": text
        }
        print(f"[+] Scraped: {url}")
        return soup
    except Exception as e:
        print(f"[!] Failed: {url} | Reason: {e}")
        return None

def crawl(start_url):
    to_visit = [start_url]
    while to_visit:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)
        soup = scrape_page(current_url)
        if soup:
            new_links = get_all_links(soup, current_url)
            to_visit.extend(new_links)
        time.sleep(1)  # be polite

def save_to_file(filename="angelone_support_data.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for url, data in pages_data.items():
            f.write(f"URL: {url}\nTitle: {data['title']}\n\n{data['content']}\n")
            f.write("\n" + "="*100 + "\n\n")

if __name__ == "__main__":
    crawl(BASE_URL)
    save_to_file()
    print("[✔] Data saved to 'angelone_support_data.txt'")
