import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

for query in ("Reliance Industries", "Apple stock"):
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(query)
           + "&hl=en-IN&gl=IN&ceid=IN:en")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        root = ET.fromstring(resp.read())
    items = root.findall("./channel/item")
    print(query, "->", len(items), "headlines")
    for item in items[:3]:
        print("  ", item.findtext("title"), "|", item.findtext("pubDate"))