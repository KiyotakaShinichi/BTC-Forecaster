from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

from .models import Document


class SearchProvider(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, start: datetime, end: datetime) -> list[Document]: ...


def deduplicate_documents(documents: Iterable[Document]) -> list[Document]:
    unique: dict[str, Document] = {}
    for document in documents:
        fingerprint = Document.stable_id(str(document.url), document.text_hash)
        current = unique.get(fingerprint)
        if current is None or document.retrieved_at < current.retrieved_at:
            unique[fingerprint] = document
    return sorted(unique.values(), key=lambda d: (d.available_at, d.document_id))


class FixtureSearchProvider(SearchProvider):
    name = "fixture"

    def __init__(self, documents: Iterable[Document]):
        self._documents = list(documents)

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        return deduplicate_documents(
            d for d in self._documents if d.query == query and start <= d.available_at <= end
        )


class JsonSearchApiProvider(SearchProvider):
    """Generic JSON API adapter. Credentials are supplied by caller from environment."""

    def __init__(self, endpoint: str, api_key: str, provider_name: str = "web-api", timeout: int = 15):
        self.endpoint, self.api_key, self.name, self.timeout = endpoint, api_key, provider_name, timeout

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        params = urllib.parse.urlencode({"q": query, "start": start.isoformat(), "end": end.isoformat()})
        request = urllib.request.Request(f"{self.endpoint}?{params}", headers={"Authorization": f"Bearer {self.api_key}"})
        retrieved = datetime.now(timezone.utc)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # nosec: configured API endpoint
            payload = json.load(response)
        documents = []
        for item in payload.get("results", []):
            text = item.get("text") or item.get("snippet") or ""
            published = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")) if item.get("published_at") else None
            # Without a vendor-supplied first-seen timestamp, retrieval is the
            # earliest time we can prove the system knew this item existed.
            available = retrieved
            text_hash = Document.content_hash(text)
            documents.append(Document(
                document_id=Document.stable_id(item["url"], text_hash), url=item["url"],
                publisher=item.get("publisher") or "unknown", title=item["title"], published_at=published,
                retrieved_at=retrieved, available_at=available, author=item.get("author"), text_hash=text_hash,
                query=query, provider=self.name,
            ))
        return deduplicate_documents(documents)


class RssSearchProvider(SearchProvider):
    """Standards-based RSS/Atom adapter; only fetches feeds explicitly configured by the operator."""

    name = "rss"

    def __init__(self, feed_urls: list[str], timeout: int = 15):
        self.feed_urls, self.timeout = feed_urls, timeout

    def search(self, query: str, start: datetime, end: datetime) -> list[Document]:
        retrieved, output = datetime.now(timezone.utc), []
        terms = query.casefold().split()
        for feed_url in self.feed_urls:
            with urllib.request.urlopen(feed_url, timeout=self.timeout) as response:  # nosec: operator allowlist
                root = ET.parse(response).getroot()
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                description = item.findtext("description", "").strip()
                if terms and not all(term in f"{title} {description}".casefold() for term in terms):
                    continue
                link = item.findtext("link", "").strip()
                pub_text = item.findtext("pubDate")
                published = parsedate_to_datetime(pub_text).astimezone(timezone.utc) if pub_text else None
                available = retrieved
                if not start <= available <= end:
                    continue
                text_hash = Document.content_hash(description)
                output.append(Document(document_id=Document.stable_id(link, text_hash), url=link,
                    publisher=urllib.parse.urlparse(feed_url).netloc, title=title, published_at=published,
                    retrieved_at=retrieved, available_at=available, author=item.findtext("author"),
                    text_hash=text_hash, query=query, provider=self.name))
        return deduplicate_documents(output)
