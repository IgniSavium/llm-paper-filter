"""Fetch public accepted papers or active submissions via OpenReview API v2/v1.

Require the exact main-conference venueid and an explicit accepted venue label.
Older venues can retain the conference venueid on unaccepted submissions.
"""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

import requests

from daily_arxiv.fetching.snapshots import load_paper_input, raw_papers, save_snapshot

BASE_DIR = Path(__file__).resolve().parent
CONFERENCES = {"ICML": "ICML.cc", "NeurIPS": "NeurIPS.cc", "ICLR": "ICLR.cc"}
API_URLS = ("https://api2.openreview.net", "https://api.openreview.net")
DEFAULT_CREDENTIALS_FILE = BASE_DIR / "credentials.json"
PAPER_STATUSES = ('accepted', 'submitted')


def corpus_name(conference, year, paper_status='accepted'):
    if paper_status not in PAPER_STATUSES:
        raise ValueError(f'Unsupported paper status: {paper_status}')
    return f'{conference}_{year}' + ('_submitted' if paper_status == 'submitted' else '')


def load_credentials(path=DEFAULT_CREDENTIALS_FILE, *, required=False):
    path = Path(path).expanduser()
    if not path.exists() and not required:
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise ValueError(f"Cannot read OpenReview credentials JSON: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError('OpenReview credentials must be a JSON object')
    if payload.get('token'):
        if not isinstance(payload['token'], str) or not payload['token'].strip():
            raise ValueError('OpenReview token must be a nonempty string')
        return {'token': payload['token'].removeprefix('Bearer ').strip()}
    if any(not isinstance(payload.get(key), str) or not payload[key].strip()
           for key in ('username', 'password')):
        raise ValueError(f"Fill in username and password in {path} before fetching")
    return {'username': payload['username'].strip(), 'password': payload['password']}


def default_years():
    """Three previous calendar years plus the current year."""
    year = datetime.now(timezone.utc).year
    return list(range(year - 3, year + 1))


def venue_id(conference, year):
    if conference not in CONFERENCES or not 2000 <= year <= datetime.now(timezone.utc).year + 1:
        raise ValueError("Unsupported conference or year (expected 2000 through next year)")
    return f"{CONFERENCES[conference]}/{year}/Conference"


def value(content, field, default=None):
    result = content.get(field, default)
    return result.get("value", default) if isinstance(result, dict) else result


ACCEPTED_VENUE_LABELS = frozenset({
    'poster', 'oral', 'spotlight', 'spotlightposter', 'oralposter',
    'spotlight poster', 'oral poster', 'regular',
    'notable top 25%', 'notable top 5%',
})


def is_accepted_paper(paper, conference, year):
    """Do not trust the synthetic status field or conference venueid alone."""
    label = paper.get('venue')
    if paper.get('venueid') != venue_id(conference, year) or not isinstance(label, str):
        return False
    normalized = ' '.join(label.casefold().split())
    prefix = f'{conference} {year} '.casefold()
    return normalized.startswith(prefix) and normalized[len(prefix):] in ACCEPTED_VENUE_LABELS


def is_submitted_paper(paper, conference, year):
    """Active main-track submissions only; exclude decisions and withdrawals."""
    venue = venue_id(conference, year)
    label = paper.get('venue')
    normalized = ' '.join(label.casefold().split()) if isinstance(label, str) else ''
    return (paper.get('venueid') in (venue, venue + '/Submission')
            and normalized in {f'Submitted to {conference} {year}'.casefold(),
                               f'{conference} {year} Conference Submission'.casefold()})


class OpenReviewClient:
    def __init__(self, *, timeout=60, retries=5, page_size=1000, session=None, credentials=None):
        if timeout <= 0 or retries < 1 or not 1 <= page_size <= 1000:
            raise ValueError("timeout/retries must be positive; page_size must be 1–1000")
        self.session = session or requests.Session()
        self.timeout, self.retries, self.page_size = timeout, retries, page_size
        self.credentials = credentials or {}
        self.tokens = {}
        self.token_times = {}

    def auth_headers(self, api):
        if not self.credentials:
            return {}
        if api not in API_URLS:
            raise ValueError('Credentials can only be sent to official OpenReview API hosts')
        if (api in self.tokens and not self.credentials.get('token')
                and time.monotonic() - self.token_times[api] >= 2700):
            self.tokens.pop(api)
        if api not in self.tokens:
            if self.credentials.get('token'):
                self.tokens[api] = self.credentials['token']
            else:
                # Same /login payload and Bearer header as the official Python client.
                payload = self.login(api)
                if not isinstance(payload, dict):
                    raise RuntimeError('Invalid OpenReview login response')
                if payload.get('mfaPending'):
                    raise RuntimeError('OpenReview requires MFA. Complete login with the official client '
                                       'and configure its session token as the token field in credentials.json.')
                token = payload.get('token')
                if not isinstance(token, str) or not token.strip():
                    raise RuntimeError('OpenReview login returned no session token')
                self.tokens[api] = token
                self.token_times[api] = time.monotonic()
        return {'Authorization': 'Bearer ' + self.tokens[api]}

    def login(self, api):
        for attempt in range(self.retries):
            response = None
            try:
                response = self.session.post(api + '/login',
                    json={'id': self.credentials['username'], 'password': self.credentials['password'],
                          'expiresIn': 3600}, timeout=self.timeout, allow_redirects=False)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if not 200 <= response.status_code < 300:
                    raise RuntimeError(f"OpenReview login failed ({response.status_code}) at {api}. "
                                       "Check local credentials; challenge verification may still be required.")
                return response.json()
            except (requests.RequestException, ValueError):
                if attempt + 1 == self.retries:
                    status = f' (HTTP {response.status_code})' if response is not None else ''
                    raise RuntimeError(f"OpenReview login request failed{status} after {self.retries} attempts at {api}; "
                                       "429 means temporary throttling; credentials were not logged") from None
                delay = min(2 ** (attempt + 1), 30)
                if response is not None:
                    retry_after = response.headers.get('Retry-After', '')
                    if retry_after.isdigit():
                        delay = min(int(retry_after), 30)
                time.sleep(delay)

    def request(self, api, params):
        headers = self.auth_headers(api)
        for attempt in range(self.retries):
            try:
                response = self.session.get(api + "/notes", params=params, timeout=self.timeout,
                                            headers=headers, allow_redirects=False)
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"OpenReview denied access ({response.status_code}) at {api}. "
                        "The service may require challenge verification or authentication. "
                        "No partial snapshot was saved; retry after resolving access.")
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                elif response.status_code >= 400:
                    raise RuntimeError(f"OpenReview request failed ({response.status_code}): {api}")
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("notes"), list):
                    raise ValueError("Invalid OpenReview notes response")
                return payload
            except (requests.RequestException, ValueError) as exc:
                if attempt + 1 == self.retries:
                    raise RuntimeError(f"OpenReview request failed after {self.retries} attempts: {exc}") from exc
                delay = min(2 ** (attempt + 1), 30)
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    retry_after = exc.response.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = min(int(retry_after), 60)
                time.sleep(delay)

    def notes(self, api, venue, *, invitation=None):
        offset, total, notes, seen = 0, None, [], set()
        while True:
            params = {"content.venueid": venue, "limit": self.page_size,
                      "sort": "number:asc"}
            if invitation is not None:
                params.pop('content.venueid')
                params['invitation'] = invitation
            # API v1 returns count by default and rejects the v2 count flag.
            if api == API_URLS[0]:
                params['count'] = 'true'
            if offset:
                params['offset'] = offset
            payload = self.request(api, params)
            count = payload.get("count")
            if (count is not None or total is None) and (
                    not isinstance(count, int) or isinstance(count, bool) or count < 0):
                raise RuntimeError("OpenReview response has no valid total count")
            if total is None:
                total = count
            elif count is not None and total != count:
                raise RuntimeError("OpenReview corpus changed during pagination; rerun to fetch a complete snapshot")
            page = payload["notes"]
            for note in page:
                if not isinstance(note, dict) or not isinstance(note.get("id"), str) or not note["id"]:
                    raise RuntimeError("OpenReview returned a note without an ID")
                if note["id"] in seen:
                    raise RuntimeError("OpenReview pagination repeated a note; no partial snapshot saved")
                seen.add(note["id"])
            notes.extend(page)
            offset += len(page)
            if offset == total:
                return notes
            if not page or offset > total:
                raise RuntimeError("Incomplete OpenReview pagination; no partial snapshot saved")

    def fetch(self, conference, year, *, allow_empty=False, paper_status='accepted'):
        corpus_name(conference, year, paper_status)
        if paper_status == 'submitted':
            return self.fetch_submitted(conference, year)
        venue = venue_id(conference, year)
        # ICLR 2023 and older venues can still live on API v1.
        apis = API_URLS[::-1] if conference == "ICLR" and year <= 2023 else API_URLS
        for api in apis:
            notes = self.notes(api, venue)
            if notes:
                papers = []
                for note in notes:
                    content = note.get('content', {})
                    if (not isinstance(content, dict) or value(content, 'venueid') != venue
                            or note.get('replyto')):
                        raise ValueError(f'Unexpected note in {venue}: {note.get("id")}')
                    if is_accepted_paper({'venueid': value(content, 'venueid'),
                                          'venue': value(content, 'venue')}, conference, year):
                        papers.append(normalize_note(note, conference, year))
                excluded = len(notes) - len(papers)
                if excluded:
                    print(f'{conference} {year}: excluded {excluded} records without an explicit accepted venue label', flush=True)
                if papers:
                    return sorted(raw_papers(papers), key=lambda p: p["openreview_id"]), api
        if allow_empty:
            return [], None
        raise RuntimeError(f"No public accepted papers found for {venue}. The year may not be published "
                           "or may not be available on OpenReview. No empty snapshot saved.")

    def fetch_submitted(self, conference, year):
        venue = venue_id(conference, year)
        apis = API_URLS[::-1] if conference == 'ICLR' and year <= 2023 else API_URLS
        for api in apis:
            invitation = venue + ('/-/Submission' if api == API_URLS[0] else '/-/Blind_Submission')
            notes = self.notes(api, venue, invitation=invitation)
            papers = []
            for note in notes:
                invitations = note.get('invitations', [note.get('invitation')])
                if invitation not in invitations or note.get('replyto'):
                    raise ValueError(f'Unexpected non-submission note: {note.get("id")}')
                content = note.get('content', {})
                if not isinstance(content, dict):
                    raise ValueError(f'Invalid submission content: {note.get("id")}')
                if ('everyone' in note.get('readers', []) and not note.get('ddate')
                        and is_submitted_paper({'venueid': value(content, 'venueid'),
                                               'venue': value(content, 'venue')}, conference, year)):
                    papers.append(normalize_note(note, conference, year, paper_status='submitted'))
            print(f'{conference} {year}: {len(papers)} public active submissions; '
                  f'excluded {len(notes) - len(papers)} private/inactive records. '
                  'Coverage is limited to publicly available submissions.', flush=True)
            if papers:
                return sorted(raw_papers(papers), key=lambda p: p['openreview_id']), api
        # An empty visible corpus never proves that a venue has no submissions.
        return [], None


def normalize_note(note, conference, year, *, paper_status='accepted'):
    corpus_name(conference, year, paper_status)
    content = note.get("content", {})
    if not isinstance(content, dict):
        raise ValueError(f"Invalid content in paper {note.get('id')}")
    venue = venue_id(conference, year)
    if (paper_status == 'accepted' and value(content, "venueid") != venue) or note.get("replyto"):
        raise ValueError(f"Unexpected non-accepted/non-submission note: {note.get('id')}")
    predicate = is_accepted_paper if paper_status == 'accepted' else is_submitted_paper
    if not predicate({'venueid': value(content, 'venueid'),
                      'venue': value(content, 'venue')}, conference, year):
        raise ValueError(f'Paper {note.get("id")} lacks an explicit {paper_status} venue label')
    if paper_status == 'submitted' and 'everyone' not in note.get('readers', []):
        raise ValueError(f'Submission {note.get("id")} is not public')
    title, abstract = value(content, "title"), value(content, "abstract")
    if any(not isinstance(text, str) or not text.strip() for text in (title, abstract)):
        raise ValueError(f"Paper {note.get('id')} lacks a public title/abstract")
    if paper_status == 'submitted' and any(
            isinstance(content.get(field), dict) and content[field].get('readers') is not None
            and 'everyone' not in content[field]['readers'] for field in ('title', 'abstract')):
        raise ValueError(f'Submission {note.get("id")} has restricted metadata')
    authors = value(content, "authors", [])
    author_field = content.get('authors')
    if (paper_status == 'submitted' and isinstance(author_field, dict)
            and author_field.get('readers') is not None and 'everyone' not in author_field['readers']):
        # Login may reveal authors to a reviewer or coauthor; keep the public view.
        authors = []
    if not isinstance(authors, list) or any(not isinstance(author, str) for author in authors):
        raise ValueError(f"Invalid authors in paper {note.get('id')}")
    note_id = note["id"]
    return {"title": title.strip(), "abstract": abstract.strip(), "authors": authors,
            "url": f"https://openreview.net/forum?id={note_id}",
            "pdf_url": f"https://openreview.net/pdf?id={note_id}",
            "openreview_id": note_id, "conference": conference, "year": year,
            "venueid": value(content, 'venueid'), "venue": value(content, "venue", f"{conference} {year}"),
            "status": f"{conference} {year} {paper_status.title()}"}


def get_snapshot(conference, year, paper_info_dir=BASE_DIR / "paper_info", *, refresh=False, client=None,
                 paper_status='accepted'):
    venue = venue_id(conference, year)
    # A year label, not a publication date. Separate directories prevent collisions.
    label = f"{year}-01-01"
    directory = Path(paper_info_dir) / corpus_name(conference, year, paper_status)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"papers_{label}.json"
    with (directory / ".fetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and not refresh:
            papers, _ = load_paper_input(path, label)
            metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
            predicate = is_accepted_paper if paper_status == 'accepted' else is_submitted_paper
            if (metadata.get("source") != "openreview" or metadata.get("venueid") != venue
                    or metadata.get('paper_status', 'accepted') != paper_status
                    or not papers or any(not predicate(p, conference, year) for p in papers)):
                raise ValueError(f"Invalid conference snapshot: {path}; use --refresh")
            return path
        options = {'paper_status': paper_status} if paper_status != 'accepted' else {}
        papers, api = (client or OpenReviewClient()).fetch(
            conference, year, allow_empty=year >= datetime.now(timezone.utc).year, **options)
        if not papers:
            print(f"{conference} {year}: no public {paper_status} papers available; skipped "
                  "(the list may be private or not released; no snapshot saved)", flush=True)
            return None
        return save_snapshot(papers, label, directory, replace=True, source="openreview",
                             metadata_extra={"conference": conference, "conference_year": year,
                                             "venueid": venue, "api_url": api,
                                             "date_basis": "conference_year",
                                             "paper_status": paper_status, "track": "main",
                                             "coverage": 'public_active_submissions_only' if paper_status == 'submitted' else 'public_accepted_papers',
                                             "fetched_at": datetime.now(timezone.utc).isoformat()})
