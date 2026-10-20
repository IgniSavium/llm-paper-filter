"""Optional real-browser tests for paging and local bookmark backup/restore.

Run with Python Playwright and its Chromium browser installed. No model or
external service is used; all requests are served from a temporary local site.
"""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from daily_arxiv.reporting.html import generate_mobile_html_report

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@unittest.skipIf(sync_playwright is None, 'Install Python Playwright for browser tests')
class ReaderBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        papers = [dict(title=f'Paper {index:03d}', abstract='A model represents $x$.', authors=['Author'],
                       url=f'https://openreview.net/forum?id=id{index}', zh_title='论文', zh_abstract='模型表示 $x$。',
                       eval=dict(scores=dict(MechInterp=5 if index < 110 else 4, CompGen=5 if 110 <= index < 115 else 1,
                                             Reasoning=1, Systems=1), reason='Mechanisms.', zh_reason='机制。'))
                  for index in range(120)]
        metadata = dict(source='openreview', conference='ICLR', conference_year=2027)
        generate_mobile_html_report(papers, '2027-01-01', cls.root / 'conference', metadata,
                                    page_size=50, asset_dir=cls.root / 'assets')
        generate_mobile_html_report(papers[:3], '2027-01-01', cls.root / 'offline', metadata)
        daily_paper = dict(papers[0], url='https://arxiv.org/abs/2610.00001v2')
        generate_mobile_html_report([daily_paper], '2027-01-01', cls.root / 'daily')
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=cls.root))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'
        cls.playwright = sync_playwright().start()
        env = dict(os.environ)
        local_libraries = Path('/tmp/arxiv-browser-libs/extracted/usr/lib/x86_64-linux-gnu')
        if local_libraries.exists():
            env['LD_LIBRARY_PATH'] = str(local_libraries) + ':' + env.get('LD_LIBRARY_PATH', '')
        try:
            cls.browser = cls.playwright.chromium.launch(headless=True, args=['--no-sandbox'], env=env)
        except Exception as exc:
            cls.playwright.stop()
            cls.server.shutdown(); cls.server.server_close(); cls.temporary.cleanup()
            raise unittest.SkipTest(f'Chromium is unavailable: {str(exc).splitlines()[0]}')

    @classmethod
    def tearDownClass(cls):
        cls.browser.close(); cls.playwright.stop()
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.temporary.cleanup()

    def setUp(self):
        self.context = self.browser.new_context(accept_downloads=True)
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))

    def open(self, kind='conference'):
        self.page.goto(self.base + f'/{kind}/html_reports/report_2027-01-01.html')
        self.page.wait_for_function("document.querySelectorAll('.paper-card:not([hidden])').length > 0")

    def wait_cards(self, count):
        self.page.wait_for_function('(count) => document.querySelectorAll(".paper-card:not([hidden])").length === count', arg=count)

    def backup(self):
        with self.page.expect_download() as download:
            self.page.click('#backup-bookmarks')
        path = download.value.path()
        return Path(path).read_bytes()

    def restore(self, payload):
        self.page.evaluate("document.getElementById('backup-status').hidden=true")
        self.page.set_input_files('#bookmark-file', dict(name='backup.json', mimeType='application/json', buffer=payload))
        self.page.wait_for_function("!document.getElementById('backup-status').hidden")

    def test_global_filter_paging_and_lazy_formulas(self):
        requests = []
        self.page.on('request', lambda request: requests.append(request.url))
        self.open()
        self.assertEqual(self.page.locator('.paper-card').count(), 50)
        self.assertEqual(len([url for url in requests if '/data_' in url]), 1)
        self.assertEqual(self.page.locator('.abstract .katex').count(), 0)
        self.page.locator('.abstract summary').first.click()
        self.page.wait_for_selector('.abstract .katex')
        self.page.click('#next-page')
        self.page.wait_for_selector('.paper-card[data-title="Paper 050"]')
        self.page.select_option('#MechInterp-mode', 'ignore')
        self.wait_cards(5)
        self.assertEqual(self.page.locator('#visible-count').inner_text(), '5')
        self.assertEqual(self.page.locator('.paper-card').first.get_attribute('data-title'), 'Paper 110')
        self.assertEqual(self.errors, [])

    def test_download_clear_browser_data_restore_order_merge_and_reload(self):
        self.open()
        self.page.locator('.bookmark-button').nth(0).click()
        self.page.wait_for_selector('.bookmark-button[aria-pressed="true"]')
        self.page.locator('.bookmark-button').nth(1).click()
        self.page.click('#next-page')
        self.page.wait_for_selector('.paper-card[data-title="Paper 050"]')
        self.page.locator('.bookmark-button').first.click()
        self.page.check('#marked-only')
        self.wait_cards(3)
        self.page.locator('.bookmark-row button[aria-label="下移这篇论文"]').first.click()
        expected = self.page.locator('.bookmark-row .bookmark-title').all_text_contents()
        backup = self.backup()
        payload = json.loads(backup)
        self.assertEqual(len(payload['bookmarks']), 3)
        self.assertEqual([row['title'] for row in payload['bookmarks']], expected)
        self.page.evaluate('localStorage.clear()')
        self.page.reload(); self.wait_cards(50)
        self.assertEqual(self.page.locator('#marked-count').inner_text(), '0')
        self.restore(backup)
        self.page.check('#marked-only'); self.wait_cards(3)
        self.assertEqual(self.page.locator('.bookmark-row .bookmark-title').all_text_contents(), expected)
        self.page.reload(); self.wait_cards(50)
        self.assertEqual(self.page.locator('#marked-count').inner_text(), '3')
        self.page.locator('.bookmark-button').nth(2).click()
        self.restore(backup)
        self.assertEqual(len(json.loads(self.backup())['bookmarks']), 4)
        self.assertEqual(self.errors, [])

    def test_legacy_keys_migrate_and_backup_includes_other_reports(self):
        self.page.add_init_script("localStorage.setItem('arxiv-reader-bookmarks-v1', JSON.stringify(['url:openreview.net/forum?id=id0','arxiv:other']));localStorage.setItem('arxiv-reader-bookmark-order-v1',JSON.stringify(['arxiv:other','url:openreview.net/forum?id=id0']));")
        self.open('offline')
        self.assertEqual(self.page.locator('#marked-count').inner_text(), '1')
        payload = json.loads(self.backup())
        self.assertEqual(len(payload['bookmarks']), 2)
        self.assertEqual(payload['order'][0], 'arxiv:other')
        self.assertEqual(payload['bookmarks'][1]['title'], 'Paper 000')
        self.assertEqual(self.errors, [])

    def test_invalid_import_is_rejected_without_changing_existing_bookmarks(self):
        self.open('offline')
        self.page.locator('.bookmark-button').first.click()
        before = self.page.evaluate("localStorage.getItem('arxiv-reader-bookmark-data-v2')")
        invalid = dict(format='llm-paper-filter-bookmarks', version=1,
                       bookmarks=[dict(key='bad', title='bad', authors=[], url='javascript:alert(1)')], order=['bad'])
        self.restore(json.dumps(invalid).encode())
        self.assertIn('恢复失败', self.page.locator('#backup-status').inner_text())
        self.assertEqual(self.page.evaluate("localStorage.getItem('arxiv-reader-bookmark-data-v2')"), before)
        self.assertEqual(self.errors, [])

    def test_hidden_score_four_favorite_is_restored_and_shown_in_marked_view(self):
        self.open()
        payload = dict(format='llm-paper-filter-bookmarks', version=1,
                       bookmarks=[dict(key='url:openreview.net/forum?id=id115', title='Paper 115', authors=[],
                                       url='https://openreview.net/forum?id=id115')],
                       order=['url:openreview.net/forum?id=id115'])
        self.restore(json.dumps(payload).encode())
        self.page.check('#marked-only'); self.wait_cards(1)
        self.assertEqual(self.page.locator('.paper-card').get_attribute('data-title'), 'Paper 115')

    def test_storage_unavailable_keeps_warning_and_allows_file_backup(self):
        self.page.add_init_script("Storage.prototype.setItem=function(){throw new Error('storage disabled')}")
        self.open('offline')
        self.page.locator('.bookmark-button').first.click()
        self.assertIn('备份收藏', self.page.locator('#bookmark-status').inner_text())
        self.assertEqual(len(json.loads(self.backup())['bookmarks']), 1)
        self.assertFalse(self.page.locator('#bookmark-status').is_hidden())
        self.assertEqual(self.errors, [])

    def test_single_file_offline_reader_can_backup_and_restore(self):
        self.page.goto((self.root / 'offline/html_reports/report_2027-01-01.html').as_uri())
        self.wait_cards(3)
        self.page.locator('.bookmark-button').first.click()
        backup = self.backup()
        self.page.evaluate('localStorage.clear()')
        self.page.reload(); self.wait_cards(3)
        self.restore(backup)
        self.assertEqual(self.page.locator('#marked-count').inner_text(), '1')
        self.assertEqual(self.errors, [])

    def test_failed_page_request_can_retry_without_resetting_filter(self):
        self.open()
        self.page.route('**/data_*/1.json', lambda route: route.abort())
        self.page.click('#next-page')
        self.page.wait_for_selector('#retry-page:not([hidden])')
        self.assertIn('加载失败', self.page.locator('#load-status').inner_text())
        self.page.unroute('**/data_*/1.json')
        self.page.click('#retry-page')
        self.page.wait_for_selector('.paper-card[data-title="Paper 050"]')
        self.assertIn('第 2 /', self.page.locator('#page-position').inner_text())
        self.assertEqual(self.errors, [])

    def test_bookmarking_keeps_reading_position_open_abstract_and_current_page(self):
        self.open()
        dimensions = self.page.evaluate("""() => ['backup-bookmarks','restore-bookmarks','expand-all','reset-filters'].map(id => {
            const button=document.getElementById(id), style=getComputedStyle(button);
            return [button.getBoundingClientRect().height, style.fontSize, style.padding];
        })""")
        self.assertTrue(all(value == dimensions[0] for value in dimensions))
        self.page.click('#next-page')
        self.page.wait_for_selector('.paper-card[data-title="Paper 050"]')
        card = self.page.locator('.paper-card[data-title="Paper 070"]')
        card.locator('.abstract summary').click()
        button = card.locator('.bookmark-button')
        button.scroll_into_view_if_needed()
        card.evaluate("card => card.dataset.positionProbe='kept'")
        position = self.page.evaluate('scrollY')
        self.assertGreater(position, 1000)
        for marked in ('true', 'false'):
            button.click()
            self.assertEqual(button.get_attribute('aria-pressed'), marked)
            self.assertAlmostEqual(self.page.evaluate('scrollY'), position, delta=2)
            self.assertEqual(card.get_attribute('data-position-probe'), 'kept')
            self.assertTrue(card.locator('.abstract').evaluate('abstract => abstract.open'))
            self.assertIn('第 2 /', self.page.locator('#page-position').inner_text())
        self.assertEqual(self.errors, [])

    def test_favorite_actions_source_return_and_removing_last_favorite(self):
        self.open('daily')
        self.assertTrue(self.page.locator('.outline-return').is_hidden())
        self.page.locator('.bookmark-button').click()
        self.page.check('#marked-only'); self.wait_cards(1)
        row = self.page.locator('.bookmark-row')
        controls = row.locator('.order-buttons > *')
        self.assertEqual(controls.evaluate_all("nodes=>nodes.map(node=>node.getAttribute('aria-label'))"),
                         ['在 arXiv 查看摘要', '复制这篇论文的 Markdown metadata',
                          '上移这篇论文', '下移这篇论文', '取消收藏这篇论文'])
        source = row.locator('.paper-source-link')
        self.assertEqual(source.get_attribute('href'), 'https://arxiv.org/abs/2610.00001v2')
        self.assertEqual(source.get_attribute('target'), '_blank')
        self.assertIn('noopener', source.get_attribute('rel'))
        for width in (390, 320):
            self.page.set_viewport_size(dict(width=width, height=844))
            boxes = controls.evaluate_all('nodes=>nodes.map(node=>{const box=node.getBoundingClientRect();return {x:box.x,y:box.y,right:box.right}})')
            self.assertTrue(all(abs(box['y']-boxes[0]['y']) < 1 for box in boxes))
            self.assertEqual([box['x'] for box in boxes], sorted(box['x'] for box in boxes))
            self.assertGreaterEqual(boxes[0]['x'], 0)
            self.assertLessEqual(boxes[-1]['right'], width)
            self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth > innerWidth'))
        row.locator('.bookmark-title').click()
        self.page.locator('.outline-return').click()
        self.assertTrue(row.locator('.bookmark-title').evaluate('link=>document.activeElement===link'))
        self.assertAlmostEqual(self.page.locator('#bookmarked-outline').bounding_box()['y'], 16, delta=2)
        row.locator('.remove-bookmark').click(); self.wait_cards(0)
        self.assertTrue(self.page.locator('#bookmarked-outline').is_hidden())
        self.assertEqual(self.page.locator('#marked-count').inner_text(), '0')
        self.page.uncheck('#marked-only'); self.wait_cards(1)
        self.assertEqual(self.page.locator('.bookmark-button').get_attribute('aria-pressed'), 'false')
        self.page.reload(); self.wait_cards(1)
        self.assertEqual(self.page.locator('#marked-count').inner_text(), '0')
        self.assertEqual(self.errors, [])

    def test_return_to_outline_across_pages_and_remove_last_item_on_page(self):
        self.open()
        records = [dict(key=f'url:openreview.net/forum?id=id{i}', title=f'Paper {i:03d}', authors=['Author'],
                        url=f'https://openreview.net/forum?id=id{i}') for i in range(51)]
        self.restore(json.dumps(dict(format='llm-paper-filter-bookmarks', version=1,
                                     bookmarks=records, order=[r['key'] for r in records])).encode())
        self.page.check('#marked-only'); self.wait_cards(50)
        row = self.page.locator('.bookmark-row').last
        self.assertEqual(row.locator('.paper-source-link').get_attribute('href'), 'https://openreview.net/forum?id=id50')
        self.assertEqual(row.locator('.paper-source-link').get_attribute('aria-label'), '在 OpenReview 查看论文')
        row.locator('.bookmark-title').click()
        self.page.wait_for_selector('.paper-card[data-title="Paper 050"]')
        self.wait_cards(1)
        self.assertIn('第 2 / 2 页', self.page.locator('#page-position').inner_text())
        self.page.locator('.outline-return').click()
        row = self.page.locator('.bookmark-row').last
        self.assertTrue(row.locator('.bookmark-title').evaluate('link=>document.activeElement===link'))
        self.assertTrue(row.is_visible())
        row.locator('.remove-bookmark').click(); self.wait_cards(50)
        self.assertEqual(self.page.locator('.bookmark-row').count(), 50)
        self.assertEqual(self.page.locator('#marked-count').inner_text(), '50')
        self.assertIn('第 1 / 1 页', self.page.locator('#page-position').inner_text())
        self.assertEqual(self.page.locator('.paper-card[data-title="Paper 050"]').count(), 0)
        self.assertEqual(self.errors, [])


if __name__ == '__main__':
    unittest.main()
