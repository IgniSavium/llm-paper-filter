(() => {
 'use strict';
 const byId = id => document.getElementById(id);
 const data = JSON.parse(byId('reader-data').textContent);
 const paged = data !== null;
 const container = byId('papers');
 const items = paged ? data.papers : Array.from(container.querySelectorAll('.paper-card'), card => {
  const metadata = JSON.parse(card.dataset.paperMetadata);
  return {id:card.id, key:card.dataset.bookmarkKey, title:card.dataset.title,
   authors:metadata.authors || [], url:metadata.url, scores:JSON.parse(card.dataset.categoryScores),
   score:Number(card.dataset.score), new:card.dataset.new === 'true', card};
 });
 const itemByKey = new Map(items.map(item => [item.key, item]));
 let currentPage = 1, pageSize = data?.pageSize || 50, renderVersion = 0;
 let shownItems = [], currentCards = [], globalMode = 'both', draggedKey = null;
 const chunks = new Map();
 const paperModes = new Map();
 const legacyKey = 'arxiv-reader-bookmarks-v1';
 const legacyOrderKey = 'arxiv-reader-bookmark-order-v1';
 const storageKey = 'arxiv-reader-bookmark-data-v2';
 let bookmarks = new Map(), bookmarkOrder = [], storageWarning = '';

 function bookmarkRecord(item, read=item.read === true) {
  return {key:item.key, title:item.title, authors:item.authors || [], url:item.url || '', read};
 }
 function validateBackup(value) {
  if (!value || value.format !== 'llm-paper-filter-bookmarks' || value.version !== 1 ||
      !Array.isArray(value.bookmarks) || !Array.isArray(value.order)) throw new Error('不是支持的收藏备份文件。');
  const records = new Map();
  for (const item of value.bookmarks) {
   if (!item || typeof item.key !== 'string' || !item.key.trim() || item.key.length > 2048 ||
       typeof item.title !== 'string' || !Array.isArray(item.authors) ||
       item.authors.some(author => typeof author !== 'string') || typeof item.url !== 'string' ||
       (item.read !== undefined && typeof item.read !== 'boolean') ||
       (item.url && !/^https?:\/\//i.test(item.url))) throw new Error('备份包含无效的收藏记录。');
   if (records.has(item.key)) throw new Error('备份包含重复的收藏标识。');
   records.set(item.key, bookmarkRecord(item));
  }
  if (value.order.some(key => typeof key !== 'string' || !records.has(key))) throw new Error('备份的阅读顺序无效。');
  const order = Array.from(new Set(value.order));
  for (const key of records.keys()) if (!order.includes(key)) order.push(key);
  return {records, order};
 }
 function ensureBookmarkOrder() {
  bookmarkOrder = bookmarkOrder.filter(key => bookmarks.has(key));
  for (const key of bookmarks.keys()) {
   if (!bookmarkOrder.includes(key)) bookmarkOrder.push(key);
   if (itemByKey.has(key)) bookmarks.set(key, bookmarkRecord(itemByKey.get(key), bookmarks.get(key).read === true));
  }
 }
 function backupPayload() {
  ensureBookmarkOrder();
  return {format:'llm-paper-filter-bookmarks', version:1, exportedAt:new Date().toISOString(),
   bookmarks:bookmarkOrder.map(key => bookmarks.get(key)), order:[...bookmarkOrder]};
 }
 function storageStatus() {
  byId('bookmark-status').textContent = storageWarning;
  byId('bookmark-status').hidden = !storageWarning;
 }
 function loadBookmarks() {
  try {
   const saved = localStorage.getItem(storageKey);
   if (saved) {
    const loaded = validateBackup(JSON.parse(saved));
    bookmarks = loaded.records; bookmarkOrder = loaded.order;
   } else {
    const keys = JSON.parse(localStorage.getItem(legacyKey) || '[]');
    const order = JSON.parse(localStorage.getItem(legacyOrderKey) || '[]');
    bookmarks = new Map((Array.isArray(keys) ? keys : []).filter(key => typeof key === 'string').map(key =>
     [key, {key, title:itemByKey.get(key)?.title || key, authors:[], url:''}]));
    bookmarkOrder = Array.from(new Set((Array.isArray(order) ? order : []).filter(key => typeof key === 'string')));
   }
   storageWarning = '';
  } catch (_) {
   storageWarning = '无法读取浏览器本地收藏；本次操作仍可备份为文件，请使用“恢复收藏”导入已有备份。';
  }
  ensureBookmarkOrder(); storageStatus();
 }
 function saveBookmarks() {
  ensureBookmarkOrder();
  try {
   localStorage.setItem(storageKey, JSON.stringify(backupPayload()));
   // Continue supporting existing single-file readers and their stable keys.
   localStorage.setItem(legacyKey, JSON.stringify(Array.from(bookmarks.keys())));
   localStorage.setItem(legacyOrderKey, JSON.stringify(bookmarkOrder));
   storageWarning = '';
  } catch (_) {
   storageWarning = '浏览器无法保存收藏；关闭页面前请点“备份收藏”下载文件。';
  }
  storageStatus();
 }
 function message(text) { byId('backup-status').textContent = text; byId('backup-status').hidden = false; }
 loadBookmarks();
 if (bookmarks.size) saveBookmarks();
 byId('backup-bookmarks').addEventListener('click', () => {
  const payload = backupPayload();
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'}));
  const link = document.createElement('a');
  const day = new Intl.DateTimeFormat('en-CA', {timeZone:'Asia/Shanghai', year:'numeric', month:'2-digit', day:'2-digit'}).format(new Date());
  link.href = url; link.download = `paper-bookmarks-${day}.json`;
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  message(`已备份全部 ${payload.bookmarks.length} 篇收藏、阅读顺序及已读状态。清空浏览器数据后，可从此文件恢复。`);
 });
 byId('restore-bookmarks').addEventListener('click', () => byId('bookmark-file').click());
 byId('bookmark-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
   if (file.size > 20 * 1024 * 1024) throw new Error('备份文件超过 20 MB。');
   const loaded = validateBackup(JSON.parse(await file.text()));
   const before = bookmarks.size;
   for (const [key, record] of loaded.records) {
    if (!bookmarks.has(key)) bookmarks.set(key, record);
    else if (record.read) bookmarks.get(key).read=true;
   }
   // Existing order wins for overlaps; restored records keep their backup order.
   for (const key of loaded.order) if (!bookmarkOrder.includes(key)) bookmarkOrder.push(key);
   saveBookmarks(); await refreshBookmarks();
   const matched = Array.from(loaded.records.keys()).filter(key => itemByKey.has(key)).length;
   message(`已合并恢复 ${bookmarks.size-before} 篇新收藏，${matched} 篇位于本报告；全部共 ${bookmarks.size} 篇。阅读顺序已保留，已读状态已合并。`);
  } catch (error) { message(`恢复失败：${error.message} 现有收藏未被替换。`); }
  finally { event.target.value = ''; }
 });
 window.addEventListener('storage', event => {
  if ([storageKey, legacyKey, legacyOrderKey, null].includes(event.key)) { loadBookmarks(); refreshBookmarks(); }
 });

 const canonical = new Map(Array.from(document.querySelectorAll('[data-direction-mode]'), input =>
  [input.dataset.directionMode.toLowerCase(), input.dataset.directionMode]));
 function compileExpression(source) {
  if (!source.trim()) return () => true;
  if (/[^A-Za-z0-9()\s=!<>]/.test(source)) throw new Error('请使用方向名称、分数比较、AND / OR / NOT 和括号。');
  const tokens = source.match(/[A-Za-z]+|>=|<=|!=|[=<>]|[0-9]+|[()]/g) || []; let position = 0;
  const peek = () => tokens[position]?.toUpperCase();
  function primary() {
   const token = tokens[position++];
   if (token === '(') { const node = or(); if (tokens[position++] !== ')') throw new Error('括号未配对。'); return node; }
   const category = canonical.get((token || '').toLowerCase()); if (!category) throw new Error('请填写有效方向名称。');
   const operator = tokens[position];
   if (['=','>=','<=','>','<','!='].includes(operator)) {
    position++; const literal = tokens[position++], target = Number(literal);
    if (!/^[1-5]$/.test(literal || '')) throw new Error('方向分数必须是 1–5 的整数。');
    return scores => { const value = scores[category]; switch (operator) {
     case '=':return value===target; case '>=':return value>=target; case '<=':return value<=target;
     case '>':return value>target; case '<':return value<target; case '!=':return value!==target;
    }};
   }
   return scores => scores[category] >= 4;
  }
  function not() { if (peek()==='NOT') { position++; const node=not(); return scores=>!node(scores); } return primary(); }
  function and() { let node=not(); while (peek()==='AND') { position++; const left=node,right=not(); node=scores=>left(scores)&&right(scores); } return node; }
  function or() { let node=and(); while (peek()==='OR') { position++; const left=node,right=and(); node=scores=>left(scores)||right(scores); } return node; }
  const node=or(); if (position!==tokens.length) throw new Error('类别之间需要 AND 或 OR。'); return node;
 }
 function directionExpression() {
  const include=[], exclude=[];
  for (const input of document.querySelectorAll('[data-direction-mode]')) {
   if (input.value==='ignore') continue;
   const category=input.dataset.directionMode, scores=byId(`${category}-score`).value.split(',');
   const term=scores.length===1 ? `${category}=${scores[0]}` : `(${scores.map(score=>`${category}=${score}`).join(' OR ')})`;
   (input.value==='exclude' ? exclude : include).push(term);
  }
  let expression=include.join(byId('category-logic').value==='and' ? ' AND ' : ' OR ');
  if (exclude.length) { if (expression) expression=`(${expression}) AND `; expression+=`NOT (${exclude.join(' OR ')})`; }
  return expression;
 }
 async function cardFor(item) {
  if (item.card) return item.card;
  if (!chunks.has(item.chunk)) {
   const pending=fetch(item.chunk).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
   }).catch(error => { chunks.delete(item.chunk); throw error; });
   chunks.set(item.chunk, pending);
   // Bound browser memory while allowing back/forward page navigation.
   if (chunks.size>8) chunks.delete(chunks.keys().next().value);
  }
  const rows=await chunks.get(item.chunk);
  if (!Array.isArray(rows) || typeof rows[item.slot]!=='string') throw new Error('论文分页数据无效');
  const template=document.createElement('template'); template.innerHTML=rows[item.slot];
  const card=template.content.firstElementChild;
  if (!card || card.dataset.bookmarkKey!==item.key) throw new Error('论文分页标识不匹配');
  return card;
 }
 function renderMath(root, includeAbstract=false) {
  for (const element of root.querySelectorAll('.math-source:not([data-rendered])')) {
   if (!includeAbstract && element.closest('.abstract')) continue;
   try { katex.render(element.dataset.tex, element, {displayMode:element.dataset.display==='true', throwOnError:false, strict:'ignore', trust:false}); element.dataset.rendered='true'; } catch (_) {}
  }
 }
 function updateExpanded() {
  const abstracts=currentCards.map(card=>card.querySelector('.abstract'));
  const allOpen=abstracts.length>0 && abstracts.every(abstract=>abstract.open);
  byId('expand-all').textContent=allOpen ? '收起本页摘要' : (paged ? '展开本页摘要' : '展开全部摘要');
  byId('expand-all').setAttribute('aria-expanded',String(allOpen)); byId('expand-all').disabled=!abstracts.length;
 }
 function updateBookmarkControls() {
  byId('marked-count').textContent=items.filter(item=>bookmarks.has(item.key)).length;
  byId('backup-bookmarks').title=`备份所有报告的 ${bookmarks.size} 篇收藏及顺序`;
  for (const card of currentCards) {
   const marked=bookmarks.has(card.dataset.bookmarkKey), button=card.querySelector('.bookmark-button');
   button.setAttribute('aria-pressed',String(marked)); button.setAttribute('aria-label',marked?'取消收藏这篇论文':'收藏这篇论文');
   button.title=marked?'取消收藏这篇论文':'收藏这篇论文'; button.querySelector('.bookmark-icon').textContent=marked?'★':'☆';
   const returnButton=card.querySelector('.outline-return');
   if (returnButton) returnButton.hidden=!marked || !byId('marked-only').checked || byId('bookmarked-outline').hidden;
  }
 }
 function refreshBookmarks() {
  updateBookmarkControls();
  // Ordinary bookmarking changes no filter results: keep the existing cards,
  // open abstracts, keyboard focus and viewport instead of rebuilding the page.
  return byId('marked-only').checked ? filter(false, true) : Promise.resolve();
 }
 async function filter(resetPage=true, preserveScroll=false) {
  const previousScroll=preserveScroll ? [scrollX, scrollY] : null;
  const version=++renderVersion;
  if (resetPage) currentPage=1;
  const advanced=byId('advanced-filter').checked;
  const expression=advanced ? byId('category-expression').value.trim() : directionExpression();
  if (!advanced) byId('category-expression').value=expression;
  const markedOnly=byId('marked-only').checked;
  let predicate, error='';
  try { predicate=compileExpression(expression); } catch (e) { error=e.message; predicate=()=>false; }
  byId('expression-error').textContent=error; byId('expression-error').hidden=!error;
  const description=markedOnly ? '本报告的全部已收藏论文（忽略方向筛选）' : expression || '全部候选论文';
  byId('filter-expression').textContent=description; byId('category-summary').textContent=description;
  shownItems=items.filter(item=>markedOnly ? bookmarks.has(item.key) : (item.scores || !expression) && predicate(item.scores || {}));
  if (markedOnly) { ensureBookmarkOrder(); const ranks=new Map(bookmarkOrder.map((key,i)=>[key,i])); shownItems.sort((a,b)=>ranks.get(a.key)-ranks.get(b.key)); }
  const visible=shownItems.length;
  byId('visible-count').textContent=visible;
  byId('five-count').textContent=shownItems.filter(item=>item.score===5).length;
  byId('four-count').textContent=shownItems.filter(item=>item.score===4).length;
  const fresh=shownItems.filter(item=>item.new).length;
  byId('new-count').textContent=fresh; byId('update-count').textContent=visible-fresh;
  updateBookmarkControls();
  byId('empty-state').hidden=visible!==0;
  const pages=Math.max(1,Math.ceil(visible/pageSize)); currentPage=Math.max(1,Math.min(currentPage,pages));
  byId('pagination').hidden=!paged;
  byId('page-position').textContent=`第 ${currentPage} / ${pages} 页 · 共 ${visible} 篇`;
  byId('previous-page').disabled=currentPage===1; byId('next-page').disabled=currentPage===pages;
  const offset=paged ? (currentPage-1)*pageSize : 0;
  const pageItems=paged ? shownItems.slice(offset,offset+pageSize) : shownItems;
  renderOutline(markedOnly ? shownItems : []);
  byId('retry-page').hidden=true; byId('load-status').hidden=!paged || !pageItems.length;
  byId('load-status').textContent='正在加载本页论文…';
  if (paged) { container.replaceChildren(); currentCards=[]; }
  try {
   const cards=await Promise.all(pageItems.map(cardFor));
   if (version!==renderVersion) return;
   if (!paged) for (const item of items) item.card.hidden=true;
   currentCards=cards;
   cards.forEach((card,index)=>{
    const item=pageItems[index]; card.hidden=false;
    for (const position of card.querySelectorAll('.result-position')) position.textContent=`${offset+index+1}/${visible}`;
    const abstract=card.querySelector('.abstract'), mode=paperModes.get(item.key) || globalMode;
    abstract.dataset.mode=mode; abstract.querySelector('.paper-mode').value=mode;
    if (!abstract.dataset.readerBound) {
     abstract.addEventListener('toggle',()=>{if(abstract.open)renderMath(abstract,true);updateExpanded();});
     abstract.dataset.readerBound='true';
    }
    container.append(card); renderMath(card); if (abstract.open) renderMath(abstract,true);
   });
   byId('load-status').hidden=true; updateExpanded(); updateBookmarkControls();
   if (previousScroll) window.scrollTo(...previousScroll);
  } catch (error) {
   if (version!==renderVersion) return;
   byId('load-status').hidden=false; byId('load-status').textContent=`本页加载失败：${error.message}。请重试。`;
   byId('retry-page').hidden=false;
  }
 }
 container.addEventListener('click',event=>{
  const returnButton=event.target.closest('.outline-return');
  if (returnButton) {
   const outline=byId('bookmarked-outline');
   if (outline.hidden) return;
   const key=returnButton.closest('.paper-card').dataset.bookmarkKey;
   const row=Array.from(byId('bookmark-list').children).find(item=>item.dataset.key===key);
   outline.scrollIntoView({block:'start'});
   row?.scrollIntoView({block:'nearest'});
   row?.querySelector('.bookmark-title')?.focus({preventScroll:true});
   return;
  }
  const button=event.target.closest('.bookmark-button'); if (!button) return;
  const key=button.closest('.paper-card').dataset.bookmarkKey;
  if (bookmarks.has(key)) bookmarks.delete(key); else bookmarks.set(key,bookmarkRecord(itemByKey.get(key)));
  saveBookmarks(); refreshBookmarks();
 });
 byId('marked-only').addEventListener('change',()=>filter());
 for (const input of document.querySelectorAll('[data-direction-mode],[data-direction-score]')) input.addEventListener('change',()=>filter());
 byId('category-logic').addEventListener('change',()=>filter()); byId('category-expression').addEventListener('input',()=>filter());
 byId('advanced-filter').addEventListener('change',()=>{const advanced=byId('advanced-filter').checked;byId('basic-filters').disabled=advanced;byId('expression-panel').hidden=!advanced;filter();});
 byId('reset-filters').addEventListener('click',()=>{
  for (const input of document.querySelectorAll('[data-direction-mode]')) input.value=['MechInterp','CompGen'].includes(input.dataset.directionMode)?'include':'ignore';
  for (const input of document.querySelectorAll('[data-direction-score]')) input.value='5';
  byId('category-logic').value='or';byId('marked-only').checked=false;byId('advanced-filter').checked=false;byId('basic-filters').disabled=false;byId('expression-panel').hidden=true;filter();
 });
 byId('previous-page').addEventListener('click',()=>{currentPage--;filter(false);});
 byId('next-page').addEventListener('click',()=>{currentPage++;filter(false);});
 byId('page-size').value=String(pageSize);
 byId('page-size').addEventListener('change',()=>{pageSize=Number(byId('page-size').value);filter();});
 byId('retry-page').addEventListener('click',()=>filter(false));
 const theme=byId('theme-toggle');
 function updateTheme(){const dark=document.documentElement.dataset.theme==='dark';theme.textContent=dark?'☀ 浅色模式':'☾ 深色模式';theme.setAttribute('aria-pressed',String(dark));}
 updateTheme();theme.addEventListener('click',()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('arxiv-reader-theme',document.documentElement.dataset.theme);}catch(_){}updateTheme();});
 byId('category-controls').open=innerWidth>760;
 function clearMatches(abstract){for(const sentence of abstract.querySelectorAll('.matched'))sentence.classList.remove('matched');}
 function highlight(event){const sentence=event.target.closest('.sentence'),abstract=sentence?.closest('.abstract');if(!abstract||abstract.dataset.mode!=='both')return;clearMatches(abstract);for(const match of abstract.querySelectorAll(`.sentence[data-pair="${sentence.dataset.pair}"]`))match.classList.add('matched');}
 container.addEventListener('pointerover',highlight);container.addEventListener('focusin',highlight);
 for (const type of ['pointerout','focusout']) container.addEventListener(type,event=>{const abstract=event.target.closest('.abstract');if(abstract)clearMatches(abstract);});
 for(const button of document.querySelectorAll('.segmented button'))button.addEventListener('click',()=>{globalMode=button.dataset.mode;paperModes.clear();for(const card of currentCards){const abstract=card.querySelector('.abstract');abstract.dataset.mode=globalMode;abstract.querySelector('.paper-mode').value=globalMode;clearMatches(abstract);}for(const other of document.querySelectorAll('.segmented button'))other.setAttribute('aria-pressed',String(other===button));});
 container.addEventListener('change',event=>{if(!event.target.matches('.paper-mode'))return;const abstract=event.target.closest('.abstract');abstract.dataset.mode=event.target.value;paperModes.set(abstract.closest('.paper-card').dataset.bookmarkKey,event.target.value);clearMatches(abstract);for(const button of document.querySelectorAll('.segmented button'))button.setAttribute('aria-pressed','false');});
 byId('expand-all').addEventListener('click',()=>{const abstracts=currentCards.map(card=>card.querySelector('.abstract'));const expand=abstracts.some(abstract=>!abstract.open);for(const abstract of abstracts){abstract.open=expand;if(expand)renderMath(abstract,true);}updateExpanded();});

 function moveBookmark(key,target,after=false){if(key===target)return;ensureBookmarkOrder();bookmarkOrder=bookmarkOrder.filter(value=>value!==key);const index=bookmarkOrder.indexOf(target);bookmarkOrder.splice(index+(after?1:0),0,key);saveBookmarks();return filter(false);}
 function renderOutline(shown){
  draggedKey=null;byId('bookmarked-outline').hidden=!shown.length;byId('outline-count').textContent=`${shown.length} 篇`;
  const list=byId('bookmark-list');list.replaceChildren();
  shown.forEach((item,index)=>{
   const row=document.createElement('li');row.className='bookmark-row';row.dataset.key=item.key;row.draggable=true;
   const handle=document.createElement('span');handle.className='drag-handle';handle.textContent='⋮⋮';
   const position=document.createElement('span');position.className='outline-index';position.textContent=index+1;
   const readToggle=document.createElement('input');readToggle.type='checkbox';readToggle.className='read-toggle';readToggle.checked=bookmarks.get(item.key)?.read === true;
   readToggle.setAttribute('aria-label',`已读：${item.title}`);readToggle.title='勾选表示已读';row.classList.toggle('is-read',readToggle.checked);
   readToggle.addEventListener('change',()=>{bookmarks.get(item.key).read=readToggle.checked;row.classList.toggle('is-read',readToggle.checked);saveBookmarks();});
   const link=document.createElement('a');link.className='bookmark-title';link.textContent=item.title;link.draggable=false;
   let sourceLabel='暂无论文原文链接';
   if (/^https?:\/\//i.test(item.url)) {
    link.href=item.url;link.target='_blank';link.rel='noopener noreferrer';
    const host=new URL(item.url).hostname;
    sourceLabel=['arxiv.org','www.arxiv.org','export.arxiv.org'].includes(host)?'在 arXiv 查看摘要':(host==='openreview.net'?'在 OpenReview 查看论文':'打开论文原文');
   } else { link.setAttribute('aria-disabled','true');link.tabIndex=-1; }
   link.title=sourceLabel;
   const actions=document.createElement('div');actions.className='order-buttons';
   const infoButton=document.createElement('button');infoButton.type='button';infoButton.className='paper-info-link';infoButton.title='查看报告内的论文信息';infoButton.setAttribute('aria-label',infoButton.title);
   infoButton.innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h4"/></svg>';
   infoButton.addEventListener('click',async()=>{if(paged){currentPage=Math.floor(index/pageSize)+1;await filter(false);}const card=document.getElementById(item.id);card?.scrollIntoView({block:'start'});card?.querySelector('.outline-return')?.focus({preventScroll:true});});actions.append(infoButton);
   const exportButton=document.createElement('button');exportButton.type='button';exportButton.textContent='↗';exportButton.title='复制这篇论文的 Markdown metadata';
   exportButton.setAttribute('aria-label',exportButton.title);
   exportButton.addEventListener('click',async()=>{try{const card=await cardFor(item);await copyMarkdown(paperMetadataMarkdown(card),'已复制论文 metadata。',exportButton);}catch(error){message(`加载失败：${error.message}`);}});actions.append(exportButton);
   for(const [direction,text,disabled] of [['up','↑',index===0],['down','↓',index===shown.length-1]]){
    const button=document.createElement('button');button.type='button';button.textContent=text;button.disabled=disabled;button.dataset.direction=direction;button.setAttribute('aria-label',direction==='up'?'上移这篇论文':'下移这篇论文');
    button.addEventListener('click',async()=>{await moveBookmark(item.key,shown[index+(direction==='up'?-1:1)].key,direction==='down');const replacement=Array.from(byId('bookmark-list').children).find(row=>row.dataset.key===item.key);const focus=replacement?.querySelector(`button[data-direction="${direction}"]:not(:disabled)`)||replacement?.querySelector('a');focus?.focus({preventScroll:true});});actions.append(button);
   }
   const removeButton=document.createElement('button');removeButton.type='button';removeButton.className='remove-bookmark';removeButton.title='取消收藏这篇论文';removeButton.setAttribute('aria-label',removeButton.title);
   removeButton.innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>';
   removeButton.addEventListener('click',async()=>{bookmarks.delete(item.key);saveBookmarks();await refreshBookmarks();const rows=byId('bookmark-list').children;const focus=rows[Math.min(index,rows.length-1)]?.querySelector('.bookmark-title')||byId('marked-only');focus.focus({preventScroll:true});});actions.append(removeButton);
   row.append(handle,position,readToggle,link,actions);list.append(row);
  });
 }
 function clearDrag(){for(const row of byId('bookmark-list').children)row.classList.remove('dragging','drop-before','drop-after');}
 byId('bookmark-list').addEventListener('dragstart',event=>{const row=event.target.closest('.bookmark-row');if(!row)return;draggedKey=row.dataset.key;event.dataTransfer.setData('text/plain',draggedKey);event.dataTransfer.effectAllowed='move';row.classList.add('dragging');});
 byId('bookmark-list').addEventListener('dragover',event=>{const row=event.target.closest('.bookmark-row');if(!row||!draggedKey||row.dataset.key===draggedKey)return;event.preventDefault();event.dataTransfer.dropEffect='move';clearDrag();row.classList.add(event.clientY>row.getBoundingClientRect().y+row.getBoundingClientRect().height/2?'drop-after':'drop-before');});
 byId('bookmark-list').addEventListener('drop',event=>{const row=event.target.closest('.bookmark-row');if(!row||!draggedKey)return;event.preventDefault();const key=draggedKey;draggedKey=null;clearDrag();moveBookmark(key,row.dataset.key,event.clientY>row.getBoundingClientRect().y+row.getBoundingClientRect().height/2);});
 byId('bookmark-list').addEventListener('dragend',()=>{draggedKey=null;clearDrag();});
 const clean=value=>String(value??'').replace(/\s+/g,' ').trim();
 const escapeMarkdown=value=>Array.from(clean(value),char=>'`*_{}[]<>'.includes(char)||char.charCodeAt(0)===92?String.fromCharCode(92)+char:char).join('');
 byId('export-bookmarks').addEventListener('click',async()=>{
  ensureBookmarkOrder();const selected=bookmarkOrder.filter(key=>itemByKey.has(key)).map(key=>itemByKey.get(key));
  const text=selected.map(item=>`- [${escapeMarkdown(item.title)}](<${item.url}>)${item.authors.length?' — '+escapeMarkdown(item.authors.join(', ')):''}`).join('\n\n')+'\n';
  await copyMarkdown(text,`已复制本报告 ${selected.length} 篇收藏的 Markdown。`,byId('export-bookmarks'));
 });
 function paperMetadataMarkdown(card){
  const data=JSON.parse(card.dataset.paperMetadata),title=clean(data.title);
  const slug=Array.from(title).filter(char=>!'/：:*?"<>|'.includes(char)&&char.charCodeAt(0)!==92).join('').replace(/\s+/g,'_');
  const dateParts=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(new Date());
  const part=type=>dateParts.find(value=>value.type===type).value;
  const stamp=value=>value?clean(value).replace('T',' ').replace(/Z$/,'+00:00'):'未记录';
  return ['# '+escapeMarkdown(title),'',slug,'','---','','## 🔥INFO','',`**Blog**: ${part('year')}/${part('month')}/${part('day')} by IgniSavium`,'',
   '- **Title**: '+escapeMarkdown(title),'- **Authors**: '+escapeMarkdown((data.authors||[]).join(', ')),
   '- **Published**: '+stamp(data.published),'- **Updated**: '+stamp(data.updated),'- **Categories**: '+escapeMarkdown((data.categories||[]).join(', ')),
   '- **Comment**: '+escapeMarkdown(data.comment??'None'),'- **Journal Reference**: '+escapeMarkdown(data.journal_ref??'None'),'- **URL**: '+clean(data.url),'','🥜**TLDR**:','','---',''].join('\n');
 }
 async function copyMarkdown(text,messageText,button){
  const status=byId('bookmark-copy-status'),fallback=byId('bookmark-markdown');button.disabled=true;status.hidden=false;fallback.hidden=true;
  try{await navigator.clipboard.writeText(text);status.textContent=messageText;}catch(_){fallback.value=text;fallback.hidden=false;fallback.focus();fallback.select();status.textContent='无法自动访问剪贴板，文本已选中，请按 Ctrl+C（Mac：⌘C）复制。';}finally{button.disabled=false;}
 }
 filter();
})();
