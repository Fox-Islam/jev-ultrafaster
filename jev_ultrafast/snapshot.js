(() => {
  if (!document.body) return null;
  const cache = window.__jevFast ||= {ids:new WeakMap(), nodes:new Map(), next:1};
  const identity = e => {
    if (!cache.ids.has(e)) cache.ids.set(e,cache.next++);
    const id=cache.ids.get(e); cache.nodes.set(id,e); return id;
  };
  for (const [id,e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);
  // Open shadow roots hold real controls - an injected toolbar mounts its whole UI in one - and
  // querySelectorAll stops at their boundary, so every query walks into them as well.
  const roots = () => {
    const found=[document];
    for (let i=0; i<found.length; i++)
      for (const e of found[i].querySelectorAll('*')) if (e.shadowRoot) found.push(e.shadowRoot);
    return found;
  };
  const allRoots=roots();
  const deep = selector => allRoots.flatMap(root=>[...root.querySelectorAll(selector)]);
  // closest() stops at a shadow boundary, so a control inside a hidden host would read as shown.
  const across = (e, selector) => {
    for (let n=e; n; n=n.parentElement || n.getRootNode()?.host) if (n.matches?.(selector)) return n;
    return null;
  };
  const safe = e => !['password','file','hidden'].includes(e.type);
  const shown = e => !across(e,'[aria-hidden="true"],[inert]') && e.checkVisibility({checkVisibilityCSS:true});
  const visible = e => shown(e) && e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  // Shown but transparent: the usual way a row reveals its delete or edit control under the
  // pointer. Such a control is in the document and works once hovered, so it is offered with
  // that said, and the executor hovers it before clicking.
  const hoverReveals = e => shown(e) && !visible(e);
  const selectorRoles=['button','link','checkbox','radio','switch','tab','menuitem','menuitemradio',
    'menuitemcheckbox','treeitem','option','gridcell','combobox','textbox','searchbox','spinbutton'];
  const selector='a[href],button,input,textarea,select,summary,[contenteditable="true"],'+
    selectorRoles.map(role=>'[role="'+role+'"]').join(',');
  // What an icon-only control shows, read from the icon's own name. Icon sets name the glyph in a
  // class (lucide-trash-2, fa-pencil, bi-x) or a <use> reference, and that is the only label such
  // a control has until its tooltip opens.
  const iconName = e => {
    for (const svg of e.querySelectorAll('svg,i,span[class*="icon"]')) {
      const titled=svg.querySelector?.('title')?.textContent?.trim() || svg.getAttribute('aria-label');
      if (titled) return titled;
      const use=svg.querySelector?.('use')?.getAttribute('href')?.split('#').pop();
      const classes=(svg.getAttribute('class')||'').split(/\s+/);
      const glyph=use || classes.map(c=>c.match(/^(?:lucide|fa|bi|icon|ti|ri|mdi|heroicon)-(?!solid$|regular$|light$|duotone$|outline$)(.+)$/)?.[1])
        .filter(Boolean).pop();
      if (glyph) return glyph.replace(/[-_]+/g,' ').replace(/\s\d+$/,'').trim()+' icon';
    }
    return '';
  };
  // `inner` is set while reading a control's content. A control that contains other controls is
  // named by its own content, not theirs, at any depth: a card holding Manage, Open and Share
  // buttons is the card, and those are offered on their own.
  const name = (e,seen=new Set(),top=true,inner=false) => {
    if (!e || seen.has(e)) return '';
    seen.add(e);
    const root=e.getRootNode?.() || document;
    const referenced=(e.getAttribute('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(root.getElementById?.(id) || document.getElementById(id),seen,false)).filter(Boolean).join(' ');
    const content=top || inner;
    const own = n => !(content && n.nodeType===1 && n.matches(selector));
    return referenced || e.getAttribute('aria-label') ||
      [...(e.labels||[])].map(l=>name(l,seen,false)).filter(Boolean).join(' ') ||
      (['button','submit','reset'].includes(e.type) ? e.value : '') || e.getAttribute('alt') ||
      (e.tagName==='INPUT' ? '' : [...e.childNodes].filter(own).map(n=>n.nodeType===3 ? n.textContent :
        n.nodeType===1 && n.getAttribute('aria-hidden')!=='true' ? name(n,seen,false,content) : '')
        .join(' ').replace(/\s+/g,' ').trim()) ||
      e.getAttribute('title') || e.getAttribute('placeholder') ||
      e.getAttribute('data-tooltip') || e.getAttribute('data-tip') || e.getAttribute('data-title') ||
      (top ? iconName(e) : '');
  };
  const role = e => {
    const explicit=e.getAttribute('role');
    if (selectorRoles.includes(explicit)) return explicit;
    if (e.tagName==='BUTTON' || e.tagName==='SUMMARY') return 'button';
    if (e.tagName==='A') return 'link';
    if (e.tagName==='SELECT') return 'combobox';
    if (e.tagName==='TEXTAREA' || e.isContentEditable) return 'textbox';
    if (e.tagName==='INPUT') {
      if (['checkbox','radio'].includes(e.type)) return e.type;
      if (['button','submit','reset','image'].includes(e.type)) return 'button';
      if (e.type==='search') return 'searchbox';
      if (e.type==='number') return 'spinbutton';
      if (['text','email','url','tel'].includes(e.type)) return 'textbox';
    }
    return null;
  };
  // The nearest ancestor that scrolls on its own, short of the document. A control below the fold
  // of such a panel cannot be reached by scrolling the page, which is the only scroll offered, so
  // it is offered directly and brought into view when used.
  const panel = e => {
    for (let n=e.parentElement; n && n!==document.body && n!==document.documentElement; n=n.parentElement) {
      const style=getComputedStyle(n);
      if (/(auto|scroll|overlay)/.test(style.overflowY+style.overflowX) &&
          (n.scrollHeight>n.clientHeight+1 || n.scrollWidth>n.clientWidth+1)) return n;
    }
    return null;
  };
  const inView = r => {
    const x=r.x+r.width/2, y=r.y+r.height/2;
    return x>=0 && y>=0 && x<innerWidth && y<innerHeight;
  };
  // Below the fold but close: a page scroll away at most. Further than this is left to scrolling,
  // where it is read with the text around it.
  const NEAR=2*innerHeight;
  // In view means in the window and inside the panel that clips it: a control under a panel's
  // fold has a place in the window and is drawn nowhere.
  const clipped = (e,r) => {
    const box=panel(e);
    if (!box) return false;
    const p=box.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
    return x<p.left || x>p.right || y<p.top || y>p.bottom;
  };
  const reachable = (e,r) => {
    if (inView(r) && !clipped(e,r)) return 'view';
    const box=panel(e);
    if (box) {
      const p=box.getBoundingClientRect();
      if (p.width>0 && p.height>0 && p.bottom>0 && p.top<innerHeight && p.right>0 && p.left<innerWidth) return 'panel';
    }
    return r.top>=innerHeight && r.top<innerHeight+NEAR && r.left>=0 && r.left<innerWidth ? 'below' : null;
  };
  const formFields = () => deep('input,textarea,select').filter(safe);
  cache.pageKey=()=>[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    formFields().map(e=>[identity(e),e.value,e.checked,e.selectedIndex,e.disabled,e.readOnly])];
  cache.guard=e=>{
    if (!e?.isConnected || !shown(e)) return null;
    const scope=across(e,'form,dialog,[role="dialog"],article,li,tr,[role="row"]') || e.parentElement;
    return [identity(e),role(e),name(e),e.value??null,e.checked??null,e.selectedIndex??null,
      e.readOnly??null,e.matches(':disabled'),e.getAttribute('aria-disabled'),
      e.getAttribute('aria-expanded'),e.getAttribute('aria-checked'),e.getAttribute('aria-selected'),
      e.getAttribute('href'),scope?.innerText?.slice(0,6000)||''];
  };
  const actions=[], later=[];
  for (const e of deep(selector)) {
    if (!safe(e) || !shown(e) || e.matches(':disabled') || across(e,'[aria-disabled="true"]')) continue;
    const r=e.getBoundingClientRect(), rname=role(e);
    if (!rname || r.width<=0 || r.height<=0) continue;
    const where=reachable(e,r);
    if (!where) continue;
    if (rname==='gridcell' && e.querySelector('button,[role="button"]')) continue;
    const base={node:identity(e),role:rname,label:name(e)||rname,
      rect:{x:r.x,y:r.y,w:r.width,h:r.height}};
    if (hoverReveals(e)) base.hover=true;
    if (where!=='view') base.offscreen=where;
    for (const key of ['checked','selected','expanded']) {
      const value=e.getAttribute('aria-'+key);
      if (value!==null) base[key]=value;
    }
    if (['checkbox','radio'].includes(e.type)) base.checked=String(e.checked);
    // On-screen controls first, so the cap drops the ones a scroll would reach instead.
    const into = where==='view' ? actions : later;
    if (e.tagName==='SELECT') {
      for (const o of e.options) if (!o.selected && !o.disabled && !o.closest('optgroup[disabled]'))
        into.push({...base,kind:'select',value:o.value,
          current_value:[...e.selectedOptions].map(o=>o.label).join(', '),label:base.label+' → '+o.label});
    } else {
      const editable=!e.readOnly && e.getAttribute('aria-readonly')!=='true' &&
        (['textbox','searchbox','spinbutton'].includes(rname) ||
          (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
      const value='value' in e ? String(e.value) :
        e.isContentEditable || rname==='combobox' ? e.innerText.trim() : '';
      into.push({...base,kind:editable?'fill':'click',value});
      if (editable) into.push({...base,kind:'click',value,label:'Open '+base.label});
    }
  }
  // Things that can be dragged, and places they can be dropped. Sources announce themselves:
  // draggable for native drag and drop, and the roledescription that dnd-kit and its kin set. A
  // drop zone rarely says it is one, so the named containers and the sources themselves stand in.
  const draggable = e => e.getAttribute('draggable')==='true' ||
    /^(draggable|sortable)$/i.test(e.getAttribute('aria-roledescription')||'') ||
    e.hasAttribute('data-rbd-drag-handle-draggable-id') || e.hasAttribute('data-rfd-drag-handle-draggable-id');
  const zoneSelector='[aria-dropeffect],[data-droppable],[data-rbd-droppable-id],[data-rfd-droppable-id],'+
    '[data-dropzone],[data-drop-target],[role="list"],[role="listbox"],[role="grid"],[role="region"],'+
    '[role="group"],[role="application"],[role="main"],main,section,ul,ol,.react-flow__pane';
  const heading = e => e.querySelector('h1,h2,h3,h4,[role="heading"]')?.innerText?.trim()?.split('\n')[0] || '';
  const sources=deep('[draggable="true"],[aria-roledescription],[data-rbd-drag-handle-draggable-id],[data-rfd-drag-handle-draggable-id]')
    .filter(e=>draggable(e) && visible(e) && inView(e.getBoundingClientRect())).slice(0,40);
  if (sources.length) {
    for (const e of sources) {
      const r=e.getBoundingClientRect();
      actions.push({node:identity(e),role:role(e)||'draggable',label:name(e)||heading(e)||'draggable item',
        rect:{x:r.x,y:r.y,w:r.width,h:r.height},kind:'drag',value:''});
    }
    // An element that handles drops says so in its handlers, not its markup. A plain handler is
    // a property; React keeps an element's props on the element, which is how an unlabelled canvas
    // that takes drops is found at all.
    const handlesDrop = e => !!e.ondrop || e.hasAttribute('ondrop') || Object.keys(e).some(key =>
      key.startsWith('__reactProps$') && (e[key]?.onDrop || e[key]?.onDragOver));
    const big = e => { const r=e.getBoundingClientRect(); return r.width>=80 && r.height>=60 && inView(r); };
    const handlers=deep('div,section,main,ul,ol,li,article').filter(e=>handlesDrop(e) && !sources.includes(e) && visible(e) && big(e));
    const zones=[...handlers, ...deep(zoneSelector).filter(e=>visible(e) && big(e))];
    // Unlabelled drop areas are named by what they hold, which is how a person would point at one.
    const described = e => {
      const text=(e.innerText||'').trim().split('\n').filter(Boolean).slice(0,2).join(' ').slice(0,50);
      return text ? 'drop area containing: '+text : 'empty drop area';
    };
    const zoned=new Set();
    for (const e of [...zones, ...sources]) {
      const label=(e.getAttribute('aria-label') || heading(e) ||
        (sources.includes(e) ? name(e) : handlers.includes(e) ? described(e) : '')).slice(0,80);
      if (!label || zoned.has(label) || zoned.size>=40) continue;
      zoned.add(label);
      const r=e.getBoundingClientRect();
      actions.push({node:identity(e),role:'drop zone',label,rect:{x:r.x,y:r.y,w:r.width,h:r.height},kind:'drop',value:''});
    }
  }
  // Controls that share a label - an Open on every card, a Remove on every row - are told apart by
  // what holds them: the control they sit in, or the nearest labelled row, item or card. Without
  // it the model is choosing between identical answers and can only guess which card it means.
  const named=new Map();
  const holder = e => {
    for (let n=e.parentElement || e.getRootNode()?.host; n && n!==document.body;
         n=n.parentElement || n.getRootNode()?.host) {
      const control=n.matches(selector) && shown(n);
      if (!control && !n.matches('li,tr,article,[role="row"],[role="listitem"],[role="article"],[role="group"],[aria-label]'))
        continue;
      // Every control in a card asks for the card's name, so it is worked out once.
      if (!named.has(n)) named.set(n, control ? name(n) : n.getAttribute('aria-label') || heading(n) || name(n));
      return named.get(n);
    }
    return '';
  };
  const counts={};
  for (const a of [...actions, ...later]) counts[a.label]=(counts[a.label]||0)+1;
  for (const a of [...actions, ...later]) {
    if (counts[a.label]<2 || a.kind==='drop') continue;
    const context=holder(cache.nodes.get(a.node)).slice(0,60);
    if (context && context!==a.label) a.label=a.label+' — '+context;
  }
  const words=[];
  let length=0;
  const range=document.createRange();
  for (const root of allRoots) {
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let node;
    while ((node=walker.nextNode()) && length<6000) {
      const value=node.textContent.trim(), parent=node.parentElement;
      if (!value || !parent || parent.closest('script,style,noscript,template') || !visible(parent)) continue;
      range.selectNodeContents(node); const r=range.getBoundingClientRect();
      if (r.width>0 && r.height>0 && r.bottom>0 && r.top<innerHeight && r.right>0 && r.left<innerWidth) {
        words.push(value); length+=value.length;
      }
    }
  }
  // Whether the page says it is still fetching what it will show: a busy region, a progress bar,
  // a skeleton, or a short "Loading…" line. A page shell is operable and still long before its
  // lists arrive, and a decision taken then answers BLOCKED about content that was on its way.
  const busy = deep('[aria-busy="true"],[role="progressbar"],[class*="skeleton"],[class*="shimmer"],[class*="animate-pulse"],[class*="animate-spin"],[class*="spinner"]')
    .filter(e=>{ const r=e.getBoundingClientRect(); return r.width>0 && r.height>0 && inView(r) && visible(e); }).length +
    words.filter(w=>w.length<=40 && /^(loading|fetching|please wait)\b/i.test(w)).length;
  const text=words.join('\n').slice(0,6000), height=document.documentElement.scrollHeight;
  actions.push(...later);
  const page_key=cache.pageKey(), guards={};
  for (const a of actions) if (!(a.node in guards)) guards[a.node]=cache.guard(cache.nodes.get(a.node));
  // Compare meaning and identity. Geometry is always resolved and hit-tested just before input.
  // Whether a control is showing because the pointer is over it is where the pointer is, not what
  // the page means, and it flickers as a row fades out under a pointer moving on.
  const semantics=actions.map(({rect,hover,...action})=>action);
  const marker=[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    document.title,text,semantics,page_key[6]];
  const omitted_actions=Math.max(0,actions.length-250);
  actions.splice(250);
  actions.forEach((a,i)=>a.id='e'+(i+1));
  if (scrollY+innerHeight<height-2) actions.push({id:'scroll_down',kind:'scroll',label:'Scroll down',delta:560});
  if (scrollY>0) actions.push({id:'scroll_up',kind:'scroll',label:'Scroll up',delta:-560});
  actions.push({id:'wait',kind:'wait',label:'Wait for the page to update'});
  return {url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,busy,
    scroll:{y:scrollY,height,view:innerHeight},actions,marker,page_key,guards,omitted_actions};
})()
