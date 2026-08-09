// Visual Editor - Scene-based regional prompt editor
// Uses inline event handlers (onclick) from HTML + global functions below

(function() {
  var VE_STATE = {};  // suffix -> { elements, selectedIdx, currentTool, dragState, nextId, activePopup }

  // Element type definitions (7 broad categories) - small default sizes
  var VE_ELEMS = {
    // 人物
    person:   { label: '人物', icon: '\uD83D\uDC64', category: '人物', maskShape: 'rect',      defaultW: 28, defaultH: 36 },
    // 物品
    cup:      { label: '杯子', icon: '\u2615',       category: '物品', maskShape: 'rect',      defaultW: 20, defaultH: 24 },
    plate:    { label: '盘子', icon: '\uD83C\uDF7D\uFE0F', category: '物品', maskShape: 'ellipse',   defaultW: 24, defaultH: 24 },
    book:     { label: '书本', icon: '\uD83D\uDCD6', category: '物品', maskShape: 'rect',      defaultW: 22, defaultH: 28 },
    pencil:   { label: '铅笔', icon: '\u270F\uFE0F', category: '物品', maskShape: 'rect',      defaultW: 16, defaultH: 28 },
    alarm:    { label: '闹钟', icon: '\u23F0',       category: '物品', maskShape: 'ellipse',   defaultW: 22, defaultH: 22 },
    key:      { label: '钥匙', icon: '\uD83D\uDD11', category: '物品', maskShape: 'rect',      defaultW: 24, defaultH: 14 },
    umbrella: { label: '雨伞', icon: '\u2602\uFE0F', category: '物品', maskShape: 'ellipse',   defaultW: 24, defaultH: 30 },
    backpack: { label: '背包', icon: '\uD83C\uDF92', category: '物品', maskShape: 'rect',      defaultW: 22, defaultH: 28 },
    table:    { label: '桌子', icon: '\uD83D\uDED1', category: '物品', maskShape: 'rect',      defaultW: 30, defaultH: 22 },
    chair:    { label: '椅子', icon: '\uD83D\uDED1', category: '物品', maskShape: 'rect',      defaultW: 20, defaultH: 26 },
    sofa:     { label: '沙发', icon: '\uD83D\uDECB\uFE0F', category: '物品', maskShape: 'rect',      defaultW: 32, defaultH: 22 },
    bed:      { label: '床',   icon: '\uD83D\uDECF\uFE0F', category: '物品', maskShape: 'rect',      defaultW: 34, defaultH: 26 },
    lamp:     { label: '台灯', icon: '\uD83D\uDCA1', category: '物品', maskShape: 'rect',      defaultW: 18, defaultH: 28 },
    bookshelf:{ label: '书架', icon: '\uD83D\uDCDA', category: '物品', maskShape: 'rect',      defaultW: 24, defaultH: 32 },
    wardrobe: { label: '衣柜', icon: '\uD83D\uDDC4\uFE0F', category: '物品', maskShape: 'rect',      defaultW: 24, defaultH: 34 },
    // 建筑
    building: { label: '房子', icon: '\uD83C\uDFE0', category: '建筑', maskShape: 'rect',      defaultW: 30, defaultH: 34 },
    castle:   { label: '城堡', icon: '\uD83C\uDFF0', category: '建筑', maskShape: 'rect',      defaultW: 32, defaultH: 36 },
    school:   { label: '学校', icon: '\uD83C\uDFEB', category: '建筑', maskShape: 'rect',      defaultW: 30, defaultH: 34 },
    shop:     { label: '商店', icon: '\uD83C\uDFEA', category: '建筑', maskShape: 'rect',      defaultW: 28, defaultH: 32 },
    bridge:   { label: '桥梁', icon: '\uD83C\uDF09', category: '建筑', maskShape: 'rect',      defaultW: 36, defaultH: 22 },
    lighthouse:{label: '灯塔', icon: '\uD83D\uDDFC', category: '建筑', maskShape: 'rect',      defaultW: 20, defaultH: 36 },
    windmill: { label: '风车', icon: '\uD83C\uDF2C\uFE0F', category: '建筑', maskShape: 'rect',      defaultW: 24, defaultH: 34 },
    tent:     { label: '帐篷', icon: '\u26FA',       category: '建筑', maskShape: 'triangle',  defaultW: 28, defaultH: 24 },
    // 交通
    car:      { label: '汽车', icon: '\uD83D\uDE97', category: '交通', maskShape: 'rect',      defaultW: 32, defaultH: 20 },
    bicycle:  { label: '自行车', icon: '\uD83D\uDEB2', category: '交通', maskShape: 'rect',      defaultW: 30, defaultH: 20 },
    plane:    { label: '飞机', icon: '\u2708\uFE0F', category: '交通', maskShape: 'rect',      defaultW: 34, defaultH: 18 },
    ship:     { label: '轮船', icon: '\uD83D\uDEA2', category: '交通', maskShape: 'rect',      defaultW: 32, defaultH: 22 },
    train:    { label: '火车', icon: '\uD83D\uDE82', category: '交通', maskShape: 'rect',      defaultW: 36, defaultH: 20 },
    rocket:   { label: '火箭', icon: '\uD83D\uDE80', category: '交通', maskShape: 'rect',      defaultW: 22, defaultH: 30 },
    balloon:  { label: '热气球', icon: '\uD83C\uDF88', category: '交通', maskShape: 'ellipse',   defaultW: 26, defaultH: 30 },
    // 天象
    rain:     { label: '雨',   icon: '\uD83C\uDF27\uFE0F', category: '天象', maskShape: 'ellipse',   defaultW: 28, defaultH: 24 },
    snow:     { label: '雪',   icon: '\u2744\uFE0F', category: '天象', maskShape: 'ellipse',   defaultW: 28, defaultH: 24 },
    rainbow:  { label: '彩虹', icon: '\uD83C\uDF08', category: '天象', maskShape: 'rect',      defaultW: 36, defaultH: 20 },
    lightning:{ label: '闪电', icon: '\u26A1',       category: '天象', maskShape: 'rect',      defaultW: 18, defaultH: 26 },
    star:     { label: '星星', icon: '\u2B50',       category: '天象', maskShape: 'ellipse',   defaultW: 20, defaultH: 20 },
    moon:     { label: '月亮', icon: '\uD83C\uDF19', category: '天象', maskShape: 'ellipse',   defaultW: 24, defaultH: 24 },
    meteor:   { label: '流星', icon: '\u2604\uFE0F', category: '天象', maskShape: 'rect',      defaultW: 28, defaultH: 14 },
    galaxy:   { label: '银河', icon: '\uD83C\uDF0C', category: '天象', maskShape: 'ellipse',   defaultW: 36, defaultH: 24 },
    // 山景
    river:    { label: '河流', icon: '\uD83C\uDF0A', category: '山景', maskShape: 'rect',      defaultW: 36, defaultH: 18 },
    lake:     { label: '湖泊', icon: '\uD83C\uDFDE\uFE0F', category: '山景', maskShape: 'ellipse',   defaultW: 34, defaultH: 26 },
    ocean:    { label: '海洋', icon: '\uD83C\uDF0A', category: '山景', maskShape: 'rect',      defaultW: 36, defaultH: 24 },
    waterfall:{ label: '瀑布', icon: '\uD83D\uDCA7', category: '山景', maskShape: 'rect',      defaultW: 22, defaultH: 32 },
    forest:   { label: '森林', icon: '\uD83C\uDF32', category: '山景', maskShape: 'ellipse',   defaultW: 34, defaultH: 30 },
    grass:    { label: '草地', icon: '\uD83C\uDF3F', category: '山景', maskShape: 'rect',      defaultW: 36, defaultH: 20 },
    desert:   { label: '沙漠', icon: '\uD83C\uDFDC\uFE0F', category: '山景', maskShape: 'rect',      defaultW: 36, defaultH: 22 },
    volcano:  { label: '火山', icon: '\uD83C\uDF0B', category: '山景', maskShape: 'triangle',  defaultW: 28, defaultH: 32 },
    island:   { label: '岛屿', icon: '\uD83C\uDFDD\uFE0F', category: '山景', maskShape: 'ellipse',   defaultW: 32, defaultH: 24 },
    cave:     { label: '洞穴', icon: '\uD83D\uDD73\uFE0F', category: '山景', maskShape: 'ellipse',   defaultW: 28, defaultH: 24 },
    // 武器
    sword:    { label: '剑',   icon: '\uD83D\uDDE1\uFE0F', category: '武器', maskShape: 'rect',      defaultW: 16, defaultH: 30 },
    shield:   { label: '盾牌', icon: '\uD83D\uDEE1\uFE0F', category: '武器', maskShape: 'ellipse',   defaultW: 22, defaultH: 24 },
    staff:    { label: '法杖', icon: '\uD83E\uDE84', category: '武器', maskShape: 'rect',      defaultW: 14, defaultH: 30 },
    bow:      { label: '弓箭', icon: '\uD83C\uDFF9', category: '武器', maskShape: 'rect',      defaultW: 26, defaultH: 22 },
    bomb:     { label: '炸弹', icon: '\uD83D\uDCA3', category: '武器', maskShape: 'ellipse',   defaultW: 22, defaultH: 22 },
    potion:   { label: '药水', icon: '\uD83E\uDDEA', category: '武器', maskShape: 'rect',      defaultW: 20, defaultH: 26 },
    chest:    { label: '宝箱', icon: '\uD83D\uDDC3\uFE0F', category: '武器', maskShape: 'rect',      defaultW: 24, defaultH: 20 },
    crown:    { label: '皇冠', icon: '\uD83D\uDC51', category: '武器', maskShape: 'rect',      defaultW: 24, defaultH: 18 },
    ring:     { label: '戒指', icon: '\uD83D\uDC8D', category: '武器', maskShape: 'ellipse',   defaultW: 20, defaultH: 20 },
  };

  // Category to element mapping (for random selection when clicking a category button)
  var VE_CAT_ITEMS = {
    '人物': ['person'],
    '物品': ['cup','plate','book','pencil','alarm','key','umbrella','backpack','table','chair','sofa','bed','lamp','bookshelf','wardrobe'],
    '建筑': ['building','castle','school','shop','bridge','lighthouse','windmill','tent'],
    '交通': ['car','bicycle','plane','ship','train','rocket','balloon'],
    '天象': ['rain','snow','rainbow','lightning','star','moon','meteor','galaxy'],
    '山景': ['river','lake','ocean','waterfall','forest','grass','desert','volcano','island','cave'],
    '武器': ['sword','shield','staff','bow','bomb','potion','chest','crown','ring'],
  };

  var VE_COLORS = [
    [255, 99, 71],   [60, 179, 113], [70, 130, 180], [238, 130, 238],
    [255, 215, 0],   [0, 206, 209],  [255, 140, 0],  [50, 205, 50],
    [147, 112, 219], [255, 105, 180],[0, 255, 127],  [72, 209, 204],
    [255, 182, 85],  [100, 149, 237],[240, 128, 128],[152, 251, 152],
    [175, 238, 238], [255, 160, 122],[218, 165, 150],[200, 200, 200]
  ];

  function veDetColor(index) {
    return VE_COLORS[index % VE_COLORS.length];
  }

  // ----- RoundRect polyfill for older browsers -----
  function veRoundRect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ----- Render function -----
  function veRender(suffix) {
    var st = VE_STATE[suffix];
    if (!st) return;
    var ctx = st.ctx, canvas = st.canvas, els = st.elements;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 3x3 grid (九宫格)
    var cw = canvas.width, ch = canvas.height;
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    for (var i = 1; i < 3; i++) {
      var gx = cw * i / 3;
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, ch); ctx.stroke();
      var gy = ch * i / 3;
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(cw, gy); ctx.stroke();
    }
    // Border
    ctx.strokeStyle = 'rgba(79,140,247,0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, cw, ch);

    // Aspect ratio label (top-left corner)
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(st.aspectRatio, 6, 6);
    ctx.restore();

    els.forEach(function(el, i) {
      var isSel = i === st.selectedIdx;
      var r = el.color[0], g = el.color[1], b = el.color[2];
      var ti = VE_ELEMS[el.type] || VE_ELEMS.person;
      var cx = el.x + el.w / 2, cy = el.y + el.h / 2;

      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,0.3)';
      ctx.shadowBlur = 8; ctx.shadowOffsetY = 2;
      var alpha = 0.30 + el.weight * 0.25;
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
      ctx.strokeStyle = isSel ? '#fff' : 'rgb(' + r + ',' + g + ',' + b + ')';
      ctx.lineWidth = isSel ? 2.5 : 1.5;
      ctx.beginPath();
      if (ti.maskShape === 'rect') { veRoundRect(ctx, el.x, el.y, el.w, el.h, 6); }
      else if (ti.maskShape === 'ellipse') { ctx.ellipse(cx, cy, el.w / 2, el.h / 2, 0, 0, Math.PI * 2); }
      else if (ti.maskShape === 'triangle') { ctx.moveTo(cx, el.y); ctx.lineTo(el.x + el.w, el.y + el.h); ctx.lineTo(el.x, el.y + el.h); ctx.closePath(); }
      ctx.fill(); ctx.stroke();
      ctx.restore();

      // Icon (emoji)
      ctx.save();
      var fontSize = Math.min(el.w, el.h) * 0.35;
      ctx.font = fontSize + 'px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(ti.icon, cx, cy);
      ctx.restore();

      // Label
      if (el.prompt) {
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(el.prompt.substring(0, 18), cx, el.y + el.h - 4);
        ctx.restore();
      }

      // Badge
      ctx.save();
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.8)';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(ti.icon + ' ' + ti.label, el.x + 4, el.y + 4);
      ctx.restore();

      // Selection handles
      if (isSel) {
        ctx.fillStyle = '#fff';
        var handles = [[el.x, el.y],[el.x+el.w,el.y],[el.x,el.y+el.h],[el.x+el.w,el.y+el.h],[el.x+el.w/2,el.y],[el.x+el.w/2,el.y+el.h],[el.x,el.y+el.h/2],[el.x+el.w,el.y+el.h/2]];
        handles.forEach(function(h){ ctx.beginPath(); ctx.arc(h[0], h[1], 4, 0, Math.PI * 2); ctx.fill(); });
      }
    });

    veUpdatePanel(suffix);
    veUpdateShapeList(suffix);
  }

  function veUpdatePanel(suffix) {
    var st = VE_STATE[suffix];
    if (!st) return;
    if (st.selectedIdx < 0 || st.selectedIdx >= st.elements.length) {
      if (st.noSel) st.noSel.style.display = 'block';
      if (st.selProps) st.selProps.style.display = 'none';
      return;
    }
    if (st.noSel) st.noSel.style.display = 'none';
    if (st.selProps) st.selProps.style.display = 'block';
    var el = st.elements[st.selectedIdx];
    var ti = VE_ELEMS[el.type] || VE_ELEMS.person;
    if (st.selColor) st.selColor.style.backgroundColor = 'rgb(' + el.color[0] + ',' + el.color[1] + ',' + el.color[2] + ')';
    if (st.selIcon) st.selIcon.textContent = ti.icon;
    if (st.selName) st.selName.textContent = ti.label;
    if (document.activeElement !== st.promptInput) st.promptInput.value = el.prompt || '';
    st.weightSlider.value = el.weight;
    if (st.weightVal) st.weightVal.textContent = el.weight.toFixed(1);
  }

  function veUpdateShapeList(suffix) {
    var st = VE_STATE[suffix];
    if (!st || !st.shapeList) return;
    if (st.elements.length === 0) {
      st.shapeList.innerHTML = '<div style="font-size:11px;color:#888;padding:8px 0;text-align:center;">点击上方元素添加到场景</div>';
      return;
    }
    st.shapeList.innerHTML = st.elements.map(function(el, i) {
      var ti = VE_ELEMS[el.type] || VE_ELEMS.person;
      var label = el.prompt ? el.prompt.substring(0, 14) : ti.label;
      return '<div class="ve-si' + (i === st.selectedIdx ? ' ve-sel' : '') + '" data-ve-idx="' + i + '">' +
        '<span class="ve-cd" style="background:rgb(' + el.color[0] + ',' + el.color[1] + ',' + el.color[2] + ')"></span>' +
        '<span class="ve-ei">' + ti.icon + '</span>' +
        '<span class="ve-el">' + label + '</span>' +
        '<span class="ve-ed" data-ve-del="' + i + '">\u2715</span></div>';
    }).join('');

    // Click on shape item
    st.shapeList.querySelectorAll('.ve-si').forEach(function(el) {
      el.addEventListener('click', function(e) {
        if (e.target.closest('.ve-ed')) return;
        st.selectedIdx = parseInt(el.dataset.veIdx);
        veRender(suffix);
      });
    });
    // Click on delete button
    st.shapeList.querySelectorAll('.ve-ed').forEach(function(el) {
      el.addEventListener('click', function(e) {
        e.stopPropagation();
        var idx = parseInt(el.dataset.veDel);
        st.elements.splice(idx, 1);
        if (st.selectedIdx === idx) st.selectedIdx = -1;
        else if (st.selectedIdx > idx) st.selectedIdx--;
        veRender(suffix);
      });
    });
  }

  // ----- Add element (supports category keys -> random element) -----
  function veAddElement(suffix, type, x, y) {
    var st = VE_STATE[suffix];
    if (!st) return;
    // If type is a category key, pick a random element from that category
    if (VE_CAT_ITEMS[type]) {
      var items = VE_CAT_ITEMS[type];
      type = items[Math.floor(Math.random() * items.length)];
    }
    var ti = VE_ELEMS[type] || VE_ELEMS.person;
    var w = ti.defaultW + (Math.random() - 0.5) * 30;
    var h = ti.defaultH + (Math.random() - 0.5) * 20;
    var ex = x - w / 2, ey = y - h / 2;
    // Ensure element stays within canvas
    ex = Math.max(0, Math.min(ex, st.canvas.width - w));
    ey = Math.max(0, Math.min(ey, st.canvas.height - h));
    var color = veDetColor(st.elements.length);
    st.elements.push({ id: st.nextId++, type: type, x: ex, y: ey, w: w, h: h, color: color, prompt: '', weight: 1.0 });
    st.selectedIdx = st.elements.length - 1;
    st.currentTool = 'select';
    st.canvas.style.cursor = 'default';
    if (st.hint) st.hint.textContent = '点击选中元素，拖动移动或调整大小';
    // Reset toolbar active state
    var toolbar = document.querySelector('#ve-canvas-wrap-' + suffix + ' .ve-toolbar');
    if (toolbar) {
      toolbar.querySelectorAll('[onclick]').forEach(function(b) { b.classList.remove('ve-active'); });
      var selBtn = toolbar.querySelector('[onclick*="select"]');
      if (selBtn) selBtn.classList.add('ve-active');
    }
    veRender(suffix);
  }

  // ----- Element hit test -----
  function veGetElementAt(suffix, mx, my) {
    var st = VE_STATE[suffix];
    if (!st) return -1;
    for (var i = st.elements.length - 1; i >= 0; i--) {
      var el = st.elements[i];
      if (mx >= el.x && mx <= el.x + el.w && my >= el.y && my <= el.y + el.h) return i;
    }
    return -1;
  }

  function veGetHandleAt(suffix, mx, my) {
    var st = VE_STATE[suffix];
    if (!st || st.selectedIdx < 0) return -1;
    var el = st.elements[st.selectedIdx];
    var handles = [
      {x:el.x,y:el.y},{x:el.x+el.w,y:el.y},{x:el.x,y:el.y+el.h},{x:el.x+el.w,y:el.y+el.h},
      {x:el.x+el.w/2,y:el.y},{x:el.x+el.w/2,y:el.y+el.h},{x:el.x,y:el.y+el.h/2},{x:el.x+el.w,y:el.y+el.h/2}
    ];
    for (var i = 0; i < handles.length; i++) {
      var dx = mx - handles[i].x, dy = my - handles[i].y;
      if (dx * dx + dy * dy < 36) return i;
    }
    return -1;
  }

  // ----- Canvas mouse handlers -----
  function veGetCanvasPos(suffix, e) {
    var st = VE_STATE[suffix];
    if (!st) return {x:0,y:0};
    var rect = st.canvas.getBoundingClientRect();
    var cx = e.touches ? e.touches[0].clientX : e.clientX;
    var cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (cx - rect.left) * (st.canvas.width / rect.width), y: (cy - rect.top) * (st.canvas.height / rect.height) };
  }

  function veCanvasDown(e, suffix) {
    var st = VE_STATE[suffix];
    if (!st) return;
    var pos = veGetCanvasPos(suffix, e);
    var hi = veGetHandleAt(suffix, pos.x, pos.y);
    if (st.currentTool !== 'select') {
      veAddElement(suffix, st.currentTool, pos.x, pos.y);
      return;
    }
    if (hi >= 0 && st.selectedIdx >= 0) {
      var el = st.elements[st.selectedIdx];
      st.dragState = { idx: st.selectedIdx, startX: pos.x, startY: pos.y, origX: el.x, origY: el.y, origW: el.w, origH: el.h, type: 'resize', handle: hi };
      return;
    }
    var idx = veGetElementAt(suffix, pos.x, pos.y);
    if (idx >= 0) {
      st.selectedIdx = idx;
      var el = st.elements[idx];
      st.dragState = { idx: idx, startX: pos.x, startY: pos.y, origX: el.x, origY: el.y, origW: el.w, origH: el.h, type: 'move' };
      veRender(suffix);
    } else {
      st.selectedIdx = -1;
      veRender(suffix);
    }
  }

  function veCanvasMove(e, suffix) {
    var st = VE_STATE[suffix];
    if (!st || !st.dragState) return;
    e.preventDefault();
    var pos = veGetCanvasPos(suffix, e);
    var dx = pos.x - st.dragState.startX, dy = pos.y - st.dragState.startY;
    var el = st.elements[st.dragState.idx];
    if (st.dragState.type === 'move') {
      el.x = st.dragState.origX + dx; el.y = st.dragState.origY + dy;
    } else if (st.dragState.type === 'resize') {
      var h = st.dragState.handle;
      var nx = st.dragState.origX, ny = st.dragState.origY, nw = st.dragState.origW, nh = st.dragState.origH;
      if (h === 0) { nx = st.dragState.origX + dx; ny = st.dragState.origY + dy; nw = st.dragState.origW - dx; nh = st.dragState.origH - dy; }
      else if (h === 1) { ny = st.dragState.origY + dy; nw = st.dragState.origW + dx; nh = st.dragState.origH - dy; }
      else if (h === 2) { nx = st.dragState.origX + dx; nw = st.dragState.origW - dx; nh = st.dragState.origH + dy; }
      else if (h === 3) { nw = st.dragState.origW + dx; nh = st.dragState.origH + dy; }
      else if (h === 4) { ny = st.dragState.origY + dy; nh = st.dragState.origH - dy; }
      else if (h === 5) { nh = st.dragState.origH + dy; }
      else if (h === 6) { nx = st.dragState.origX + dx; nw = st.dragState.origW - dx; }
      else if (h === 7) { nw = st.dragState.origW + dx; }
      if (nw >= 20) { el.x = nx; el.w = nw; }
      if (nh >= 20) { el.y = ny; el.h = nh; }
    }
    veRender(suffix);
  }

  function veCanvasUp(e, suffix) {
    var st = VE_STATE[suffix];
    if (st && st.dragState) { st.dragState = null; veRender(suffix); }
  }

  // ----- Helper: set Gradio Textbox value via actual input element -----
  function veSetTextInput(elemId, val) {
    var wrapper = document.getElementById(elemId);
    console.log('[VE] veSetTextInput:', elemId, 'wrapper:', !!wrapper);
    if (!wrapper) return;
    var input = wrapper.querySelector('input[data-testid="textbox"], textarea[data-testid="textbox"]');
    console.log('[VE] input found:', !!input, 'tag:', input ? input.tagName : 'N/A');
    if (!input) return;
    
    var nativeSetter = Object.getOwnPropertyDescriptor(
      input.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype,
      'value'
    );
    if (!nativeSetter || !nativeSetter.set) {
      input.value = val;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
      return;
    }
    
    // Step 1: Clear first to force Gradio to detect the change (value: '' -> actual value)
    nativeSetter.set.call(input, '');
    input.dispatchEvent(new Event('input', { bubbles: true }));
    
    // Step 2: Set the actual value
    nativeSetter.set.call(input, val);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    console.log('[VE] value set, length:', val.length);
    
    // Step 3: Also try setting via Svelte internal if available
    if (wrapper.__svelte && wrapper.__svelte.component) {
      wrapper.__svelte.component.$set({ value: val });
      console.log('[VE] svelte set OK');
    }
    
    // Step 4: Trigger change on wrapper to notify Gradio Block component
    wrapper.dispatchEvent(new Event('change', { bubbles: true }));
  }

  // ----- Generate mask -----
  function veGenerateMask(suffix) {
    var st = VE_STATE[suffix];
    st._maskVersion = (st._maskVersion || 0) + 1;
    console.log('[VE] veGenerateMask called, suffix:', suffix, 'version:', st._maskVersion, 'elements:', st ? st.elements.length : 0);
    if (!st || st.elements.length === 0) return;
    var maskW = 512, maskH = 512;
    
    // Log current element data for debugging
    console.log('[VE] Current elements:');
    st.elements.forEach(function(el, idx) {
      console.log('  [' + idx + '] type=' + el.type + ' pos=(' + Math.round(el.x) + ',' + Math.round(el.y) + ') size=(' + Math.round(el.w) + 'x' + Math.round(el.h) + ') prompt="' + (el.prompt || '') + '" weight=' + el.weight + ' color=' + JSON.stringify(el.color));
    });
    
    // Group elements by grid cell (3x3)
    var cellMap = {};
    console.log('[VE] Canvas size:', st.canvas.width, 'x', st.canvas.height, 'ratio:', st.aspectRatio);
    st.elements.forEach(function(el) {
      var cx = el.x + el.w / 2, cy = el.y + el.h / 2;
      var cellCol = Math.min(2, Math.floor((cx / st.canvas.width) * 3));
      var cellRow = Math.min(2, Math.floor((cy / st.canvas.height) * 3));
      var cellKey = cellRow + '-' + cellCol;
      console.log('[VE] Element:', el.type, 'center=(' + Math.round(cx) + ',' + Math.round(cy) + ')', 'cell=' + cellKey, 'color=' + JSON.stringify(el.color));
      if (!cellMap[cellKey]) {
        cellMap[cellKey] = { elements: [], color: el.color, row: cellRow, col: cellCol };
      }
      cellMap[cellKey].elements.push(el);
    });

    var offscreen = document.createElement('canvas');
    offscreen.width = maskW; offscreen.height = maskH;
    var octx = offscreen.getContext('2d');
    octx.fillStyle = '#ffffff';
    octx.fillRect(0, 0, maskW, maskH);

    var cellKeys = Object.keys(cellMap);
    console.log('[VE] cellKeys order:', cellKeys);
    cellKeys.forEach(function(cellKey) {
      var cell = cellMap[cellKey];
      var r = cell.color[0], g = cell.color[1], b = cell.color[2];
      var cellLeft = (cell.col / 3) * maskW;
      var cellTop = (cell.row / 3) * maskH;
      var cellRight = ((cell.col + 1) / 3) * maskW;
      var cellBottom = ((cell.row + 1) / 3) * maskH;
      octx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      octx.fillRect(cellLeft, cellTop, cellRight - cellLeft, cellBottom - cellTop);
    });

    // Build prompts per cell (combine prompts if multiple elements in same cell)
    var cellPrompts = [];
    cellKeys.forEach(function(cellKey) {
      var cell = cellMap[cellKey];
      var combinedPrompt = cell.elements.map(function(el) { return el.prompt || ''; }).filter(function(s) { return s; }).join(', ');
      cellPrompts.push({ prompt: combinedPrompt, weight: cell.elements[0].weight, type: cell.elements[0].type, color: cell.color });
    });

    var dataUrl = offscreen.toDataURL('image/png');
    console.log('[VE] mask generated, version:', st._maskVersion, 'cells:', cellKeys.length, 'dataUrl length:', dataUrl.length);
    console.log('[VE] prompts JSON:', JSON.stringify(cellPrompts));
    
    // Reset: clear old data first, then set new data
    // This ensures Gradio detects the change even if the data is similar
    veSetTextInput('ve-mask-output-' + suffix, dataUrl);
    veSetTextInput('ve-prompts-output-' + suffix, JSON.stringify(cellPrompts));
    
    // Verify after 100ms
    setTimeout(function() {
      var m = document.getElementById('ve-mask-output-' + suffix);
      if (m) {
        var inp = m.querySelector('input[data-testid="textbox"]');
        console.log('[VE] verify mask field:', inp ? 'input found, value length=' + (inp.value ? inp.value.length : 0) : 'no input');
      }
      var p = document.getElementById('ve-prompts-output-' + suffix);
      if (p) {
        var inp2 = p.querySelector('input[data-testid="textbox"]');
        console.log('[VE] verify prompts field:', inp2 ? 'input found, value=' + (inp2.value ? inp2.value.substring(0, 60) : 'EMPTY') : 'no input');
      }
    }, 100);
    // Visual feedback
    var btn = st.genBtn;
    if (btn) {
      var origText = btn.textContent;
      btn.textContent = '\u2713 已应用 (' + cellKeys.length + ' 区域)';
      btn.style.background = '#27ae60';
      setTimeout(function() {
        btn.textContent = origText;
        btn.style.background = '';
      }, 1500);
    }
  }

  // ----- Resize canvas -----
  function veResizeCanvas(suffix) {
    var st = VE_STATE[suffix];
    if (!st) return;
    var cw = st.container.clientWidth;
    var ch = st.container.clientHeight;
    if (cw < 50 || ch < 50) return;
    var parts = st.aspectRatio.split(':').map(Number);
    var ar = parts[0] / parts[1];
    var pad = 16;
    var w, h;
    if (cw / ch > ar) {
      h = ch - pad;
      w = Math.round(h * ar);
    } else {
      w = cw - pad;
      h = Math.round(w / ar);
    }
    w = Math.max(w, 100); h = Math.max(h, 100);
    st.canvas.width = w;
    st.canvas.height = h;
    st.canvas.style.width = w + 'px';
    st.canvas.style.height = h + 'px';
    veRender(suffix);
  }

  // ===== GLOBAL FUNCTIONS (called from inline onclick) =====
  window.ve_clickTool = function(event, btn, suffix, tool) {
    var st = VE_STATE[suffix];
    if (!st) return;
    st.currentTool = tool;
    var toolbar = btn.parentElement;
    toolbar.querySelectorAll('button').forEach(function(b) { b.classList.remove('ve-active'); });
    btn.classList.add('ve-active');
    st.canvas.style.cursor = 'default';
    if (st.hint) st.hint.textContent = '点击选中元素，拖动移动或调整大小';
  };

  window.ve_delete = function(suffix) {
    var st = VE_STATE[suffix];
    if (!st) return;
    if (st.selectedIdx >= 0 && st.selectedIdx < st.elements.length) {
      st.elements.splice(st.selectedIdx, 1);
      st.selectedIdx = -1;
      veRender(suffix);
    }
  };

  window.ve_clear = function(suffix) {
    var st = VE_STATE[suffix];
    if (!st) return;
    if (st.elements.length > 0) {
      st.elements = [];
      st.selectedIdx = -1;
      veRender(suffix);
    }
  };

  // ----- Aspect ratio -----
  window.ve_setRatio = function(suffix, ratio, btn) {
    var st = VE_STATE[suffix];
    if (!st) return;
    st.aspectRatio = ratio;
    btn.parentElement.querySelectorAll('.ve-btn-ratio').forEach(function(b) { b.classList.remove('ve-active'); });
    btn.classList.add('ve-active');
    veResizeCanvas(suffix);
  };

  // ----- Snap to grid (3x3) -----
  function veSnapPos(val, cellSize) {
    var cell = Math.round(val / cellSize);
    return Math.max(0, cell * cellSize);
  }

  // ----- Direct category selection (click canvas to place random element from category) -----
  window.ve_selectElement = function(suffix, catKey, btnEl) {
    var st = VE_STATE[suffix];
    if (!st) return;
    st.currentTool = catKey;
    st.canvas.style.cursor = 'crosshair';
    // Highlight this button in toolbar
    var toolbar = btnEl.parentNode;
    toolbar.querySelectorAll('button').forEach(function(b) { b.classList.remove('ve-active'); });
    btnEl.classList.add('ve-active');
  };

  // ===== Initialize a single editor =====
  function veInitEditor(suffix) {
    if (VE_STATE[suffix] && VE_STATE[suffix].initialized) return;

    var container = document.getElementById('ve-canvas-wrap-' + suffix);
    var canvas = document.getElementById('ve-canvas-' + suffix);
    if (!container || !canvas) return;

    // Wait for container to be visible
    if (container.clientWidth === 0 || container.clientHeight === 0) {
      setTimeout(function() { veInitEditor(suffix); }, 300);
      return;
    }

    var ctx = canvas.getContext('2d');
    if (!ctx) return;

    var st = {
      initialized: true,
      suffix: suffix,
      container: container,
      canvas: canvas,
      ctx: ctx,
      elements: [],
      selectedIdx: -1,
      currentTool: 'select',
      aspectRatio: '1:1',
      dragState: null,
      nextId: 0,
      hint: document.getElementById('ve-hint-' + suffix),
      noSel: document.getElementById('ve-nosel-' + suffix),
      selProps: document.getElementById('ve-selprops-' + suffix),
      selColor: document.getElementById('ve-scd-' + suffix),
      selIcon: document.getElementById('ve-sic-' + suffix),
      selName: document.getElementById('ve-snm-' + suffix),
      promptInput: document.getElementById('ve-pt-' + suffix),
      weightSlider: document.getElementById('ve-ws-' + suffix),
      weightVal: document.getElementById('ve-wv-' + suffix),
      genBtn: document.getElementById('ve-gen-' + suffix),
      shapeList: document.getElementById('ve-sl-' + suffix),
    };

    // Check that critical elements exist
    if (!st.promptInput || !st.weightSlider) {
      setTimeout(function() { veInitEditor(suffix); }, 300);
      return;
    }

    VE_STATE[suffix] = st;

    // Set canvas size (via veResizeCanvas which handles aspect ratio)
    veResizeCanvas(suffix);

    // Canvas mouse events
    canvas.addEventListener('mousedown', function(e) { veCanvasDown(e, suffix); });
    canvas.addEventListener('mousemove', function(e) { veCanvasMove(e, suffix); });
    canvas.addEventListener('mouseup', function(e) { veCanvasUp(e, suffix); });
    canvas.addEventListener('mouseleave', function(e) { veCanvasUp(e, suffix); });
    canvas.addEventListener('touchstart', function(e) { e.preventDefault(); veCanvasDown(e, suffix); }, { passive: false });
    canvas.addEventListener('touchmove', function(e) { e.preventDefault(); veCanvasMove(e, suffix); }, { passive: false });
    canvas.addEventListener('touchend', function(e) { e.preventDefault(); veCanvasUp(e, suffix); }, { passive: false });

    // Prompt input
    st.promptInput.addEventListener('input', function() {
      if (st.selectedIdx >= 0 && st.selectedIdx < st.elements.length) {
        st.elements[st.selectedIdx].prompt = st.promptInput.value;
        veRender(suffix);
      }
    });

    // Weight slider
    st.weightSlider.addEventListener('input', function() {
      if (st.selectedIdx >= 0 && st.selectedIdx < st.elements.length) {
        st.elements[st.selectedIdx].weight = parseFloat(st.weightSlider.value);
        if (st.weightVal) st.weightVal.textContent = st.weightSlider.value;
        veRender(suffix);
      }
    });

    // Generate button
    if (st.genBtn) {
      st.genBtn.addEventListener('click', function() { veGenerateMask(suffix); });
    }

    // Resize observer
    var ro = new ResizeObserver(function() { veResizeCanvas(suffix); });
    ro.observe(container);

    // Initial render
    veRender(suffix);
  }

  // ===== Watch for accordion open to retry initialization =====
  function veWatchAccordion() {
    // Find all Gradio accordions that contain our editor
    var accordions = document.querySelectorAll('[id^="RP_main"]');
    accordions.forEach(function(acc) {
      // Look for open attribute changes
      var mo = new MutationObserver(function() {
        var wrappers = acc.querySelectorAll('[id^="ve-canvas-wrap-"]');
        wrappers.forEach(function(w) {
          var suffix = w.id.replace('ve-canvas-wrap-', '');
          if (!VE_STATE[suffix] || !VE_STATE[suffix].initialized) {
            veInitEditor(suffix);
          } else {
            veResizeCanvas(suffix);
          }
        });
      });
      mo.observe(acc, { attributes: true, attributeFilter: ['open', 'class', 'style'] });
    });
  }

  // ===== Poll for new editors =====
  function vePoll() {
    var wrappers = document.querySelectorAll('[id^="ve-canvas-wrap-"]');
    wrappers.forEach(function(w) {
      var suffix = w.id.replace('ve-canvas-wrap-', '');
      if (!VE_STATE[suffix] || !VE_STATE[suffix].initialized) {
        veInitEditor(suffix);
      }
    });
    veWatchAccordion();
    setTimeout(vePoll, 1000);
  }

  // Start
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', vePoll);
  } else {
    vePoll();
  }
})();