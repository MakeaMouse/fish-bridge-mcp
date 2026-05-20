// fish_bridge graph viewer — Cytoscape.js interaction layer
// Loaded after cytoscape.min.js

const NODE_COLORS = {
  question: '#ed8936',  // orange
  decision: '#4299e1',  // blue
  concept:  '#a0aec0',  // grey
  skill:    '#9f7aea',  // purple
  file:     '#38b2ac',  // teal
  error:    '#f56565',  // red
  task:     '#48bb78',  // green
};

const NODE_SHAPES = {
  question: 'ellipse',
  decision: 'diamond',
  concept:  'round-rectangle',
  skill:    'hexagon',
  file:     'rectangle',
  error:    'star',
  task:     'round-triangle',
};

const STATUS_COLORS = {
  active:      '#e2e8f0',
  in_progress: '#f59e0b',
  adopted:     '#22c55e',
  proposed:    '#38bdf8',
  pending:     '#67e8f9',
  blocked:     '#ef4444',
  conflicted:  '#eab308',
  resolved:    '#4ade80',
  done:        '#4ade80',
  fixed:       '#4ade80',
  deferred:    '#64748b',
  unconfirmed: '#6b7280',
  superseded:  '#94a3b8',
};
let pollInterval;
let _currentNodeId = null;   // tracks which node has the detail/edit panel open
let _activeSession = null;   // currently viewed session id (null = server default)

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  cy = cytoscape({
    container: document.getElementById('cy'),
    style: buildStyle(),
    elements: [],
    layout: { name: 'preset' },
    wheelSensitivity: 0.3,
  });

  cy.on('tap', 'node', e => showDetail(e.target));
  cy.on('tap', evt => { if (evt.target === cy) closeDetail(); });

  // Wire filter checkboxes
  document.querySelectorAll('.type-filter, .status-filter').forEach(cb => {
    cb.addEventListener('change', applyFilters);
  });

  fetchSessions();
  fetchGraph();
  // Poll every 3 seconds for live updates (graph data + session list)
  pollInterval = setInterval(() => { fetchGraph(); fetchSessions(); }, 3000);
});

// ---------------------------------------------------------------------------
// Session switcher (B1)
// ---------------------------------------------------------------------------

function fetchSessions() {
  fetch('/api/sessions')
    .then(r => r.json())
    .then(sessions => {
      if (!sessions || sessions.length <= 1) return;  // only show if >1 session
      const sel = document.getElementById('session-select');
      const currentVal = sel.value;  // preserve selection across refreshes
      sel.innerHTML = '';
      sessions.forEach(s => {
        // Support both old string format and new object format {id, title?, nodes, updated}
        const sid   = (typeof s === 'object') ? s.id : s;
        const name  = (typeof s === 'object' && s.title) ? s.title : sid;
        const label = (typeof s === 'object')
          ? `${name}  (${s.nodes} nodes · ${s.updated})`
          : s;
        const opt = document.createElement('option');
        opt.value = sid;
        opt.textContent = label;
        sel.appendChild(opt);
      });
      // Restore previously-selected session (or the active one on first load)
      if (currentVal) {
        sel.value = currentVal;
      } else if (_activeSession) {
        sel.value = _activeSession;
      }
      document.getElementById('session-switcher').style.display = 'flex';
    })
    .catch(() => {});  // no sessions endpoint = older server, hide switcher
}

function switchSession(sid) {
  _activeSession = sid;
  fetchGraph();
}

// ---------------------------------------------------------------------------
// Data fetch
// ---------------------------------------------------------------------------

function fetchGraph() {
  const url = _activeSession
    ? `/api/graph?session=${encodeURIComponent(_activeSession)}`
    : '/api/graph';
  fetch(url)
    .then(r => r.json())
    .then(data => {
      updateGraph(data);
      // Sync the dropdown to whatever the server reports as the active session
      if (data.session_id && !_activeSession) {
        _activeSession = data.session_id;
        const sel = document.getElementById('session-select');
        if (sel) sel.value = data.session_id;
      }
    })
    .catch(() => {
      // Silently ignore fetch errors (server may be starting up)
    });
}

function updateGraph(data) {
  const session = data.session_id || 'unknown';
  const nodes   = data.nodes || [];
  const edges   = data.edges || [];

  document.getElementById('session-label').textContent = session;
  document.getElementById('stats').textContent =
    `${nodes.length} nodes · ${edges.length} edges`;

  // Build Cytoscape elements
  const cyNodes = nodes.map(n => ({
    group: 'nodes',
    data: {
      id:         n.id,
      label:      n.label,
      type:       n.type,
      status:     n.status,
      summary:    n.summary || '',
      confidence: n.confidence || 1.0,
    },
  }));

  const cyEdges = edges.map(e => ({
    group: 'edges',
    data: {
      id:       e.id,
      source:   e.from_id,
      target:   e.to_id,
      relation: e.relation,
      weight:   e.weight || 1.0,
    },
  }));

  allElements = [...cyNodes, ...cyEdges];

  // Only re-render if data changed (compare node counts)
  const currentCount = cy.nodes().length;
  if (currentCount !== cyNodes.length) {
    cy.elements().remove();
    cy.add(allElements);
    runLayout('cose');
  }

  applyFilters();
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

// Sort order used by grid layout to cluster same-type nodes together,
// which keeps connected nodes nearby and reduces edge crossings.
const _TYPE_SORT_ORDER  = { question:0, decision:1, task:2, error:3, skill:4, concept:5, file:6 };
const _STATUS_SORT_ORDER = {
  active:0, in_progress:1, adopted:2, proposed:3, pending:4,
  blocked:5, conflicted:6, unconfirmed:7, resolved:8, done:9, fixed:10, deferred:11,
};

function _gridSort(a, b) {
  const ta = _TYPE_SORT_ORDER[a.data('type')]   ?? 99;
  const tb = _TYPE_SORT_ORDER[b.data('type')]   ?? 99;
  if (ta !== tb) return ta - tb;
  return (_STATUS_SORT_ORDER[a.data('status')] ?? 99)
       - (_STATUS_SORT_ORDER[b.data('status')] ?? 99);
}

function runLayout(name) {
  const options = {
    // Force-directed — good general-purpose layout
    cose: {
      name: 'cose',
      animate: false,
      nodeRepulsion: 9000,
      idealEdgeLength: 90,
      gravity: 1.0,
      nodeDimensionsIncludeLabels: true,
      padding: 20,
    },

    // Hierarchical tree — directed edges flow top-to-bottom.
    // spacingFactor: 1.5 keeps same-row nodes readable without stretching the
    // canvas so wide that inter-row vertical gaps appear enormous.
    // nodeDimensionsIncludeLabels is kept so labels don't clip neighbours,
    // but we rely on avoidOverlap + padding rather than aggressive spacingFactor.
    // NOTE: layout follows edge topology (parent→child), NOT chat turn order.
    // NOTE: maximal:true crashes on disconnected graphs — omitted.
    breadthfirst: {
      name: 'breadthfirst',
      directed: true,
      spacingFactor: 1.5,
      nodeDimensionsIncludeLabels: true,
      avoidOverlap: true,
      animate: true,
      animationDuration: 350,
      padding: 24,
    },

    // Grid — nodes sorted by type then status so same-type nodes are
    // adjacent, keeping connected nodes close and reducing edge crossings.
    grid: {
      name: 'grid',
      avoidOverlap: true,
      avoidOverlapPadding: 32,
      nodeDimensionsIncludeLabels: true,
      spacingFactor: 1.5,
      condense: false,
      sort: _gridSort,
      animate: true,
      animationDuration: 350,
      padding: 24,
    },

    // Circle — space nodes so labels don't clip each other
    circle: {
      name: 'circle',
      nodeDimensionsIncludeLabels: true,
      spacingFactor: 1.4,
      animate: true,
      animationDuration: 350,
      padding: 24,
    },

    preset: { name: 'preset' },
  };

  try {
    cy.layout(options[name] || options.cose).run();
  } catch (e) {
    console.warn(`Layout "${name}" failed (${e.message}), falling back to CoSE`);
    cy.layout(options.cose).run();
  }
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

function applyFilters() {
  const activeTypes   = new Set([...document.querySelectorAll('.type-filter:checked')].map(cb => cb.value));
  const activeStatuses = new Set([...document.querySelectorAll('.status-filter:checked')].map(cb => cb.value));

  cy.nodes().forEach(node => {
    const d = node.data();
    const visible = activeTypes.has(d.type) && activeStatuses.has(d.status);
    if (visible) node.removeClass('hidden-node'); else node.addClass('hidden-node');
  });

  // Hide edges where either endpoint is hidden
  cy.edges().forEach(edge => {
    const srcHidden = edge.source().hasClass('hidden-node');
    const tgtHidden = edge.target().hasClass('hidden-node');
    if (srcHidden || tgtHidden) edge.addClass('hidden-edge'); else edge.removeClass('hidden-edge');
  });
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

function buildStyle() {
  const nodeStyles = Object.entries(NODE_COLORS).map(([type, color]) => ({
    selector: `node[type="${type}"]`,
    style: {
      'background-color': color,
      'shape': NODE_SHAPES[type] || 'ellipse',
    },
  }));

  return [
    // ── Base node style ──────────────────────────────────────────────────
    {
      selector: 'node',
      style: {
        'label':            'data(label)',
        'font-size':        '11px',
        'color':            '#e0e0e0',
        'text-wrap':        'wrap',
        'text-max-width':   '110px',
        'text-valign':      'bottom',
        'text-halign':      'center',
        'width':            '44px',
        'height':           '44px',
        'border-width':     '3px',
        'border-color':     '#e2e8f0',   // active default — bright white ring
        'background-opacity': 'data(confidence)',
      },
    },
    // ── Per-status border colours (most specific wins) ───────────────────
    {
      selector: 'node[status="active"]',
      style: { 'border-color': '#e2e8f0', 'border-width': 3, 'border-style': 'solid' },
    },
    {
      selector: 'node[status="in_progress"]',
      style: { 'border-color': '#f59e0b', 'border-width': 3, 'border-style': 'solid' },
    },
    {
      selector: 'node[status="adopted"]',
      style: { 'border-color': '#22c55e', 'border-width': 3, 'border-style': 'solid' },
    },
    {
      selector: 'node[status="proposed"]',
      style: { 'border-color': '#38bdf8', 'border-width': 2, 'border-style': 'solid' },
    },
    {
      selector: 'node[status="pending"]',
      style: { 'border-color': '#67e8f9', 'border-width': 2, 'border-style': 'solid' },
    },
    {
      selector: 'node[status="blocked"]',
      style: { 'border-color': '#ef4444', 'border-width': 4, 'border-style': 'solid' },
    },
    {
      selector: 'node[status="conflicted"]',
      style: { 'border-color': '#eab308', 'border-width': 4, 'border-style': 'solid' },
    },
    {
      selector: 'node[status="unconfirmed"]',
      style: { 'border-color': '#6b7280', 'border-width': 2, 'border-style': 'dashed', 'background-opacity': 0.6 },
    },
    // Terminal — muted opacity, dashed border
    {
      selector: 'node[status="done"], node[status="resolved"], node[status="fixed"]',
      style: {
        'border-color':       '#4ade80',
        'border-width':       2,
        'border-style':       'dashed',
        'background-opacity': 0.30,
        'color':              '#94a3b8',
      },
    },
    {
      selector: 'node[status="deferred"]',
      style: {
        'border-color':       '#64748b',
        'border-width':       2,
        'border-style':       'dashed',
        'background-opacity': 0.22,
        'color':              '#64748b',
      },
    },
    // ── Type-based colour + shape (applied after status so shape is preserved) ──
    ...nodeStyles,
    // ── Selected ────────────────────────────────────────────────────────
    {
      selector: 'node:selected',
      style: {
        'border-color': '#ffffff',
        'border-width': 4,
        'border-style': 'solid',
      },
    },
    {
      selector: 'node.hidden-node',
      style: { 'display': 'none' },
    },
    // ── Edges ───────────────────────────────────────────────────────────
    {
      selector: 'edge',
      style: {
        'width':              1.5,
        'line-color':         '#334155',
        'target-arrow-color': '#334155',
        'target-arrow-shape': 'triangle',
        'curve-style':        'bezier',
        'label':              'data(relation)',
        'font-size':          '9px',
        'color':              '#64748b',
        'text-rotation':      'autorotate',
        'text-background-color':   '#1a1a2e',
        'text-background-opacity': 0.8,
        'text-background-padding': '2px',
      },
    },
    {
      selector: 'edge.hidden-edge',
      style: { 'display': 'none' },
    },
    {
      selector: 'edge:selected',
      style: { 'line-color': '#e94560', 'target-arrow-color': '#e94560' },
    },
  ];
}

// ---------------------------------------------------------------------------
// Detail panel
// ---------------------------------------------------------------------------

function showDetail(node) {
  const d = node.data();
  _currentNodeId = d.id;

  const typeColor  = NODE_COLORS[d.type]  || '#a0aec0';
  const statusColor = STATUS_COLORS[d.status] || '#e2e8f0';

  document.getElementById('detail-label').textContent = d.label;
  document.getElementById('detail-type').innerHTML =
    `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${typeColor};margin-right:5px;vertical-align:middle"></span>${d.type}`;
  document.getElementById('detail-status').innerHTML =
    `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;border:2px solid ${statusColor};margin-right:5px;vertical-align:middle"></span>${d.status}`;
  document.getElementById('detail-summary').textContent    = d.summary || '(no summary)';
  document.getElementById('detail-confidence').textContent = `Confidence: ${(d.confidence * 100).toFixed(0)}%`;
  document.getElementById('detail-panel').classList.remove('hidden');
}

function closeDetail() {
  document.getElementById('detail-panel').classList.add('hidden');
  closeEdit();
  _currentNodeId = null;
}

// ---------------------------------------------------------------------------
// Edit panel
// ---------------------------------------------------------------------------

function openEdit() {
  if (!_currentNodeId) return;
  const node = cy.getElementById(_currentNodeId);
  if (!node || node.empty()) return;
  const d = node.data();

  document.getElementById('edit-label').textContent   = d.label;
  document.getElementById('edit-status').value        = d.status || 'active';
  document.getElementById('edit-summary').value       = d.summary || '';
  document.getElementById('edit-status-msg').textContent = '';

  document.getElementById('detail-panel').classList.add('hidden');
  document.getElementById('edit-panel').classList.remove('hidden');
}

function closeEdit() {
  document.getElementById('edit-panel').classList.add('hidden');
  // Re-show detail panel if a node is still selected
  if (_currentNodeId) {
    document.getElementById('detail-panel').classList.remove('hidden');
  }
}

function saveEdit() {
  if (!_currentNodeId) return;

  const newStatus  = document.getElementById('edit-status').value;
  const newSummary = document.getElementById('edit-summary').value;
  const msgEl      = document.getElementById('edit-status-msg');

  msgEl.textContent = 'Saving…';
  msgEl.style.color = '#a0aec0';

  fetch(`/api/node/${encodeURIComponent(_currentNodeId)}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ status: newStatus, summary: newSummary }),
  })
    .then(r => r.json())
    .then(resp => {
      if (resp.ok) {
        msgEl.style.color = '#22c55e';
        msgEl.textContent = 'Saved!';
        // Update the Cytoscape node in-memory so filters/colours update immediately
        const node = cy.getElementById(_currentNodeId);
        if (node && !node.empty()) {
          node.data('status',  newStatus);
          node.data('summary', newSummary);
        }
        // Refresh detail panel text
        document.getElementById('detail-status').textContent  = `Status: ${newStatus}`;
        document.getElementById('detail-summary').textContent = newSummary || '(no summary)';
        applyFilters();
        // Close edit panel after a short delay
        setTimeout(() => closeEdit(), 900);
      } else {
        msgEl.style.color = '#f56565';
        msgEl.textContent = `Error: ${resp.error || 'unknown'}`;
      }
    })
    .catch(err => {
      msgEl.style.color = '#f56565';
      msgEl.textContent = `Network error: ${err.message}`;
    });
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

function exportPNG() {
  const png = cy.png({ scale: 2, bg: '#1a1a2e' });
  const a = document.createElement('a');
  a.href = png;
  a.download = 'fish_bridge_graph.png';
  a.click();
}
