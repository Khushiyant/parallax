"""Optional visualization helpers for PRLX traces.

All heavy dependencies (matplotlib, plotly) are imported lazily so that
``import prlx`` never pulls them in.
"""

from typing import Optional

from .trace_reader import TraceData, EVENT_NAMES


def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def plot_warp_timeline(
    trace: TraceData,
    max_warps: int = 32,
):
    """Scatter plot of events by warp, colored by event type.

    Returns a matplotlib ``Figure``.
    """
    _mpl, plt = _require_matplotlib()

    warps = trace.warps()[:max_warps]

    fig, ax = plt.subplots(figsize=(14, max(4, len(warps) * 0.3)))

    color_map = {
        0: "#e6b800",  # Branch, yellow
        1: "#cc44cc",  # SharedMemStore, magenta
        2: "#4488cc",  # Atomic, blue
        3: "#44cc44",  # FuncEntry, green
        4: "#88cc88",  # FuncExit, light green
        5: "#cc8844",  # Switch, orange
        6: "#cc4444",  # GlobalStore, red
    }

    for w in warps:
        for i, ev in enumerate(w.events):
            color = color_map.get(ev.event_type, "#888888")
            ax.scatter(i, w.warp_idx, c=color, s=8, marker="s", edgecolors="none")

    # Legend
    handles = []
    import matplotlib.patches as mpatches
    for etype, name in sorted(EVENT_NAMES.items()):
        handles.append(mpatches.Patch(color=color_map.get(etype, "#888"), label=name))
    ax.legend(handles=handles, loc="upper right", fontsize=7)

    ax.set_xlabel("Event index")
    ax.set_ylabel("Warp index")
    ax.set_title(f"Warp timeline, {trace.header.kernel_name}")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_divergence_heatmap(
    trace_a: TraceData,
    trace_b: TraceData,
    max_warps: int = 64,
):
    """Heatmap of per-warp, per-site divergence counts between two traces.

    Compares event counts at each ``(warp_idx, site_id)`` between traces.
    Returns a matplotlib ``Figure``.
    """
    _mpl, plt = _require_matplotlib()
    import numpy as np

    n_warps = min(trace_a.num_warps, trace_b.num_warps, max_warps)

    # Collect all site_ids
    site_set: set = set()
    for w in trace_a.warps()[:n_warps]:
        for ev in w.events:
            site_set.add(ev.site_id)
    for w in trace_b.warps()[:n_warps]:
        for ev in w.events:
            site_set.add(ev.site_id)

    sites = sorted(site_set)
    if not sites:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        return fig

    site_to_col = {s: i for i, s in enumerate(sites)}
    heatmap = np.zeros((n_warps, len(sites)), dtype=int)

    for wi in range(n_warps):
        wa = trace_a.warps()[wi] if wi < trace_a.num_warps else None
        wb = trace_b.warps()[wi] if wi < trace_b.num_warps else None

        counts_a: dict = {}
        counts_b: dict = {}
        if wa:
            for ev in wa.events:
                counts_a[ev.site_id] = counts_a.get(ev.site_id, 0) + 1
        if wb:
            for ev in wb.events:
                counts_b[ev.site_id] = counts_b.get(ev.site_id, 0) + 1

        for sid in sites:
            diff = abs(counts_a.get(sid, 0) - counts_b.get(sid, 0))
            heatmap[wi, site_to_col[sid]] = diff

    fig, ax = plt.subplots(figsize=(max(6, len(sites) * 0.4), max(4, n_warps * 0.2)))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Site index")
    ax.set_ylabel("Warp index")
    ax.set_title("Divergence heatmap (event count diff)")
    fig.colorbar(im, ax=ax, label="Abs diff")
    fig.tight_layout()
    return fig


def display_trace_summary(trace: TraceData) -> str:
    """Return an HTML table summarising the trace.

    Works in Jupyter notebooks (via ``IPython.display.HTML``) or returns
    the raw HTML string for other use.
    """
    s = trace.summary()
    rows = [
        ("Kernel", s["kernel_name"]),
        ("Grid dim", f"{s['grid_dim']}"),
        ("Block dim", f"{s['block_dim']}"),
        ("Warps", str(s["num_warps"])),
        ("Total events", str(s["total_events"])),
        ("Total overflows", str(s["total_overflows"])),
    ]
    for etype, count in sorted(s["events_by_type"].items()):
        rows.append((f"  {etype}", str(count)))

    html = "<table style='border-collapse:collapse;font-family:monospace;'>\n"
    html += "<tr><th style='text-align:left;padding:2px 8px;border-bottom:1px solid #ccc;'>Field</th>"
    html += "<th style='text-align:left;padding:2px 8px;border-bottom:1px solid #ccc;'>Value</th></tr>\n"
    for label, value in rows:
        html += (
            f"<tr><td style='padding:2px 8px;'>{label}</td>"
            f"<td style='padding:2px 8px;'>{value}</td></tr>\n"
        )
    html += "</table>"

    # Try to display in Jupyter
    try:
        from IPython.display import HTML, display
        display(HTML(html))
    except ImportError:
        pass

    return html
