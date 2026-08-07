FLOW_RE = re.compile(r"^Flow_(\d+)_(\d+)_t-(\d+)$")
INJ_RE  = re.compile(r"^Inj_(\d+)_t-(\d+)$")

def build_index_maps(columns):
    flows, inj = {}, {}
    max_lag = 0
    for c in columns:
        m = FLOW_RE.match(c)
        if m:
            i, j, lag = int(m.group(1)), int(m.group(2)), int(m.group(3))
            max_lag = max(max_lag, lag)
            flows.setdefault((i, j), {})[lag] = c
            continue
        m = INJ_RE.match(c)
        if m:
            i, lag = int(m.group(1)), int(m.group(2))
            max_lag = max(max_lag, lag)
            inj.setdefault(i, {})[lag] = c
    T = max_lag + 1
    return flows, inj, T

def extract_time_series_row(row, flows_map, inj_map, T, edge_list_1based, n_buses):
    node_ts = np.zeros((n_buses, T), dtype=np.float32)
    for bus1 in range(1, n_buses + 1):
        for lag in range(T):
            col = inj_map.get(bus1, {}).get(lag, None)
            if col is not None:
                node_ts[bus1 - 1, T - 1 - lag] = float(row[col])

    edge_ts = np.zeros((len(edge_list_1based), T), dtype=np.float32)
    for eidx, (i, j) in enumerate(edge_list_1based):
        fmap = flows_map.get((i, j), None) or flows_map.get((j, i), None)
        if fmap:
            for lag in range(T):
                col = fmap.get(lag, None)
                if col is not None:
                    edge_ts[eidx, T - 1 - lag] = float(row[col])

    return node_ts, edge_ts
